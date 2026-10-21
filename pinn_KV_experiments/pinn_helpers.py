"""Shared KdV PINN utilities: u_t + 6 u u_x + beta u_xxx = 0.

A traveling single soliton supplies initial/boundary data and an exact reference.
Boundary constraints are u at both ends and u_x at the right end.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Avoid JAX reserving most accelerator memory up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap
from functools import partial


# -----------------------------------------------------------------------------
# Default experiment configuration
# -----------------------------------------------------------------------------
SEED = 5
SPEED = 1.0
BETA = 1.0
X0 = -2.0
X_MIN, X_MAX = -10.0, 10.0
T_MIN, T_MAX = 0.0, 1.0

NETWORK_SIZE = (2, 64, 164, 64, 1)
N_COEFFS = 9

OUTER_STEPS = 8000
INNER_STEPS = 40
LR_PARAMS = 1e-4
LR_COEFFS = 5e-4

N_F_TRAIN = 512
N_IC_TRAIN = 128
N_BC_TRAIN = 128
N_F_VAL = 256
N_IC_VAL = 64
N_BC_VAL = 64

LAMBDA_F = 1.0
LAMBDA_IC = 5.0
LAMBDA_BC = 5.0
PRINT_EVERY = 50

NX_REF = 256
NT_REF = 1001
PRED_BATCH_SIZE = 8192
KAPPA = 3.0
PATH = Path(__file__).resolve().parent / "results"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def exact_solution(x, t, speed=SPEED, beta=BETA, x0=X0):
    """Single soliton for positive speed and dispersion beta."""
    phase = 0.5 * np.sqrt(speed / beta) * (np.asarray(x) - speed * np.asarray(t) - x0)
    return 0.5 * speed / np.cosh(phase)**2


def exact_solution_x(x, t):
    phase = 0.5 * np.sqrt(SPEED / BETA) * (np.asarray(x) - SPEED * np.asarray(t) - X0)
    return -np.sqrt(SPEED / BETA) * exact_solution(x, t) * np.tanh(phase)


def initial_condition(x):
    return exact_solution(x, T_MIN)


def make_pinn_points(
    n_f: int,
    n_ic: int,
    n_bc: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create fixed training/validation points for one KdV realization."""
    rng = np.random.default_rng(seed)

    x_f = rng.uniform(X_MIN, X_MAX, (n_f, 1))
    t_f = rng.uniform(T_MIN, T_MAX, (n_f, 1))
    X_f = np.hstack([x_f, t_f])

    x_ic = rng.uniform(X_MIN, X_MAX, (n_ic, 1))
    X_ic = np.hstack([x_ic, np.full_like(x_ic, T_MIN)])
    y_ic = initial_condition(x_ic)

    # n_bc points are generated on each side.
    t_bc = rng.uniform(T_MIN, T_MAX, (n_bc, 1))
    X_left = np.hstack([np.full_like(t_bc, X_MIN), t_bc])
    X_right = np.hstack([np.full_like(t_bc, X_MAX), t_bc])
    X_bc = np.vstack([X_left, X_right])
    y_bc = exact_solution(X_bc[:, :1], X_bc[:, 1:])
    y_bc_x = exact_solution_x(X_right[:, :1], X_right[:, 1:])

    return {
        "X_f": X_f.astype(np.float32),
        "X_ic": X_ic.astype(np.float32),
        "y_ic": y_ic.astype(np.float32),
        "X_bc": X_bc.astype(np.float32),
        "y_bc": y_bc.astype(np.float32),
        "X_bc_x": X_right.astype(np.float32),
        "y_bc_x": y_bc_x.astype(np.float32),
    }


def to_jax_batch(data: dict[str, np.ndarray]):
    return tuple(jnp.asarray(data[k]) for k in ("X_f", "X_ic", "y_ic", "X_bc", "y_bc", "X_bc_x", "y_bc_x"))


def fixed_datasets():
    """Same points are returned by all three experiment scripts."""
    train = make_pinn_points(N_F_TRAIN, N_IC_TRAIN, N_BC_TRAIN, seed=SEED)
    val = make_pinn_points(N_F_VAL, N_IC_VAL, N_BC_VAL, seed=SEED + 1)
    return train, val


# -----------------------------------------------------------------------------
# Exact KdV reference (no numerical discretization error)
# -----------------------------------------------------------------------------
def reference_solution(nx: int = NX_REF, nt: int = NT_REF):
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    U = exact_solution(x[None, :], t[:, None])
    return x.astype(np.float32), t.astype(np.float32), U.astype(np.float32)


# -----------------------------------------------------------------------------
# Parameter initialization
# -----------------------------------------------------------------------------
def init_network_params(sizes=NETWORK_SIZE, seed: int = 42):
    """Xavier-normal initialization, identical across the three experiments."""
    keys = random.split(random.PRNGKey(seed), len(sizes) - 1)
    params = []
    for m, n, key in zip(sizes[:-1], sizes[1:], keys):
        std = np.sqrt(2.0 / (m + n))
        w = std * random.normal(key, (m, n))
        b = jnp.zeros((n,))
        params.append((w, b))
    return params


def init_activation_coeffs(sizes=NETWORK_SIZE, n_coeffs: int = N_COEFFS, key = jax.random.key(42), scale=0.5):

    keys = random.split(key, len(sizes) - 2)

    coeffs = []

    for width, k in zip(sizes[1:-1], keys):

        k1, k2 = random.split(k)

        c = scale * random.normal(
            k1,
            (width, n_coeffs)
        )

        # Mesma inicializacao aleatoria dos experimentos de Burgers.
        # Na ordem de jnp.polyval, c[:, 1] multiplica y**3.
        c = c.at[:, 1].set(
            1.0 + scale * random.normal(
                k2,
                (width,)
            )
        )

        coeffs.append(c)

    return coeffs


def tree_sgd(tree, grads, lr):
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, tree, grads)


# -----------------------------------------------------------------------------
# Forward models
# -----------------------------------------------------------------------------
def normalize_inputs(X):
    """Scale physical coordinates; autodiff retains the chain-rule factors."""
    lower = jnp.array([X_MIN, T_MIN])
    upper = jnp.array([X_MAX, T_MAX])
    return 2.0 * (X - lower) / (upper - lower) - 1.0


def forward_tanh(X, params):
    h = normalize_inputs(X)
    for i, (w, b) in enumerate(params):
        z = h @ w + b
        h = jnp.tanh(z) if i < len(params) - 1 else z
    return h


def pau_scalar(y, c):
    split = (c.shape[0] + 1) // 2
    P = jnp.polyval(c[:split], y)
    Q = jnp.polyval(c[split:], y)
    return P / (1.0 + jnp.abs(Q))


def smpaf_scalar(y, c, kappa: float = KAPPA):
    split = (c.shape[0] + 1) // 2
    P = jnp.polyval(c[:split], y)
    Q = jnp.polyval(c[split:], y)
    return P / (1.0 + jnp.sqrt(Q**2 + kappa))


def forward_adaptive(X, params, coeffs, activation: str):
    h = normalize_inputs(X)
    for i, (w, b) in enumerate(params):
        z = h @ w + b
        if i < len(params) - 1:
            c = coeffs[i]
            if activation == "pau":
                h = vmap(pau_scalar, in_axes=(1, 0), out_axes=1)(z, c)
            elif activation == "smpaf":
                h = vmap(smpaf_scalar, in_axes=(1, 0), out_axes=1)(z, c)
            else:
                raise ValueError(f"Unknown adaptive activation: {activation}")
        else:
            h = z
    return h


# -----------------------------------------------------------------------------
# PINN losses: tanh
# -----------------------------------------------------------------------------
def u_tanh(params, z):
    return forward_tanh(z[None, :], params)[0, 0]


def residual_tanh_single(params, z, beta):
    u = u_tanh(params, z)
    du = grad(u_tanh, argnums=1)(params, z)
    ux, ut = du[0], du[1]

    def ux_fn(zz):
        return grad(u_tanh, argnums=1)(params, zz)[0]

    def uxx_fn(zz):
        return grad(ux_fn)(zz)[0]

    uxxx = grad(uxx_fn)(z)[0]
    return ut + 6.0 * u * ux + beta * uxxx


residual_tanh = vmap(residual_tanh_single, in_axes=(None, 0, None))


def tanh_loss_components(params, batch, beta=BETA):
    Xf, Xic, yic, Xbc, ybc, Xbc_x, ybc_x = batch
    r = residual_tanh(params, Xf, beta)
    lf = jnp.mean(r**2)
    lic = jnp.mean((forward_tanh(Xic, params) - yic) ** 2)
    ux_bc = vmap(lambda z: grad(u_tanh, argnums=1)(params, z)[0])(Xbc_x)
    lbc = (jnp.mean((forward_tanh(Xbc, params) - ybc) ** 2)
           + jnp.mean((ux_bc[:, None] - ybc_x) ** 2))
    total = LAMBDA_F * lf + LAMBDA_IC * lic + LAMBDA_BC * lbc
    return total, (lf, lic, lbc)


def tanh_loss(params, batch, beta=BETA):
    return tanh_loss_components(params, batch, beta)[0]


@partial(jit, static_argnames=("inner_steps",))
def update_tanh(params, batch, lr=LR_PARAMS, inner_steps=INNER_STEPS):
    def body(_, theta):
        g = grad(tanh_loss)(theta, batch, BETA)
        return tree_sgd(theta, g, lr)
    return jax.lax.fori_loop(0, inner_steps, body, params)


# -----------------------------------------------------------------------------
# PINN losses: PAU / smPAF
# -----------------------------------------------------------------------------
def u_adaptive(params, coeffs, z, activation: str):
    return forward_adaptive(z[None, :], params, coeffs, activation)[0, 0]


def residual_adaptive_single(params, coeffs, z, beta, activation: str):
    def u_of_z(zz):
        return u_adaptive(params, coeffs, zz, activation)

    u = u_of_z(z)
    du = grad(u_of_z)(z)
    ux, ut = du[0], du[1]

    def ux_of_z(zz):
        return grad(u_of_z)(zz)[0]

    def uxx_of_z(zz):
        return grad(ux_of_z)(zz)[0]

    uxxx = grad(uxx_of_z)(z)[0]
    return ut + 6.0 * u * ux + beta * uxxx


def adaptive_residual(params, coeffs, Xf, beta, activation: str):
    fn = lambda z: residual_adaptive_single(params, coeffs, z, beta, activation)
    return vmap(fn)(Xf)


def adaptive_loss_components(params, coeffs, batch, activation: str, beta=BETA):
    Xf, Xic, yic, Xbc, ybc, Xbc_x, ybc_x = batch
    r = adaptive_residual(params, coeffs, Xf, beta, activation)
    lf = jnp.mean(r**2)
    lic = jnp.mean((forward_adaptive(Xic, params, coeffs, activation) - yic) ** 2)
    ux_bc = vmap(lambda z: grad(lambda zz: u_adaptive(params, coeffs, zz, activation))(z)[0])(Xbc_x)
    lbc = (jnp.mean((forward_adaptive(Xbc, params, coeffs, activation) - ybc) ** 2)
           + jnp.mean((ux_bc[:, None] - ybc_x) ** 2))
    total = LAMBDA_F * lf + LAMBDA_IC * lic + LAMBDA_BC * lbc
    return total, (lf, lic, lbc)


def adaptive_loss(params, coeffs, batch, activation: str, beta=BETA):
    return adaptive_loss_components(params, coeffs, batch, activation, beta)[0]


def make_adaptive_update(activation: str):
    """Create a memory-conscious first-order bilevel update.

    theta: one update for PAU; inner_steps updates for smPAF on train points.
    coefficients: one validation-loss update with theta treated as fixed.

    This intentionally avoids differentiating through the full unrolled inner
    loop, which is considerably more memory intensive for a third-order PINN.
    """
    @partial(jit, static_argnames=("inner_steps",))
    def update(params, coeffs, train_batch, val_batch,
               lrp=LR_PARAMS, lrc=LR_COEFFS, inner_steps=INNER_STEPS):
        def body(_, theta):
            g = grad(adaptive_loss, argnums=0)(theta, coeffs, train_batch, activation, BETA)
            return tree_sgd(theta, g, lrp)

        effective_inner_steps = 1 if activation == "pau" else inner_steps
        new_params = jax.lax.fori_loop(0, effective_inner_steps, body, params)
        frozen_params = jax.lax.stop_gradient(new_params)
        gc = grad(adaptive_loss, argnums=1)(
            frozen_params, coeffs, val_batch, activation, BETA
        )
        new_coeffs = tree_sgd(coeffs, gc, lrc)
        return new_params, new_coeffs

    return update


# -----------------------------------------------------------------------------
# Evaluation and persistence
# -----------------------------------------------------------------------------
def predict_batched(forward_fn, X: np.ndarray, batch_size: int = PRED_BATCH_SIZE):
    chunks = []
    for start in range(0, len(X), batch_size):
        xb = jnp.asarray(X[start:start + batch_size])
        chunks.append(np.asarray(forward_fn(xb)))
    return np.vstack(chunks)


def evaluate_tanh(params):
    x, t, U_ref = reference_solution()
    T, X = np.meshgrid(t, x, indexing="ij")
    Xin = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    pred = predict_batched(lambda xb: forward_tanh(xb, params), Xin)
    U_pred = pred.reshape(len(t), len(x)).astype(np.float32)
    rel_l2 = float(np.linalg.norm(U_pred - U_ref) / np.linalg.norm(U_ref))
    return x, t, U_ref, U_pred, rel_l2


def evaluate_adaptive(params, coeffs, activation: str):
    x, t, U_ref = reference_solution()
    T, X = np.meshgrid(t, x, indexing="ij")
    Xin = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    pred = predict_batched(lambda xb: forward_adaptive(xb, params, coeffs, activation), Xin)
    U_pred = pred.reshape(len(t), len(x)).astype(np.float32)
    rel_l2 = float(np.linalg.norm(U_pred - U_ref) / np.linalg.norm(U_ref))
    return x, t, U_ref, U_pred, rel_l2


def empty_history():
    return {"step": [], "train_total": [], "train_pde": [], "train_ic": [], "train_bc": [],
            "val_total": [], "val_pde": [], "val_ic": [], "val_bc": []}


def record(history, step, train_values, val_values):
    history["step"].append(step)
    for prefix, values in (("train", train_values), ("val", val_values)):
        total, lf, lic, lbc = [float(v) for v in values]
        history[f"{prefix}_total"].append(total)
        history[f"{prefix}_pde"].append(lf)
        history[f"{prefix}_ic"].append(lic)
        history[f"{prefix}_bc"].append(lbc)


def components_to_tuple(result):
    total, (lf, lic, lbc) = result
    return total, lf, lic, lbc


def params_to_npz(params, prefix="param"):
    out = {}
    for i, (w, b) in enumerate(params):
        out[f"{prefix}_W_{i}"] = np.asarray(w)
        out[f"{prefix}_b_{i}"] = np.asarray(b)
    return out


def coeffs_to_npz(coeffs):
    return {f"coeffs_{i}": np.asarray(c) for i, c in enumerate(coeffs)}


def save_result(
    path: str | Path,
    model_name: str,
    history: dict,
    train_data: dict,
    val_data: dict,
    params,
    x_grid,
    t_grid,
    U_ref,
    U_pred,
    relative_l2: float,
    elapsed_seconds: float,
    coeffs=None,
):
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "seed": SEED,
        "equation": "u_t + 6*u*u_x + beta*u_xxx = 0",
        "speed": SPEED,
        "x0": X0,
        "x_domain": [X_MIN, X_MAX],
        "t_domain": [T_MIN, T_MAX],
        "reference": "exact_single_soliton",
        "boundary_conditions": "u(left,t), u(right,t), ux(right,t)",
        "beta": float(BETA),
        "network_size": list(NETWORK_SIZE),
        "outer_steps": OUTER_STEPS,
        "inner_steps": 1 if model_name == "pau" else INNER_STEPS,
        "lr_params": LR_PARAMS,
        "lr_coeffs": LR_COEFFS if coeffs is not None else None,
        "lambda_f": LAMBDA_F,
        "lambda_ic": LAMBDA_IC,
        "lambda_bc": LAMBDA_BC,
        "kappa": KAPPA if model_name == "smpaf" else None,
        "adaptive_update": (
            "first_order_bilevel"
            if coeffs is not None
            else None
        ),
    }

    payload = {
        "model_name": np.array(model_name),
        "config_json": np.array(json.dumps(config)),
        "relative_l2": np.array(relative_l2, dtype=np.float64),
        "elapsed_seconds": np.array(elapsed_seconds, dtype=np.float64),
        "x_grid": x_grid,
        "t_grid": t_grid,
        "U_reference": U_ref,
        "U_prediction": U_pred,
    }

    payload.update({
        k: np.asarray(v)
        for k, v in history.items()
    })

    payload.update({
        f"train_{k}": np.asarray(v)
        for k, v in train_data.items()
    })

    payload.update({
        f"val_{k}": np.asarray(v)
        for k, v in val_data.items()
    })

    payload.update(params_to_npz(params))

    if coeffs is not None:
        payload.update(coeffs_to_npz(coeffs))

    np.savez_compressed(path, **payload)

    print(f"Saved results to: {path}")


def print_progress(model, step, tr, va):
    trt, trf, tric, trbc = tr
    vat, vaf, vaic, vabc = va
    print(
        f"[{model}] step={step:5d} | "
        f"train={float(trt):.3e} (PDE={float(trf):.3e}, IC={float(tric):.3e}, BC={float(trbc):.3e}) | "
        f"val={float(vat):.3e}"
    )
