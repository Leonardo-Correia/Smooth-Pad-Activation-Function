"""Shared utilities for the Burgers PINN experiments.

Each experiment solves the same fixed problem

    u_t + u u_x - nu u_xx = 0,
    u(x,0) = -alpha sin(pi x),
    u(-1,t) = u(1,t) = 0,

with alpha fixed by the experiment configuration.  The neural network input is
therefore only (x,t).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

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
ALPHA = 1.0
NU = 0.01 / np.pi
X_MIN, X_MAX = -1.0, 1.0
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
PATH = '/home/pcleonardo/pade_AF/pinn_burgers_experiments/results/'


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def initial_condition(x: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    return -alpha * np.sin(np.pi * x)


def make_pinn_points(
    n_f: int,
    n_ic: int,
    n_bc: int,
    seed: int,
    alpha: float = ALPHA,
) -> dict[str, np.ndarray]:
    """Create fixed training/validation points for one Burgers realization."""
    rng = np.random.default_rng(seed)

    x_f = rng.uniform(X_MIN, X_MAX, (n_f, 1))
    t_f = rng.uniform(T_MIN, T_MAX, (n_f, 1))
    X_f = np.hstack([x_f, t_f])

    x_ic = rng.uniform(X_MIN, X_MAX, (n_ic, 1))
    X_ic = np.hstack([x_ic, np.zeros_like(x_ic)])
    y_ic = initial_condition(x_ic, alpha)

    # n_bc points are generated on each side.
    t_bc = rng.uniform(T_MIN, T_MAX, (n_bc, 1))
    X_left = np.hstack([np.full_like(t_bc, X_MIN), t_bc])
    X_right = np.hstack([np.full_like(t_bc, X_MAX), t_bc])
    X_bc = np.vstack([X_left, X_right])
    y_bc = np.zeros((2 * n_bc, 1), dtype=np.float32)

    return {
        "X_f": X_f.astype(np.float32),
        "X_ic": X_ic.astype(np.float32),
        "y_ic": y_ic.astype(np.float32),
        "X_bc": X_bc.astype(np.float32),
        "y_bc": y_bc.astype(np.float32),
    }


def to_jax_batch(data: dict[str, np.ndarray]):
    return tuple(jnp.asarray(data[k]) for k in ("X_f", "X_ic", "y_ic", "X_bc", "y_bc"))


def fixed_datasets():
    """Same points are returned by all three experiment scripts."""
    train = make_pinn_points(N_F_TRAIN, N_IC_TRAIN, N_BC_TRAIN, seed=SEED, alpha=ALPHA)
    val = make_pinn_points(N_F_VAL, N_IC_VAL, N_BC_VAL, seed=SEED + 1, alpha=ALPHA)
    return train, val


# -----------------------------------------------------------------------------
# Reference Burgers solver (stable explicit Rusanov + central diffusion)
# -----------------------------------------------------------------------------
def solve_burgers_rusanov(
    nx: int = NX_REF,
    nt: int = NT_REF,
    nu: float = NU,
    alpha: float = ALPHA,
):
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    stability = abs(alpha) * dt / dx + 2.0 * nu * dt / dx**2
    if stability > 0.9:
        raise ValueError(
            f"Reference solver may be unstable: stability number={stability:.3f}. "
            "Increase NT_REF."
        )

    U = np.zeros((nt, nx), dtype=np.float64)
    U[0] = initial_condition(x, alpha)
    U[0, 0] = U[0, -1] = 0.0

    def flux(v):
        return 0.5 * v**2

    for n in range(nt - 1):
        u = U[n]
        ul = u[:-1]
        ur = u[1:]
        a = np.maximum(np.abs(ul), np.abs(ur))
        f_half = 0.5 * (flux(ul) + flux(ur)) - 0.5 * a * (ur - ul)

        un = u.copy()
        un[1:-1] = (
            u[1:-1]
            - (dt / dx) * (f_half[1:] - f_half[:-1])
            + nu * dt / dx**2 * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        )
        un[0] = un[-1] = 0.0
        U[n + 1] = un

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


# def init_activation_coeffs(sizes=NETWORK_SIZE, n_coeffs: int = N_COEFFS, seed: int = 21):
#     """One Padé coefficient vector per hidden neuron.

#     The numerator starts near the identity P(y)=y.  Small non-zero denominator
#     coefficients prevent the adaptive denominator from starting at a point with
#     zero sensitivity.
#     """

#     rng = np.random.default_rng(seed)
#     coeffs = []
#     for width in sizes[1:-1]:
#         c = np.zeros((width, n_coeffs), dtype=np.float32)
#         # polyval order: [x^3, x^2, x, 1]
#         c[:, 2] = 1.0
#         c[:, 4:] = 0.03 * rng.standard_normal((width, 4)).astype(np.float32)
#         coeffs.append(jnp.asarray(c))
#     return coeffs

def init_activation_coeffs(sizes=NETWORK_SIZE, n_coeffs: int = N_COEFFS, key = jax.random.key(42), scale=0.5):

    keys = random.split(key, len(sizes) - 2)

    coeffs = []

    for width, k in zip(sizes[1:-1], keys):

        k1, k2 = random.split(k)

        c = scale * random.normal(
            k1,
            (width, n_coeffs)
        )

        # aproximadamente P(y) = y
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
def forward_tanh(X, params):
    h = X
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
    h = X
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


def residual_tanh_single(params, z, nu):
    u = u_tanh(params, z)
    du = grad(u_tanh, argnums=1)(params, z)
    ux, ut = du[0], du[1]

    def ux_fn(zz):
        return grad(u_tanh, argnums=1)(params, zz)[0]

    uxx = grad(ux_fn)(z)[0]
    return ut + u * ux - nu * uxx


residual_tanh = vmap(residual_tanh_single, in_axes=(None, 0, None))


def tanh_loss_components(params, batch, nu=NU):
    Xf, Xic, yic, Xbc, ybc = batch
    r = residual_tanh(params, Xf, nu)
    lf = jnp.mean(r**2)
    lic = jnp.mean((forward_tanh(Xic, params) - yic) ** 2)
    lbc = jnp.mean((forward_tanh(Xbc, params) - ybc) ** 2)
    total = LAMBDA_F * lf + LAMBDA_IC * lic + LAMBDA_BC * lbc
    return total, (lf, lic, lbc)


def tanh_loss(params, batch, nu=NU):
    return tanh_loss_components(params, batch, nu)[0]


@partial(jit, static_argnames=("inner_steps",))
def update_tanh(params, batch, lr=LR_PARAMS, inner_steps=INNER_STEPS):
    def body(_, theta):
        g = grad(tanh_loss)(theta, batch, NU)
        return tree_sgd(theta, g, lr)
    return jax.lax.fori_loop(0, inner_steps, body, params)


# -----------------------------------------------------------------------------
# PINN losses: PAU / smPAF
# -----------------------------------------------------------------------------
def u_adaptive(params, coeffs, z, activation: str):
    return forward_adaptive(z[None, :], params, coeffs, activation)[0, 0]


def residual_adaptive_single(params, coeffs, z, nu, activation: str):
    def u_of_z(zz):
        return u_adaptive(params, coeffs, zz, activation)

    u = u_of_z(z)
    du = grad(u_of_z)(z)
    ux, ut = du[0], du[1]

    def ux_of_z(zz):
        return grad(u_of_z)(zz)[0]

    uxx = grad(ux_of_z)(z)[0]
    return ut + u * ux - nu * uxx


def adaptive_residual(params, coeffs, Xf, nu, activation: str):
    fn = lambda z: residual_adaptive_single(params, coeffs, z, nu, activation)
    return vmap(fn)(Xf)


def adaptive_loss_components(params, coeffs, batch, activation: str, nu=NU):
    Xf, Xic, yic, Xbc, ybc = batch
    r = adaptive_residual(params, coeffs, Xf, nu, activation)
    lf = jnp.mean(r**2)
    lic = jnp.mean((forward_adaptive(Xic, params, coeffs, activation) - yic) ** 2)
    lbc = jnp.mean((forward_adaptive(Xbc, params, coeffs, activation) - ybc) ** 2)
    total = LAMBDA_F * lf + LAMBDA_IC * lic + LAMBDA_BC * lbc
    return total, (lf, lic, lbc)


def adaptive_loss(params, coeffs, batch, activation: str, nu=NU):
    return adaptive_loss_components(params, coeffs, batch, activation, nu)[0]


def make_adaptive_update(activation: str):
    """Create a memory-conscious first-order bilevel update.

    theta: INNER_STEPS updates on train points.
    coefficients: one validation-loss update with theta treated as fixed.

    This intentionally avoids differentiating through the full unrolled inner
    loop, which is considerably more memory intensive for a second-order PINN.
    """
    @partial(jit, static_argnames=("inner_steps",))
    def update(params, coeffs, train_batch, val_batch,
               lrp=LR_PARAMS, lrc=LR_COEFFS, inner_steps=INNER_STEPS):
        def body(_, theta):
            g = grad(adaptive_loss, argnums=0)(theta, coeffs, train_batch, activation, NU)
            return tree_sgd(theta, g, lrp)

        new_params = jax.lax.fori_loop(0, inner_steps, body, params)
        frozen_params = jax.lax.stop_gradient(new_params)
        gc = grad(adaptive_loss, argnums=1)(
            frozen_params, coeffs, val_batch, activation, NU
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
    x, t, U_ref = solve_burgers_rusanov()
    T, X = np.meshgrid(t, x, indexing="ij")
    Xin = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    pred = predict_batched(lambda xb: forward_tanh(xb, params), Xin)
    U_pred = pred.reshape(len(t), len(x)).astype(np.float32)
    rel_l2 = float(np.linalg.norm(U_pred - U_ref) / np.linalg.norm(U_ref))
    return x, t, U_ref, U_pred, rel_l2


def evaluate_adaptive(params, coeffs, activation: str):
    x, t, U_ref = solve_burgers_rusanov()
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


from pathlib import Path

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
        "alpha": ALPHA,
        "nu": float(NU),
        "network_size": list(NETWORK_SIZE),
        "outer_steps": OUTER_STEPS,
        "inner_steps": INNER_STEPS,
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
