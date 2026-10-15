"""Parameterized data generation for the 1D viscous Burgers equation.

We solve

    u_t + u u_x = nu u_xx,   x in [-1, 1], t in [0, 1]
    u(x, 0; alpha) = -alpha sin(pi x)
    u(-1, t; alpha) = u(1, t; alpha) = 0

The scalar ``alpha`` parameterizes the initial condition.  Consequently, a
parametric PINN can use (x, t, alpha) as input and learn a family of Burgers
solutions instead of one fixed trajectory.
"""

from __future__ import annotations

import numpy as np


def initial_condition(x: np.ndarray, alpha: float | np.ndarray = 1.0) -> np.ndarray:
    """Parameterized initial condition u(x,0;alpha) = -alpha sin(pi x)."""
    return -np.asarray(alpha) * np.sin(np.pi * np.asarray(x))


def solve_burgers_rusanov(
    nx: int = 256,
    nt: int = 1001,
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    nu: float = 0.01 / np.pi,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve one Burgers realization for a selected initial-condition alpha."""
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(t_min, t_max, nt)

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Conservative warnings for the explicit convection/diffusion scheme.
    diffusion_number = nu * dt / dx**2
    convection_number = abs(alpha) * dt / dx
    if diffusion_number > 0.45:
        print(
            f"Warning: diffusion number={diffusion_number:.3f} may be too large. "
            "Increase nt or reduce nx/nu."
        )
    if convection_number > 0.45:
        print(
            f"Warning: convection CFL~{convection_number:.3f} may be too large. "
            "Increase nt or reduce |alpha|."
        )

    U = np.zeros((nt, nx), dtype=np.float64)
    U[0, :] = initial_condition(x, alpha)
    U[:, 0] = 0.0
    U[:, -1] = 0.0

    def flux(v: np.ndarray) -> np.ndarray:
        return 0.5 * v**2

    for n in range(nt - 1):
        un = U[n].copy()

        ul = un[:-1]
        ur = un[1:]
        a = np.maximum(np.abs(ul), np.abs(ur))
        f_half = 0.5 * (flux(ul) + flux(ur)) - 0.5 * a * (ur - ul)

        conv = -(dt / dx) * (f_half[1:] - f_half[:-1])
        diff = nu * dt / dx**2 * (un[2:] - 2.0 * un[1:-1] + un[:-2])

        U[n + 1, 1:-1] = un[1:-1] + conv + diff
        U[n + 1, 0] = 0.0
        U[n + 1, -1] = 0.0

    return x, t, U


def solve_burgers_family(
    alpha_values: np.ndarray,
    **solver_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve Burgers for several initial-condition amplitudes.

    Returns U with shape (n_alpha, nt, nx).
    """
    alpha_values = np.asarray(alpha_values, dtype=float).reshape(-1)
    solutions = []
    x = t = None

    for alpha in alpha_values:
        x, t, U = solve_burgers_rusanov(alpha=float(alpha), **solver_kwargs)
        solutions.append(U)

    return x, t, np.stack(solutions, axis=0)


def make_supervised_pairs(
    x: np.ndarray,
    t: np.ndarray,
    U: np.ndarray,
    alpha_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised pairs.

    For a single solution U(nt,nx), X=(x,t).
    For a family U(n_alpha,nt,nx), X=(x,t,alpha).
    """
    if U.ndim == 2:
        T, X = np.meshgrid(t, x, indexing="ij")
        X_all = np.stack([X.reshape(-1), T.reshape(-1)], axis=1)
        return X_all.astype(np.float32), U.reshape(-1, 1).astype(np.float32)

    if U.ndim != 3 or alpha_values is None:
        raise ValueError("For a solution family, provide U(n_alpha,nt,nx) and alpha_values.")

    alpha_values = np.asarray(alpha_values).reshape(-1)
    if len(alpha_values) != U.shape[0]:
        raise ValueError("len(alpha_values) must equal U.shape[0].")

    X_blocks, y_blocks = [], []
    T, X = np.meshgrid(t, x, indexing="ij")
    xt = np.stack([X.reshape(-1), T.reshape(-1)], axis=1)

    for k, alpha in enumerate(alpha_values):
        a = np.full((xt.shape[0], 1), alpha)
        X_blocks.append(np.hstack([xt, a]))
        y_blocks.append(U[k].reshape(-1, 1))

    return (
        np.vstack(X_blocks).astype(np.float32),
        np.vstack(y_blocks).astype(np.float32),
    )


def split_supervised_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Random train/validation/test split for reference/evaluation data."""
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be in [0,1].")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_train = int(train_ratio * len(idx))
    n_val = int(val_ratio * len(idx))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
    }


def make_pinn_points(
    n_f: int = 20_000,
    n_ic: int = 1024,
    n_bc: int = 1024,
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    alpha_min: float = 0.5,
    alpha_max: float = 1.5,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Create parametric PINN points with X=(x,t,alpha).

    alpha is sampled independently in [alpha_min, alpha_max].
    """
    if alpha_min >= alpha_max:
        raise ValueError("alpha_min must be smaller than alpha_max.")

    rng = np.random.default_rng(seed)

    # Interior physics collocation points.
    x_f = rng.uniform(x_min, x_max, (n_f, 1))
    t_f = rng.uniform(t_min, t_max, (n_f, 1))
    a_f = rng.uniform(alpha_min, alpha_max, (n_f, 1))
    X_f = np.hstack([x_f, t_f, a_f])

    # Initial-condition points: t=t_min and u=-alpha sin(pi x).
    x_ic = rng.uniform(x_min, x_max, (n_ic, 1))
    a_ic = rng.uniform(alpha_min, alpha_max, (n_ic, 1))
    t_ic = np.full_like(x_ic, t_min)
    X_ic = np.hstack([x_ic, t_ic, a_ic])
    y_ic = initial_condition(x_ic, a_ic)

    # Boundary points: u(x_min,t;alpha)=u(x_max,t;alpha)=0.
    t_bc = rng.uniform(t_min, t_max, (n_bc, 1))
    a_bc = rng.uniform(alpha_min, alpha_max, (n_bc, 1))
    X_left = np.hstack([np.full_like(t_bc, x_min), t_bc, a_bc])
    X_right = np.hstack([np.full_like(t_bc, x_max), t_bc, a_bc])
    X_bc = np.vstack([X_left, X_right])
    y_bc = np.zeros((2 * n_bc, 1))

    return {
        "X_f": X_f.astype(np.float32),
        "X_ic": X_ic.astype(np.float32),
        "y_ic": y_ic.astype(np.float32),
        "X_bc": X_bc.astype(np.float32),
        "y_bc": y_bc.astype(np.float32),
    }


def generate_burgers_dataset(
    nx: int = 256,
    nt: int = 1001,
    nu: float = 0.01 / np.pi,
    alpha_values: np.ndarray | None = None,
    alpha_min: float = 0.5,
    alpha_max: float = 1.5,
    n_f: int = 20_000,
    n_ic: int = 1024,
    n_bc: int = 1024,
    seed: int = 0,
    save_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Generate reference solutions plus parametric PINN training points."""
    if alpha_values is None:
        alpha_values = np.linspace(alpha_min, alpha_max, 5)
    alpha_values = np.asarray(alpha_values, dtype=np.float64)

    x, t, U = solve_burgers_family(
        alpha_values,
        nx=nx,
        nt=nt,
        nu=nu,
    )
    X_all, y_all = make_supervised_pairs(x, t, U, alpha_values)
    supervised = split_supervised_data(X_all, y_all, seed=seed)
    pinn = make_pinn_points(
        n_f=n_f,
        n_ic=n_ic,
        n_bc=n_bc,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        seed=seed,
    )

    data = {
        "x_grid": x.astype(np.float32),
        "t_grid": t.astype(np.float32),
        "alpha_values": alpha_values.astype(np.float32),
        "U": U.astype(np.float32),
        "X_all": X_all,
        "y_all": y_all,
        **supervised,
        **pinn,
        "nu": np.array([nu], dtype=np.float32),
        "alpha_min": np.array([alpha_min], dtype=np.float32),
        "alpha_max": np.array([alpha_max], dtype=np.float32),
    }

    if save_path is not None:
        np.savez_compressed(save_path, **data)

    return data


if __name__ == "__main__":
    data = generate_burgers_dataset(
        alpha_values=np.linspace(0.5, 1.5, 5),
        save_path="burgers_parametric_dataset.npz",
    )
    print("Parameterized Burgers dataset generated.")
    for key, value in data.items():
        print(f"{key}: {value.shape}")
