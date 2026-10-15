"""Train only the PINN with smPAF and save its data to NPZ."""
from time import perf_counter
import jax

import pinn_helpers as bc

UPDATE = bc.make_adaptive_update("smpaf")

print("Backend:", jax.default_backend())
print("Devices:", jax.devices())



EXPERIMENT = 'pinn_smPAF_exp7'


def main():
    train_np, val_np = bc.fixed_datasets()
    train = bc.to_jax_batch(train_np)
    val = bc.to_jax_batch(val_np)

    params = bc.init_network_params()
    coeffs = bc.init_activation_coeffs()
    history = bc.empty_history()
    start = perf_counter()

    print("Training PINN-smPAF")
    print(f"alpha={bc.ALPHA}, architecture={bc.NETWORK_SIZE}, kappa={bc.KAPPA}")
    print("Coefficient update: memory-conscious first-order bilevel")

    for step in range(bc.OUTER_STEPS):
        params, coeffs = UPDATE(
            params, coeffs, train, val,
            inner_steps=bc.INNER_STEPS,
        )

        if step % bc.PRINT_EVERY == 0 or step == bc.OUTER_STEPS - 1:
            tr = bc.components_to_tuple(bc.adaptive_loss_components(params, coeffs, train, "smpaf"))
            va = bc.components_to_tuple(bc.adaptive_loss_components(params, coeffs, val, "smpaf"))
            bc.record(history, step, tr, va)
            bc.print_progress("smPAF", step, tr, va)

    elapsed = perf_counter() - start
    x, t, U_ref, U_pred, rel_l2 = bc.evaluate_adaptive(params, coeffs, "smpaf")

    out = f"{bc.PATH}{EXPERIMENT}.npz"
    bc.save_result(out, "smpaf", history, train_np, val_np, params,
                   x, t, U_ref, U_pred, rel_l2, elapsed, coeffs=coeffs)
    print(f"relative L2={rel_l2:.6e}")
    print(f"Saved: {out}")
    jax.clear_caches()


if __name__ == "__main__":
    main()
