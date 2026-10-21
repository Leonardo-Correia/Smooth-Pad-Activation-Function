"""Train only the PINN with the original non-smooth PAU and save to NPZ."""
from time import perf_counter
import jax

import pinn_helpers as kv

UPDATE = kv.make_adaptive_update("pau")


print("Backend:", jax.default_backend())
print("Devices:", jax.devices())


EXPERIMENT = 'pinn_kdv_pau_exp1'

def main():
    train_np, val_np = kv.fixed_datasets()
    train = kv.to_jax_batch(train_np)
    val = kv.to_jax_batch(val_np)

    params = kv.init_network_params()
    coeffs = kv.init_activation_coeffs()
    history = kv.empty_history()
    start = perf_counter()

    print("Training PINN-PAU")
    print(f"KdV: speed={kv.SPEED}, beta={kv.BETA}, architecture={kv.NETWORK_SIZE}")
    print("Coefficient update: validation loss, one inner step")

    for step in range(kv.OUTER_STEPS):
        params, coeffs = UPDATE(
            params, coeffs, train, val,
            inner_steps=1,
        )

        if step % kv.PRINT_EVERY == 0 or step == kv.OUTER_STEPS - 1:
            tr = kv.components_to_tuple(kv.adaptive_loss_components(params, coeffs, train, "pau"))
            va = kv.components_to_tuple(kv.adaptive_loss_components(params, coeffs, val, "pau"))
            kv.record(history, step, tr, va)
            kv.print_progress("PAU", step, tr, va)

    elapsed = perf_counter() - start
    x, t, U_ref, U_pred, rel_l2 = kv.evaluate_adaptive(params, coeffs, "pau")

    out = kv.PATH / f"{EXPERIMENT}.npz"
    kv.save_result(out, "pau", history, train_np, val_np, params,
                   x, t, U_ref, U_pred, rel_l2, elapsed, coeffs=coeffs)
    print(f"relative L2={rel_l2:.6e}")
    print(f"Saved: {out}")
    jax.clear_caches()


if __name__ == "__main__":
    main()
