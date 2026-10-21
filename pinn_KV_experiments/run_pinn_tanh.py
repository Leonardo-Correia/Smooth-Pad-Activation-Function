"""Train only the conventional tanh PINN and save its data to NPZ."""
from time import perf_counter
import jax

import pinn_helpers as kv

print("Backend:", jax.default_backend())
print("Devices:", jax.devices())


EXPERIMENT = 'pinn_kdv_tanh_exp1'

def main():
    train_np, val_np = kv.fixed_datasets()
    train = kv.to_jax_batch(train_np)
    val = kv.to_jax_batch(val_np)

    params = kv.init_network_params()
    history = kv.empty_history()
    start = perf_counter()

    print("Training conventional PINN (tanh)")
    print(f"KdV: speed={kv.SPEED}, beta={kv.BETA}, architecture={kv.NETWORK_SIZE}")

    for step in range(kv.OUTER_STEPS):
        params = kv.update_tanh(params, train, inner_steps=kv.INNER_STEPS)

        if step % kv.PRINT_EVERY == 0 or step == kv.OUTER_STEPS - 1:
            tr = kv.components_to_tuple(kv.tanh_loss_components(params, train))
            va = kv.components_to_tuple(kv.tanh_loss_components(params, val))
            kv.record(history, step, tr, va)
            kv.print_progress("tanh", step, tr, va)

    elapsed = perf_counter() - start
    x, t, U_ref, U_pred, rel_l2 = kv.evaluate_tanh(params)

    out = kv.PATH / f"{EXPERIMENT}.npz"
    kv.save_result(out, "tanh", history, train_np, val_np, params,
                   x, t, U_ref, U_pred, rel_l2, elapsed)
    print(f"relative L2={rel_l2:.6e}")
    print(f"Saved: {out}")
    jax.clear_caches()


if __name__ == "__main__":
    main()
