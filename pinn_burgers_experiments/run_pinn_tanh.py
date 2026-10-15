"""Train only the conventional tanh PINN and save its data to NPZ."""
from time import perf_counter
import jax

import pinn_helpers as bc

print("Backend:", jax.default_backend())
print("Devices:", jax.devices())


EXPERIMENT = 'pinn_tahn_exp3'

def main():
    train_np, val_np = bc.fixed_datasets()
    train = bc.to_jax_batch(train_np)
    val = bc.to_jax_batch(val_np)

    params = bc.init_network_params()
    history = bc.empty_history()
    start = perf_counter()

    print("Training conventional PINN (tanh)")
    print(f"alpha={bc.ALPHA}, architecture={bc.NETWORK_SIZE}")

    for step in range(bc.OUTER_STEPS):
        params = bc.update_tanh(params, train, inner_steps=bc.INNER_STEPS)

        if step % bc.PRINT_EVERY == 0 or step == bc.OUTER_STEPS - 1:
            tr = bc.components_to_tuple(bc.tanh_loss_components(params, train))
            va = bc.components_to_tuple(bc.tanh_loss_components(params, val))
            bc.record(history, step, tr, va)
            bc.print_progress("tanh", step, tr, va)

    elapsed = perf_counter() - start
    x, t, U_ref, U_pred, rel_l2 = bc.evaluate_tanh(params)

    out = f"{bc.PATH}{EXPERIMENT}.npz"
    bc.save_result(out, "tanh", history, train_np, val_np, params,
                   x, t, U_ref, U_pred, rel_l2, elapsed)
    print(f"relative L2={rel_l2:.6e}")
    print(f"Saved: {out}")
    jax.clear_caches()


if __name__ == "__main__":
    main()
