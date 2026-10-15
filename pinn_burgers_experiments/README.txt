Burgers PINN experiments
========================

Run each model in a separate Python process:

  python run_pinn_tanh.py
  python run_pinn_pau.py
  python run_pinn_smpaf.py

Outputs:

  results/pinn_tanh.npz
  results/pinn_pau.npz
  results/pinn_smpaf.npz

All scripts use the same:
- Burgers problem with alpha=1.0;
- architecture and initial network weights;
- fixed train/validation collocation points;
- number of weight updates and learning rate;
- reference numerical solution.

The NPZ files store loss histories, train/validation points, final network
parameters, reference/predicted solutions, relative L2 error and runtime.
No plotting is done during training.

Memory note:
PAU/smPAF use a first-order bilevel coefficient update (theta is frozen during
the validation coefficient update) instead of differentiating through the
entire unrolled inner optimization. This is intentional to lower memory use for
the second-order PINN residual.
