# File created for Bi-level optimization in MLP


import jax
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
from jax.nn import softmax

import mlp_helpers as mh
import pade_ac as paf
import dataset_pade_ac as dpa

import matplotlib.pyplot as plt

#Nerwork Parameters
lrp = 0.002
lrc = 0.005
n_int = 40
inner_steps = 100
batch_size = 128
key = random.PRNGKey(0)

#Network Archtecture
size = [108,15,15,1]

key_params = random.PRNGKey(42)
key_coeffs = random.PRNGKey(21)

params = mh.init_network_params(size, key_params)
coeffs = mh.init_coeffs(size, n_coeffs=8, key=key_coeffs)
coeffs_init = coeffs

# Data transform
X, y = dpa.load_adult_dataset()

categorical_cols, numeric_cols = dpa.get_feature_types(X)

preprocess = dpa.create_preprocessor(
    categorical_cols,
    numeric_cols,
)

X_train, X_val, X_test, y_train, y_val, y_test = dpa.split_and_preprocess_data(X,y)


X_train = preprocess.fit_transform(X_train)
X_val = preprocess.transform(X_val)
X_test = preprocess.transform(X_test)

y_train = jnp.array(y_train.to_numpy().reshape(-1, 1))
y_val = jnp.array(y_val.to_numpy().reshape(-1, 1))
y_test = jnp.array(y_test.to_numpy().reshape(-1, 1))

#LOOPING
loss_count_outer = []
loss_count_inner = []
for epoch in range(n_int):
    key, subkey = random.split(key)

    train_batches = mh.make_batches(X_train, y_train, batch_size, key)
    val_batches = mh.make_batches(X_val, y_val, batch_size, key)

    for xb, yb in train_batches:
        try:
            xvb, yvb = next(val_batches)
        except StopIteration:
            val_batches = mh.make_batches(X_val, y_val, batch_size, key)
            xvb, yvb = next(val_batches)
        params, coeffs = mh.update_val_bilevel_bce(
            params, coeffs,
            xb, xvb,
            yb, yvb,
            lrp, lrc,
            inner_steps=inner_steps
        )
        train_loss = mh.bce_loss(params, coeffs, xb, yb)
        loss_count_inner.append(train_loss)
    train_loss = mh.bce_loss(params, coeffs, X_train, y_train)
    loss_count_outer.append(train_loss)
    if epoch % 5 == 0:
        print(
            f"Step {epoch}, "
            f"Train Loss: {mh.bce_loss(params, coeffs, X_train, y_train):.6f}, "
            f"Val Loss: {mh.bce_loss(params, coeffs, X_val, y_val):.6f}"
        )

# Results 
mh.classification_report_complete(X_test, y_test, params, coeffs, plot_roc_curve=True)