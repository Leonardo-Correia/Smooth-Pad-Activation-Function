#File to create functions for MLP model and optimization 

import jax 
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import pade_ac as paf


def make_batches(X, y, batch_size, key):
    """
    Generate shuffled mini-batches.

    Parameters
    ----------
    X (array-like): Input features.
    y (array-like): Output freatures.
    batch_size (int):  Batch size.
    key (jax.random.PRNGKey): Random key used for shuffling.

    Yields
    ------
    X_batch : jnp.ndarray
        Batch of inputs.
    y_batch : jnp.ndarray
        Batch of targets.
    """
    X = jnp.array(X)
    y = jnp.array(y)

    n = len(X)

    indices = jax.random.permutation(key, n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        batch_idx = indices[start:end]

        yield X[batch_idx], y[batch_idx]


def train_val_test_split(x, y,
                         train_ratio=0.5,
                         val_ratio=0.30,
                         test_ratio=0.20,
                         key=random.PRNGKey(0)):
    """
    Split the input and output data into training, validation, and test sets based on the specified ratios.

    Parameters:
        x (array-like): Input data.
        y (array-like): Output data.
        train_ratio (float): Proportion of the data to be used for training. Default is 0.7.
        val_ratio (float): Proportion of the data to be used for validation. Default is 0.15.
        test_ratio (float): Proportion of the data to be used for testing. Default is 0.15.
        key (jax.random.PRNGKey): Random key for shuffling the data. Default is random.PRNGKey(0).
    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test).
    """

    n = x.shape[0]

    indices = random.permutation(key, n)

    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_val = x[val_idx]
    y_val = y[val_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    return (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test
    )


#ARCH
def random_scale_params(m,n,key,scale = 1e-1):
    """
    Creating a function to generate random weights (m x n) and bias (n)

    Parameters:
        m (int): Number of rows for the weight matrix.
        n (int): Number of columns for the weight matrix and size of the bias vector.
        key (jax.random.PRNGKey): Random key for generating random numbers.
        scale (float): Scaling factor for the random weights and biases. Default is 1e-1.

    Returns:
        Tuple of (weights, biases) where weights is an (m x n) matrix and biases is a vector of size n."""
    
    subkey_1, subkey_2 = random.split(key)
    # We will return a normal distribution multiplied by the scale
    return scale * jax.random.normal(subkey_1, (m,n)), scale * jax.random.normal(subkey_2, (n,))

def init_network_params(size,key): #Creating a function to generate random params for whole network (archtecture = size)
    keys =  random.split(key, len(size)) #Transforming the random key into size chape 
    return [random_scale_params(m,n,k) for m,n,k in zip(size[:-1], size[1:], keys)] 

def init_coeffs(size, n_coeffs, key):

    keys = random.split(key, len(size))

    coeffs = [
        random.normal(k, (n_neurons, n_coeffs))
        for n_neurons, k in zip(size, keys)
    ]

    return coeffs

@jit
def forward(x, params, coeffs):
    """
    Compute the forward pass of the MLP with Padé approximation activation functions.

    Parameters:
        x (array-like): Input data.
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
    Returns:
        Output of the MLP after the forward pass.
    """

    for (w, b), c in zip(params, coeffs[1:]):
        z = jnp.dot(x, w) + b

        x = vmap(
            paf.s_pade,
            in_axes=(1, 0),
            out_axes=1
        )(z, c)

    return x

@jit
def forward_logits(x, params, coeffs):
    """
    Compute the forward pass of the MLP with Padé approximation activation functions without the last activation function.

    Parameters:
        x (array-like): Input data.
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
    Returns:
        Output of the MLP after the forward pass.
    """

    for (w, b), c in zip(params[:-1], coeffs[1:-1]):
        z = jnp.dot(x, w) + b

        x = vmap(
            paf.s_pade,
            in_axes=(1, 0),
            out_axes=1
        )(z, c)

    w_last, b_last = params[-1]
    logits = jnp.dot(x, w_last) + b_last

    return logits

@jit
def mse_loss(params, coeffs, input, output):
    """
    Compute the mean squared error loss between the predicted output of the MLP and the true output.

    Parameters:
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
        input (array-like): Input data.
        output (array-like): True output data.
    Returns:
        Mean squared error loss as a scalar value.
    """

    preds = forward(input, params, coeffs)

    return jnp.mean((preds - output) ** 2)

@jit
def bce_loss(params, coeffs, input, output):
    """
    Compute the mean binary cross entropy with logits between the predictet output of MLP and the true output for classification probles.

    Parameters:
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
        input (array-like): Input data.
        output (array-like): True output data.
    Returns:
        Mean Binary Cross Entropy loss as scalar value.
    """
    logits = forward_logits(input,params, coeffs)

    return jnp.mean(jnp.maximum(logits,0) - logits * output + jnp.log1p(jnp.exp(-jnp.abs(logits))))


@jit
def update_alternate(params,coeffs, x, y, lr):
    """
    Perform an alternate update of the parameters and coefficients using gradient descent.

    Parameters:
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
        x (array-like): Input data.
        y (array-like): True output data.
        lr (float): Learning rate for the gradient descent update.
    Returns:
        Tuple of (new_params, new_coeffs) where new_params is the updated list of
        parameters and new_coeffs is the updated list of coefficients.
    """

    grads_p= grad(mse_loss, argnums = 0)(params, coeffs, x, y)
    flat_params, treedef_p = jax.tree.flatten(params)
    flat_grads_p, _ = jax.tree.flatten(grads_p)

    new_params = [p - lr * d for p,d in zip(flat_params,flat_grads_p)]
    new_params = jax.tree.unflatten(treedef_p, new_params)

    grads_c = grad(mse_loss,argnums = 1)(new_params, coeffs, x,  y)
    flat_coeff, treedef_c = jax.tree.flatten(coeffs)
    flat_grads_c, _ = jax.tree.flatten(grads_c)

    new_coeffs = [c - 10*lr*g for c,g in zip(flat_coeff,flat_grads_c)]

    return new_params, jax.tree.unflatten(treedef_c, new_coeffs)


@partial(jit, static_argnums=(6,))
def update_non_val_bilevel(params, coeffs,x, y, lrp, lrc, inner_steps):
    """"
    Perform a bilevel optimization update without validation dataset of the parameters and coefficients using gradient descent.
    
    Parameters:
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
        x (array-like): Input data.
        y (array-like): True output data.
        lrp (float): Learning rate of parameter update.
        lrc (float): Learning rate of coefficients update.
        inner_steps (int): Number of inner optimization steps.
    Returns:
        Tuple of (new_params, new_coeffs) where new_params is the updated list of
        parameters and new_coeffs is the updated list of coefficients.
        """
    
    def inner_update(params, coeffs,x,y,lrp,inner_steps):
        """
        Intermediate function for inner update loop.

        Returns:
            Tuple of new_params resulted from inner loop.
        """
        for i in range(inner_steps):
            grads_p = grad(mse_loss, argnums=0)(params, coeffs, x, y)
            flat_params, treedef_p = jax.tree.flatten(params)
            flat_grads_p, _ = jax.tree.flatten(grads_p)

            params = [p - lrp * d for p,d in zip(flat_params,flat_grads_p)]
            params = jax.tree.unflatten(treedef_p, params)
        return params
    
    new_params = inner_update(params,coeffs, x, y, lrp, inner_steps)
    grads_c = grad(mse_loss,argnums = 1)(new_params, coeffs, x,  y)
    flat_coeff, treedef_c = jax.tree.flatten(coeffs)
    flat_grads_c, _ = jax.tree.flatten(grads_c)

    new_coeffs = [c - lrc*g for c,g in zip(flat_coeff,flat_grads_c)]
    return new_params, jax.tree.unflatten(treedef_c, new_coeffs)

@partial(jit, static_argnums=(8,))
def update_val_bilevel(params, coeffs,x_train, x_val, y_train, y_val, lrp, lrc, inner_steps):
    """
    Perform a bilevel optimization update with validation dataset of the parameters and coefficients using gradient descent.
    This function uses MLE loss as could be used for general purpose.
    Parameters:
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
        x_train (array-like): Input train data.
        x_val (array-like): Input validation data.
        y_train (array-like): True train output data.
        y_val (array-like): True validation output data.
        lrp (float): Learning rate of parameter update.
        lrc (float): Learning rate of coefficients update.
        inner_steps (int): Number of inner optimization steps.
    Returns:
        Tuple of (new_params, new_coeffs) where new_params is the updated list of
        parameters and new_coeffs is the updated list of coefficients.
    """
    def inner_update(params, coeffs, x_inner, y_inner, lrp, inner_steps):
        """
        Intermediate function for inner update loop.

        Returns:
            Tuple of new_params resulted from inner loop.
        """
        for i in range(inner_steps):
            grads_p = grad(mse_loss, argnums=0)(params, coeffs, x_inner, y_inner)
            flat_params, treedef_p = jax.tree.flatten(params)
            flat_grads_p, _ = jax.tree.flatten(grads_p)

            params = [p - lrp * d for p,d in zip(flat_params,flat_grads_p)]
            params = jax.tree.unflatten(treedef_p, params)
        return params
    
    new_params = inner_update(params,coeffs, x_train, y_train, lrp, inner_steps)
    grads_c = grad(mse_loss,argnums = 1)(new_params, coeffs, x_val, y_val)
    flat_coeff, treedef_c = jax.tree.flatten(coeffs)
    flat_grads_c, _ = jax.tree.flatten(grads_c)

    new_coeffs = [c - lrc*g for c,g in zip(flat_coeff,flat_grads_c)]
    return new_params, jax.tree.unflatten(treedef_c, new_coeffs)


@partial(jit, static_argnums=(8,))
def update_val_bilevel_bce(params, coeffs,x_train, x_val, y_train, y_val, lrp, lrc, inner_steps):    
    """
    Perform a bilevel optimization update with validation dataset of the parameters and coefficients using gradient descent.
    This function uses BCE loss an could be used for classification purpose.

    Parameters:
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
        x_train (array-like): Input train data.
        x_val (array-like): Input validation data.
        y_train (array-like): True train output data.
        y_val (array-like): True validation output data.
        lrp (float): Learning rate of parameter update.
        lrc (float): Learning rate of coefficients update.
        inner_steps (int): Number of inner optimization steps.
    Returns:
        Tuple of (new_params, new_coeffs) where new_params is the updated list of
        parameters and new_coeffs is the updated list of coefficients.
    """ 
    def inner_update(params, coeffs, x_inner, y_inner, lrp, inner_steps):
        """
        Intermediate function for inner update loop.

        Returns:
            Tuple of new_params resulted from inner loop.
        """
        for i in range(inner_steps):
            grads_p = grad(bce_loss, argnums=0)(params, coeffs, x_inner, y_inner)
            flat_params, treedef_p = jax.tree.flatten(params)
            flat_grads_p, _ = jax.tree.flatten(grads_p)

            params = [p - lrp * d for p,d in zip(flat_params,flat_grads_p)]
            params = jax.tree.unflatten(treedef_p, params)
        return params
    
    new_params = inner_update(params,coeffs, x_train, y_train, lrp, inner_steps)
    grads_c = grad(bce_loss,argnums = 1)(new_params, coeffs, x_val, y_val)
    flat_coeff, treedef_c = jax.tree.flatten(coeffs)
    flat_grads_c, _ = jax.tree.flatten(grads_c)

    new_coeffs = [c - lrc * g for c,g in zip(flat_coeff,flat_grads_c)]
    return new_params, jax.tree.unflatten(treedef_c, new_coeffs)

#Visualizating and Results
def plot_predictions(params, coeffs, x_test, y_test):
    """
    Plot the predictions against the true values using forward function.

    Parameters:
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
        x_test (array-like): Input test data.
        y_test (array-like): True test output data.
    
    returns:
        None. Displays a scatter plot of the true values and predictions.
    """
    y_pred = forward(x_test, params, coeffs)

    y_pred_complete = forward(jnp.linspace(0,10,1000).reshape(1000,1), params, coeffs)
    plt.figure(figsize=(7, 4))
    plt.scatter(x_test[:, 0], y_test[:, 0], label="Real data", alpha=0.5)
    plt.scatter(x_test[:, 0], y_pred[:, 0], label="Predictions", alpha=0.5)
    plt.plot(jnp.linspace(0,10,1000).reshape(1000,1),y_pred_complete, color = "red")
    plt.title("Predictions vs Real Data")
    plt.xlabel("x[0]")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_used_activations(coeffs, x_min=-5, x_max=5, n_points=500):
    """
    Plot the Padé activation functions used in the network.

    Parameters:
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
        x_min (float): Minimum value of the x-axis for plotting.
        x_max (float): Maximum value of the x-axis for plotting.
        n_points (int): Number of points to use for plotting.

    Returns:
        None. Displays plots of the activation functions.
    """
    xs = jnp.linspace(x_min, x_max, n_points)

    for layer_idx, layer_coeffs in enumerate(coeffs[1:], start=1):
        plt.figure(figsize=(7, 4))

        for neuron_idx, c in enumerate(layer_coeffs):
            ys = paf.s_pade(xs, c)
            plt.plot(xs, ys, label=f"neurônio {neuron_idx}")

        plt.title(f"Funções de ativação usadas - camada {layer_idx}")
        plt.xlabel("x")
        plt.ylabel("pade(x, coeff)")
        plt.legend()
        plt.grid(True)
        plt.show()


def classification_report_complete(x_test, y_test, params, coeffs, plot_roc_curve=False):
    """
    Generate a classification report including accuracy, precision, recall, and F1-score.

    Parameters:
        x_test (array-like): Test input data.
        y_test (array-like): True labels.
        params (list): List of tuples containing weights and biases for each layer.
        coeffs (list): List of coefficient arrays for the Padé approximation activation functions for each layer.
    Returns:
        None. Prints the classification report and optionally plots the ROC curve.
    """
    probs = jax.nn.sigmoid(forward_logits(x_test, params, coeffs))
    y_pred = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)

    print("Confusion Matrix:")
    print(cm)

    print(f"ROC AUC  : {auc:.4f}")

    print(classification_report(y_test, y_pred, digits=4))

    if plot_roc_curve:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, probs)

        plt.figure(figsize=(7, 4))
        plt.plot(fpr, tpr, label=f"ROC curve (area = {auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
