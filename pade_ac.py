# File created for Padé Approximation Activation Function

## safe Padé: sp = sum(ai*xi)/ 1 + |sum(bi*xi)|

import jax.numpy as jnp
import jax 
from jax import jit, vmap

import matplotlib.pyplot as plt

@jit
def pade(x, coeffs):
    """
    Compute the Padé approximation of the activation function from the coefficients.

    Parameters:
    x (array-like): Input array.
    coeffs (array-like): Coefficients of the Padé approximation, where we will use safe Padé form.
    OBS: For n coefficients values we will assume (n-1)/2 ate the numerator and (n+1)/2 at the quocient

    Returns: A scalar value in R
    """
    n = len(coeffs)
    split = (n+1)//2

    p_coeffs = coeffs[:split]
    q_coeffs = coeffs[split:]

    Px = jnp.polyval(p_coeffs,x) #sum(ci*xi^i) for i = 0 up to (n-1)/2
    Qx = 1+ jnp.abs(jnp.polyval(q_coeffs,x)) #1 + |sum(cj*xj) - cj| for j up to (n+1)/2
    return Px/Qx


@jit
def s_pade(x, coeffs, epi = 0.4):
    """
    Compute the smooth Padé (s_Padé) approximation of the activation function from the coefficients.

    Parameters:
    x (array-like): Input array.
    coeffs (array-like): Coefficients of the Padé approximation, where we will use safe Padé form.
    OBS: For n coefficients values we will assume (n-1)/2 ate the numerator and (n+1)/2 at the quocient

    Returns: A scalar value in R
    """
    n = len(coeffs)
    split = (n+1)//2

    p_coeffs = coeffs[:split]
    q_coeffs = coeffs[split:]

    Px = jnp.polyval(p_coeffs,x) #sum(ci*xi^i) for i = 0 up to (n-1)/2
    Qx = 1 + jnp.sqrt(jnp.polyval(q_coeffs,x)**2 + epi) #1 + sqrt(sum(cj*xj) - cj + epi) for j up to (n+1)/2
    return Px/Qx

