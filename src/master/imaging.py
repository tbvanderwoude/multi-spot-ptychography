import jax.numpy as jnp
from jax import jit


@jit
def make_circular_mask(R, radius):
    return (R <= radius).astype(R.dtype)


@jit
def calc_power(x):
    return jnp.sum(jnp.abs(x) ** 2)
