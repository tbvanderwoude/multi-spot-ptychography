from jax import jit
from .phase_retrieval import twin_image
import jax.numpy as jnp


@jit(static_argnames="axis")
def vecdotimage(a, b, axis=(-2, -1)):
    """Interesting corner case not in JAX jnp: need to flatten two axes (which vecdot would), not also batch axes (like vdot would) and calculate complex inner product"""
    return jnp.sum(jnp.conj(a) * b, axis=axis)


@jit
def choose_twin_broadcast(x, xref):
    x_t = twin_image(x)
    choose_t = jnp.abs(vecdotimage(x_t, xref, axis=(-2, -1))) > jnp.abs(
        vecdotimage(x, xref, axis=(-2, -1))
    )
    return jnp.where(choose_t[..., None, None], x_t, x)


@jit
def choose_representative_broadcast(x, xref):
    # Break conjugate symmetry by choosing best fitting choice of image vs twin image
    x = choose_twin_broadcast(x, xref)
    # Fix global phase difference
    phi = jnp.angle(vecdotimage(x, xref, axis=(-2, -1)))
    return x * jnp.exp(1j * phi[..., None, None])


def choose_twin(x, xref):
    x_t = twin_image(x)
    if jnp.abs(jnp.vdot(x_t, xref)) > jnp.abs(jnp.vdot(x, xref)):
        return x_t
    return x


def choose_representative(x, xref):
    # Break conjugate symmetry by choosing best fitting choice of image vs twin image
    x = choose_twin(x, xref)
    # Fix global phase difference
    phi = jnp.angle(jnp.vdot(x, xref))
    return x * jnp.exp(1j * phi)
