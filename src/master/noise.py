import jax.numpy as jnp
import jax
import secrets


def sample_poisson(mu, base_key, id, shape=None):
    key_i = jax.random.fold_in(base_key, id)
    # Poisson part (integer counts + dark current contribution)
    counts = jax.random.poisson(key_i, mu, shape)
    return counts


def sample_from_mu(x_mu, batch_dim=1, base_key=None):
    if base_key is None:
        base_key = jax.random.PRNGKey(secrets.randbits(32))
    x_sample_I = sample_poisson(x_mu, base_key, 0, shape=(batch_dim,) + x_mu.shape)
    # Return noisy yhat in units of field magnitude (not intensity)
    x_sample = jnp.sqrt(x_sample_I)
    return x_mu, x_sample


def get_mu(x, N_photons=1e9):
    I = jnp.abs(x) ** 2
    total_I = jnp.sum(I)
    x_mu = I / total_I * N_photons
    return x_mu


def get_unit_intensity(x):
    I = jnp.abs(x) ** 2
    total_I = jnp.sum(I, axis=(-2, -1))
    x_ph = x / jnp.sqrt(total_I)
    return x_ph


def get_unit_photons(x, N_photons=1e9):
    x_unit = get_unit_intensity(x)
    # Intensity of x_ph has unit photons now, normalized to the given N_photons in total
    x_ph = x_unit * jnp.sqrt(N_photons)
    return x_ph


def sample_field(x, N_photons=1e9, batch_dim=1, base_key=None):
    """Output is in unit of field, so that its intensity is equivalent to the sampled photon counts."""
    x_mu = get_mu(x, N_photons)
    return sample_from_mu(x_mu, batch_dim, base_key=base_key)
