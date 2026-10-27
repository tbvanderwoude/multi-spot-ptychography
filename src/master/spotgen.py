from functools import partial
from jax import jit
import jax.numpy as jnp
from master.fft import *
import numpy as np

R_FRAC_JITTER_DEFAULT = 0.1
PROBE_SEED_DEFAULT = 2026


@jit
def shift(x, delta_x, delta_y, K_X, K_Y):
    x_spectrum = ifft2_shifted(x)
    displacement_ramp = jnp.exp(delta_x * 1.0j * K_X + delta_y * 1.0j * K_Y)
    shifted_x_spectrum = x_spectrum * displacement_ramp
    shifted_x = fft_shifted(shifted_x_spectrum)
    return shifted_x


@jit
def create_displaced_spot_spectrum(NA, delta_x, delta_y, K_X, K_Y, k, k_0):
    F_airy = jnp.astype(k <= k_0 * NA, jnp.float64)
    displacement_ramp = jnp.exp(delta_x * 1.0j * K_X + delta_y * 1.0j * K_Y)
    spot_spectrum = F_airy * displacement_ramp
    return spot_spectrum


def geometric_sum(theta, N, eps=1e-14):
    z = 1j * theta
    num = jnp.expm1(N * z)
    den = jnp.expm1(z)
    return jnp.where(jnp.abs(den) < eps, N + 0j, num / den)


def lattice_sum_from_linspace(k1d, min_val, max_val, N):
    if N == 1:
        return jnp.exp(1j * min_val * k1d)

    a = (max_val - min_val) / (N - 1)
    theta = a * k1d
    return jnp.exp(1j * min_val * k1d) * geometric_sum(theta, N)


@jit(static_argnames=("NA", "N_x", "N_y"))
def create_multispot_array_product(
    NA, min_x, max_x, min_y, max_y, N_x, N_y, K_X, K_Y, k, k_0
):
    F_airy = jnp.astype(k <= k_0 * NA, jnp.float64)
    dx = jnp.linspace(min_x, max_x, N_x)[:, None, None]
    dy = jnp.linspace(min_y, max_y, N_y)[:, None, None]

    Sx = jnp.exp(1j * dx * K_X[None, :, :]).sum(axis=0)
    Sy = jnp.exp(1j * dy * K_Y[None, :, :]).sum(axis=0)

    total_ramp = Sx * Sy
    # Use simple linearity of fft2_shifted to avoid making too many fft2s
    return fft2_shifted(F_airy * total_ramp)


@partial(jit, static_argnames=("N_x", "N_y"))
def create_multispot_array_implicit(
    NA, min_x, max_x, min_y, max_y, N_x, N_y, kx1d, ky1d, k, k_0
):
    F_airy = (k <= k_0 * NA).astype(jnp.float64)

    Sx = lattice_sum_from_linspace(kx1d, min_x, max_x, N_x)
    Sy = lattice_sum_from_linspace(ky1d, min_y, max_y, N_y)

    total_ramp = Sy[:, None] * Sx[None, :]
    return fft2_shifted(F_airy * total_ramp)


@partial(jit, static_argnames=("N_x", "N_y"))
def create_multispot_array_original(
    NA, min_x, max_x, min_y, max_y, N_x, N_y, K_X, K_Y, k, k_0
):

    delta_xs = jnp.linspace(min_x, max_x, N_x)
    delta_ys = jnp.linspace(min_y, max_y, N_y)
    total = jnp.zeros_like(k)
    for delta_x in delta_xs:
        for delta_y in delta_ys:
            spot_spectrum = create_displaced_spot_spectrum(
                NA, delta_x, delta_y, K_X, K_Y, k, k_0
            )
            total += fft2_shifted(spot_spectrum)
    # Use simple linearity of fft2_shifted to avoid calculating too many fft2s
    return total


@partial(jit, static_argnames=("N_x", "N_y"))
def create_multispot_array_loops(
    NA, min_x, max_x, min_y, max_y, N_x, N_y, K_X, K_Y, k, k_0
):

    delta_xs = jnp.linspace(min_x, max_x, N_x)
    delta_ys = jnp.linspace(min_y, max_y, N_y)
    total = jnp.zeros_like(k)
    for delta_x in delta_xs:
        for delta_y in delta_ys:
            spot_spectrum = create_displaced_spot_spectrum(
                NA, delta_x, delta_y, K_X, K_Y, k, k_0
            )
            total += spot_spectrum
    # Use simple linearity of fft2_shifted to avoid calculating too many fft2s
    return fft2_shifted(total)


@jit(static_argnames=("NA", "N_x", "N_y"))
def create_multispot_array_explicit(
    NA, min_x, max_x, min_y, max_y, N_x, N_y, K_X, K_Y, k, k_0
):
    delta_xs = jnp.linspace(min_x, max_x, N_x)[:, None, None, None]
    delta_ys = jnp.linspace(min_y, max_y, N_y)[None, :, None, None]
    KX = K_X[None, None, :, :]
    KY = K_Y[None, None, :, :]
    total_ramp = jnp.zeros_like(k)
    F_airy = jnp.astype(k <= k_0 * NA, jnp.float64)
    ramps = jnp.exp(1j * (delta_xs * KX + delta_ys * KY))
    total_ramp = ramps.sum(axis=(0, 1))
    # Use simple linearity of fft2_shifted to avoid calculating too many fft2s
    return fft2_shifted(F_airy * total_ramp)


def create_multispot_array_product_jittered(
    NA,
    min_x,
    max_x,
    min_y,
    max_y,
    N_x,
    N_y,
    K_X,
    K_Y,
    k,
    k_0,
    r_frac_jitter=R_FRAC_JITTER_DEFAULT,
    probe_seed=PROBE_SEED_DEFAULT,
    be=jnp,
):
    F_airy = be.astype(k <= k_0 * NA, be.float64)
    dx = be.linspace(min_x, max_x, N_x)
    dy = be.linspace(min_y, max_y, N_y)

    delta_r = (dx[1] - dx[0]) * r_frac_jitter
    np.random.seed(probe_seed)
    total_ramp = be.zeros_like(F_airy)
    delta_x = np.random.uniform(low=-delta_r / 2.0, high=delta_r / 2.0, size=(N_x, N_y))
    delta_y = np.random.uniform(low=-delta_r / 2.0, high=delta_r / 2.0, size=(N_y, N_y))

    total_ramp = np.sum(
        be.exp(
            1j
            * (
                (dx[:, None, None, None] + delta_x[:, :, None, None]) * K_X
                + (dy[None, :, None, None] + delta_y[:, :, None, None]) * K_Y
            )
        ),
        axis=(0, 1),
    )
    # Use simple linearity of fft2_shifted to avoid making too many fft2s
    return fft2_shifted(F_airy * total_ramp)


def create_multispot_array_product_jittered_separable(
    NA,
    min_x,
    max_x,
    min_y,
    max_y,
    N_x,
    N_y,
    K_X,
    K_Y,
    k,
    k_0,
    r_frac_jitter=R_FRAC_JITTER_DEFAULT,
    probe_seed=PROBE_SEED_DEFAULT,
):

    F_airy = jnp.astype(k <= k_0 * NA, jnp.float64)
    dx = jnp.linspace(min_x, max_x, N_x)[:, None, None]
    dy = jnp.linspace(min_y, max_y, N_y)[:, None, None]

    delta_r = (dx[1] - dx[0]) * r_frac_jitter
    np.random.seed(probe_seed)
    delta_x = np.random.uniform(
        low=-delta_r / 2.0, high=delta_r / 2.0, size=(N_x, 1, 1)
    )
    delta_y = np.random.uniform(
        low=-delta_r / 2.0, high=delta_r / 2.0, size=(N_y, 1, 1)
    )
    Sx = jnp.exp(1j * (dx + delta_x) * K_X[None, :, :]).sum(axis=0)
    Sy = jnp.exp(1j * (dy + delta_y) * K_Y[None, :, :]).sum(axis=0)

    total_ramp = Sx * Sy
    # Use simple linearity of fft2_shifted to avoid making too many fft2s
    return fft2_shifted(F_airy * total_ramp)
