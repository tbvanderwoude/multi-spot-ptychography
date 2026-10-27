import jax.numpy as jnp
import jax

from master.obj import center_crop
from .fft import fft2_shifted


def bouchet_bounds(x_true):
    n_k = jnp.abs(x_true) ** 2
    n_k_std_bouchet = jnp.sqrt(n_k)
    phi_k_std_bouchet = jnp.sqrt(1 / (4 * n_k))
    return n_k_std_bouchet, phi_k_std_bouchet


def bouchet_bounds_cart(x_true):
    u_std_bouchet = jnp.ones_like(x_true) * 0.5
    v_std_bouchet = jnp.ones_like(x_true) * 0.5
    return u_std_bouchet, v_std_bouchet


@jax.jit
def fisher_2x2_block_fast(x_ph, y_ph, N_y, eps=1e-12):
    N1, N2 = x_ph.shape
    d = N_y

    n = jnp.abs(x_ph) ** 2
    phi = jnp.angle(x_ph)
    psi = jnp.angle(y_ph)

    # g_k = exp(-i 2 psi_k)
    g = jnp.exp(-2j * psi)

    # Centered inverse FFT gives:
    # H[J] = (1/sqrt(d)) * sum_k g[k] * exp(+2πi k·J/N)
    H = fft2_shifted(g)

    S = construct_S(N1, N2, d, phi, H)

    ReS = jnp.real(S)
    ImS = jnp.imag(S)

    # Use closed forms for the three sums
    cos2_sum = 0.5 * (d + ReS)
    sin2_sum = 0.5 * (d - ReS)
    cos_sin_sum = 0.5 * ImS

    Jn_diag = cos2_sum / (d * (n + eps))
    Jphi_diag = 4.0 * (n + eps) * sin2_sum / d
    Jnphi_off = -2.0 * cos_sin_sum / d

    n_var_diag = 1.0 / (Jn_diag + eps)
    phi_var_diag = 1.0 / (Jphi_diag + eps)

    return Jn_diag, Jphi_diag, Jnphi_off, n_var_diag, phi_var_diag


@jax.jit
def fisher_2x2_block_fast_cart(x_ph, y_ph, N_y, eps=1e-12):
    N1, N2 = x_ph.shape
    d = N_y

    psi = jnp.angle(y_ph)

    # g_k = exp(-i 2 psi_k)
    g = jnp.exp(-2j * psi)

    # Centered inverse FFT gives:
    # H[J] = (1/sqrt(d)) * sum_k g[k] * exp(+2πi k·J/N)
    H = fft2_shifted(g)

    T = construct_T(N1, N2, d, H)

    ReT = jnp.real(T) / d
    ImT = jnp.imag(T) / d

    Juu_diag = 2.0 * (1.0 + ReT)
    Jvv_diag = 2.0 * (1.0 - ReT)
    Juv_off = -2.0 * ImT

    u_var_diag = 1.0 / (Juu_diag + eps)
    v_var_diag = 1.0 / (Jvv_diag + eps)
    return Juu_diag, Jvv_diag, Juv_off, u_var_diag, v_var_diag


def construct_H2(N1, N2, H):
    # Build centered object-coordinate grids J1, J2
    J1 = jnp.arange(N1) - N1 // 2
    J2 = jnp.arange(N2) - N2 // 2
    J1, J2 = jnp.meshgrid(J1, J2, indexing="ij")

    # Need the transform evaluated at doubled indices: 2J modulo N
    I1 = jnp.mod(2 * J1, N1)
    I2 = jnp.mod(2 * J2, N2)

    # Convert centered indices back to array indices
    I1 = (I1 + N1 // 2) % N1
    I2 = (I2 + N2 // 2) % N2

    # Gather H[2J]
    H2 = H[I1, I2]
    return H2


def construct_T(N1, N2, d, H):
    H2 = construct_H2(N1, N2, H)

    # T_j = sum_k exp(2i alpha_kj) = sqrt(d) H[2J]   (comparing to S, no exp(2i phi) factor)
    T = jnp.sqrt(d) * H2
    return T


def construct_S(N1, N2, d, phi, H):
    # S_j = exp(i 2 phi_j) * sum_k exp(i 2 beta_kj) exp(-i 2 psi_k)
    # and sum_k(...) = sqrt(d) * H[2J]
    return jnp.exp(2j * phi) * construct_T(N1, N2, d, H)


@jax.jit
def fisher_pinv_2x2_block_cart(x_ph, y_ph, N_y, eps=1e-12):
    Juu, Jvv, Juv, _, _ = fisher_2x2_block_fast_cart(x_ph, y_ph, N_y, eps)
    Jpinv_uu = Juu / 16
    Jpinv_vv = Jvv / 16
    Jpinv_uv = Juv / 16
    return Jpinv_uu, Jpinv_vv, Jpinv_uv


@jax.jit
def fisher_pinv_2x2_block(x_ph, y_ph, N_y, eps=1e-12):
    n = jnp.abs(x_ph) ** 2
    Jnn, Jpp, Jnp, _, _ = fisher_2x2_block_fast(x_ph, y_ph, N_y, eps)
    return n**2 * Jnn, Jpp / (16 * n**2), Jnp / 4


@jax.jit
def fisher_random_phase(x_ph, eps=1e-12):
    n = jnp.abs(x_ph) ** 2.0
    Jn = 1.0 / (2.0 * (n + eps))
    Jphi = 2.0 * (n + eps)
    Jnphi = jnp.zeros_like(n)
    return Jn, Jphi, Jnphi, 1.0 / Jn, 1.0 / Jphi


@jax.jit
def fisher_random_phase_cart(x_ph, eps=1e-12):
    J = 2.0 * jnp.ones_like(jnp.abs(x_ph))
    z = jnp.zeros_like(J)
    return J, J, z, 1.0 / J, 1.0 / J


def ptycho_S_random_phase(probes, det_shape, pupil=1.0):
    N = probes.shape[-2] * probes.shape[-1]
    return (
        jnp.sum(jnp.broadcast_to(pupil, det_shape))
        / N
        * jnp.sum(jnp.abs(probes) ** 2, 0)
    )


def fisher_pty_random_phase(probes, det_shape, x, pupil=1.0):
    S_random_phase = ptycho_S_random_phase(probes, det_shape, pupil)
    return fisher_pty_2x2_cart(S_random_phase, 0.0), fisher_pty_2x2_polar(
        S_random_phase, 0.0, x
    )


def ptycho_SQ(probes, ys, pad, pupil=1.0):
    # Get the image dimension
    N1, N2 = probes.shape[-2:]
    N = N1 * N2
    # Construct the pupil weighted
    H = jax.vmap(fft2_shifted)(pad(pupil * jnp.exp(-2j * jnp.angle(ys))))
    S = (
        jnp.sum(jnp.broadcast_to(pupil, ys.shape[-2:]))
        / N
        * jnp.sum(jnp.abs(probes) ** 2, 0)
    )
    T = jax.vmap(lambda h: construct_T(N1, N2, N, h))(H)
    Q = jnp.sum(probes**2 * T, 0) / N
    return S, Q


def construct_MSE_bound_from_CRB(x_true, var_bound_u, var_bound_v, N_crop):
    var_bound_u = center_crop(var_bound_u, N_crop)
    var_bound_v = center_crop(var_bound_v, N_crop)
    pixel_mse = var_bound_u + var_bound_v
    # MSE takes squared sum of u, v error. Summing u, v in units of variance
    # of u, v gives the right bound.
    SSE = jnp.sum(pixel_mse, axis=(-2, -1))
    norm = jnp.sum(jnp.abs(x_true) ** 2, axis=(-2, -1))
    SSE_norm = SSE / norm
    return SSE_norm


def fisher_pty_2x2_cart(S, Q):
    return 2 * (S + Q.real), 2 * (S - Q.real), -2 * Q.imag


def fisher_pty_2x2_polar(S, Q, x):
    n, Qp = jnp.abs(x) ** 2, jnp.exp(2j * jnp.angle(x)) * Q
    return (S + Qp.real) / (2 * n), 2 * n * (S - Qp.real), -Qp.imag


def pinv_2x2_cart(S, Q):
    return tuple(c / 16 for c in fisher_pty_2x2_cart(S, Q))


def pinv_2x2_polar(S, Q, x):
    n = jnp.abs(x) ** 2
    Jnn, Jpp, Jnp_ = fisher_pty_2x2_polar(S, Q, x)
    return n**2 * Jnn, Jpp / (16 * n**2), Jnp_ / 4


def ls_fs_cov(probes, ys, pad, x, pupil=1.0):
    S, Q = ptycho_SQ(probes, ys, pad, pupil)
    D2 = jnp.sum(jnp.abs(probes) ** 2, 0) ** 2
    cart = tuple(c / D2 for c in pinv_2x2_cart(S, Q))
    polar = tuple(c / D2 for c in pinv_2x2_polar(S, Q, x))
    return cart, polar


@jax.jit
def diag_of_2x2_inverse(a, b, c, d):
    denom = a * d - b * c
    return d / (denom), a / denom
