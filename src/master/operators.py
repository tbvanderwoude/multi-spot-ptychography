import jax.numpy as jnp
from jax import jit
from master.fft import *


def _window(N_full, N_crop):
    assert 0 < N_crop <= N_full
    # Computes the high and low indices of the crop window
    # in a way handling all parity combinations of N_full, N_crop
    lo = N_full // 2 - N_crop // 2
    hi = N_full - N_crop - lo
    assert (
        N_full // 2 - lo == N_crop // 2
    ), "DC bin (i.e. central bin) shifts after the round-trip"
    return lo, hi


def make_crop_pad(N_full, N_crop):
    lo, hi = _window(N_full, N_crop)
    sl = slice(lo, lo + N_crop)

    @jit
    def crop(x):
        """Crops the final two dimensions using the slice sl."""
        return x[..., sl, sl]

    @jit
    def pad(y):
        """Pads the final two dimensions in a way matching crop exactly."""
        pw = ((0, 0),) * (y.ndim - 2) + ((lo, hi), (lo, hi))
        return jnp.pad(y, pw)

    return crop, pad


def make_geometry(N_full, N_crop):
    crop, pad = make_crop_pad(N_full, N_crop)
    support = pad(jnp.ones((N_crop, N_crop)))
    return crop, pad, support


def make_pty_ops_and_support(N_full, N_crop, F_pupil):
    crop, pad, support = make_geometry(N_full, N_crop)
    cropped_pupil = crop(F_pupil)
    A = jit(lambda x: cropped_pupil * crop(fft2_shifted(x)))
    A_dagger = jit(lambda y: ifft2_shifted(pad(jnp.conjugate(cropped_pupil) * y)))
    return A, A_dagger, support
