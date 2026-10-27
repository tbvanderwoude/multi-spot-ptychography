from jax import jit
import numpy as np
import jax.numpy as jnp
from scipy.stats import qmc
from .fft import fft2_shifted, ifft2_shifted
from .ptychography import create_probes_no_rescale
from .spotgen import create_multispot_array_product
from .imaging import make_circular_mask


def create_probe_Rm(M, delta_r, seed, probe_type):
    """Previous API assumes this factor 2, otherwise old simulations change."""
    return create_probe_Rm_corrected(M, 2.0 * delta_r, seed, probe_type)


def create_probe_Rm_corrected(M, delta_r, seed, probe_type):
    match probe_type:
        case "sobol":

            def create_R_probes_Sobol(M, delta_r, seed=seed):
                sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
                u = sampler.random_base2(m=int(np.log2(M)))
                deltas = 1.0 * delta_r * u - 0.5 * delta_r
                return np.asarray(deltas)

            R_probes = create_R_probes_Sobol(M, delta_r, seed)
        case "regular":
            M_sqrt = int(np.sqrt(M))
            assert M_sqrt**2 == M

            x = np.linspace(0.0, delta_r * (1 - 1 / M_sqrt), M_sqrt) - (
                (0.5 - 0.5 / M_sqrt) * delta_r
            )
            y = np.linspace(0.0, delta_r * (1 - 1 / M_sqrt), M_sqrt) - (
                (0.5 - 0.5 / M_sqrt) * delta_r
            )

            X, Y = np.meshgrid(x, y, indexing="xy")

            R_probes = np.stack([X.ravel(), Y.ravel()], axis=1)
        case "uniform":
            np.random.seed(seed)
            R_probes = np.random.uniform(
                low=-delta_r / 2.0, high=delta_r / 2.0, size=(M, 2)
            )
        case _:
            raise ValueError(f"Unknown probe_type {probe_type}")
    return R_probes


def calc_array_properties(L_x, L_y, frac_x, frac_y, r_extent):
    min_x, max_x, min_y, max_y = r_extent
    array_bounds = min_x * frac_x, max_x * frac_x, min_y * frac_y, max_y * frac_y
    array_period_x = (array_bounds[1] - array_bounds[0]) / (L_x - 1)
    array_period_y = (array_bounds[3] - array_bounds[2]) / (L_y - 1)
    return array_bounds, array_period_x, array_period_y


def construct_probe_fields(
    probe_type,
    probe_seed,
    NA_spot,
    L_x,
    L_y,
    frac_x,
    frac_y,
    r_extent,
    M,
    N_probe_photons,
    K_X,
    K_Y,
    k,
    k_0,
    cdtype,
    f_probe=create_multispot_array_product,
    delta_r_coeff=1.0,
    use_shift=True,
    **kwargs,
):
    f_ms, R_probes = construct_probe_implicit(
        probe_type,
        probe_seed,
        NA_spot,
        L_x,
        L_y,
        frac_x,
        frac_y,
        r_extent,
        M,
        K_X,
        K_Y,
        k,
        k_0,
        f_probe,
        delta_r_coeff=delta_r_coeff,
    )
    probes = create_probes_no_rescale(
        R_probes,
        f_ms,
        N_probe_photons,
        cdtype,
        use_shift,
        K_X,
        K_Y,
        **kwargs,
    )
    return probes, f_ms, R_probes


# TODO: start using this to avoid useless materialization of all probe fields AND all probe object fields
def construct_probe_implicit(
    probe_type,
    probe_seed,
    NA_spot,
    L_x,
    L_y,
    frac_x,
    frac_y,
    r_extent,
    M,
    K_X,
    K_Y,
    k,
    k_0,
    f_probe=create_multispot_array_product,
    delta_r_coeff=1.0,
):
    array_bounds, array_period_x, array_period_y = calc_array_properties(
        L_x, L_y, frac_x, frac_y, r_extent
    )
    print(f"array_period_x={array_period_x:.3g}, array_period_y={array_period_y:.3g}")
    # Define 25 probe positions using random distribution in +-3 p_x central area
    r = array_period_x
    delta_r = r * delta_r_coeff

    R_probes = create_probe_Rm(
        M,
        delta_r,
        probe_seed,
        probe_type,
    )

    # Maps probe shift (R_x, R_y) to a probe field
    f_ms = lambda R_x, R_y, **kwargs: f_probe(
        NA_spot,
        array_bounds[0] + R_x,
        array_bounds[1] + R_x,
        array_bounds[2] + R_y,
        array_bounds[3] + R_y,
        L_x,
        L_y,
        K_X,
        K_Y,
        k,
        k_0,
        **kwargs,
    )
    return f_ms, R_probes


def build_spectral_support(probe, F_pupil, threshold=5):
    """Spectral support of the ptychography measurement is a function of the detector NA and probe spectrum."""
    PSF = ifft2_shifted(F_pupil)
    # Real-space product gives the spectral convolution
    filtered_probe = ifft2_shifted(jnp.abs(fft2_shifted(probe))) * PSF
    spectral_support = (fft2_shifted(filtered_probe) > threshold).astype(
        np.real(filtered_probe).dtype
    )
    return filtered_probe, spectral_support


def build_spectral_support_rel(
    probe, F_pupil, rel_threshold=1e-5, cross_correlation=False
):
    PSF = ifft2_shifted(F_pupil)
    # Real-space product gives the spectral convolution
    if cross_correlation:
        filtered_probe = ifft2_shifted(jnp.abs(fft2_shifted(probe))) * PSF
    else:
        filtered_probe = jnp.conj(ifft2_shifted(jnp.abs(fft2_shifted(probe)))) * PSF
    spectrum_filtered_probe = fft2_shifted(filtered_probe)
    mx = np.max(spectrum_filtered_probe)
    threshold = mx * rel_threshold
    spectral_support = (spectrum_filtered_probe > threshold).astype(
        np.real(filtered_probe).dtype
    )
    return spectrum_filtered_probe, spectral_support


def build_spectral_support_analytical(k, k_0, NA_synth):
    spectral_support = make_circular_mask(k, k_0 * NA_synth)
    return spectral_support


def construct_probe_inf_probe_approx(base_probe, F_pupil):
    # Here be dragons... Needs to be refactored to just iterate straight over bin indices within NA_spot to compute aperture
    PSF = ifft2_shifted(F_pupil)
    # Real-space product gives the spectral convolution
    filtered_probe = ifft2_shifted(jnp.abs(fft2_shifted(base_probe))) * PSF
    abs_p_spect = np.abs(fft2_shifted(base_probe))
    abs_p_max = np.max(abs_p_spect)
    p_inf_spect = (abs_p_spect > abs_p_max * 0.25).astype(np.real(filtered_probe).dtype)
    nonzeros = np.nonzero(p_inf_spect)
    real_p = ifft2_shifted(p_inf_spect)
    N_nonzero_orders = len(nonzeros[0])
    filtered_probe = real_p * PSF
    f_probe = jnp.abs(fft2_shifted(filtered_probe))
    f_probe_max = jnp.max(f_probe)
    spectral_support = (f_probe > f_probe_max * 0.00001).astype(
        np.real(filtered_probe).dtype
    )
    return p_inf_spect, N_nonzero_orders, spectral_support
