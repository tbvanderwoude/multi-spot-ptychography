from master.fft import fft2_shifted, ifft2_shifted
from master.obj import crop_and_pad, normalize_image_intensity
from master.symmetry import choose_representative
from master.plotting import plot, plot_intensity_phase
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax


def analyse_probe(probes, x_obj, r_extent, r_tol=1e-2, single_plot=False):
    """Basic diagonstic plots for a probe stack, returning the probed area where the total sum of probe intensities is greater than
    a fraction of the maximum probe sum."""
    probe_sum = jnp.sum(jnp.abs(probes) ** 2, axis=0)
    object_probe_sum = jnp.sum(jnp.abs(probes * x_obj) ** 2, axis=0)
    probed = jnp.astype(probe_sum >= jnp.max(probe_sum) * r_tol, jnp.float64)
    if not single_plot:
        plot(probe_sum, r_extent)
        plot(probed, r_extent)
    plot(object_probe_sum, r_extent)
    return probed


def fourier_shift(probe, delta_x, delta_y, K_X, K_Y):
    shift = jnp.exp(1j * (delta_x * K_X + delta_y * K_Y))
    return ifft2_shifted(fft2_shifted(probe) * shift)


def create_probes_no_rescale(
    R_probes,
    f_probe,
    N_photons,
    cdtype,
    use_shift=True,
    K_X=None,
    K_Y=None,
    **kwargs,
):
    """f_probe should be a function of R_x, R_y and create a probe field. This is rescaled to match N_photons in total intensity."""
    M = len(R_probes)
    base_probe = f_probe(0.0, 0.0, **kwargs)
    probes = jnp.zeros((M, *base_probe.shape), dtype=cdtype)
    for m in range(M):
        R_x = R_probes[m, 0]
        R_y = R_probes[m, 1]
        if use_shift:
            unscaled_probe_m = fourier_shift(base_probe, R_x, R_y, K_X, K_Y)
        else:
            unscaled_probe_m = f_probe(R_x, R_y, **kwargs)
        probe_m = unscaled_probe_m * jnp.sqrt(N_photons)
        probes = probes.at[m, :].set(probe_m)
    return probes


def create_probes(R_probes, f_probe, N_photons, cdtype):
    """f_probe should be a function of R_x, R_y and create a probe field. This is rescaled to match N_photons in total intensity."""
    M = len(R_probes)
    base_probe = f_probe(0.0, 0.0)
    base_I = jnp.sum(jnp.abs(base_probe) ** 2)
    probes = jnp.zeros((M, *base_probe.shape), dtype=cdtype)
    for m in range(M):
        R_x = R_probes[m, 0]
        R_y = R_probes[m, 1]
        base_probe = f_probe(R_x, R_y)
        # Normalize probe intensity in a way that avoids complicating graph
        probe = base_probe * jnp.sqrt(N_photons / base_I)
        probes = probes.at[m, :].set(probe)
    return probes


def plot_R_probes(R_probes, r, label=r"$R_{probe}$"):
    plt.scatter(R_probes[:, 0] / r, R_probes[:, 1] / r, label=label)
    plt.xlabel(r"$R_{m,x}$ [p]")
    plt.ylabel(r"$R_{m,y}$ [p]")


def compare_estimate(x_est_raw, x_obj, N_compare_crop=100):
    # Break phase and if applicable twin symmetry
    x_est = choose_representative(x_est_raw, x_obj)
    # Compensate seemingly common global amplitude scale mismatches
    obj_crop, _ = crop_and_pad(x_obj, N_compare_crop, 0)
    est_crop, _ = crop_and_pad(x_est, N_compare_crop, 0)

    est_crop_norm = normalize_image_intensity(est_crop)
    obj_crop_norm = normalize_image_intensity(obj_crop)
    plot_intensity_phase(obj_crop_norm)
    plot_intensity_phase(est_crop_norm)
