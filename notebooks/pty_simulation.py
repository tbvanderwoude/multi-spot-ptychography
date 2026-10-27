import jax.numpy as jnp
from master.fft import *
import jax
import master.device as device
from master.mle_pty import (
    mle_ptycho_first_order_consensus,
    mle_ptycho_second_order_consensus,
)
from master.spotgen import create_multispot_array_product_jittered
from master.symmetry import *
from master.pies import *
from master.obj import *
from master.noise import *
from master.mle import *
from master.probes import *
import logging
import argparse
from master.mle import *
import math
from master.h5_io import *
from tqdm import tqdm
from pty_defs import ProbeParams, PtySim, cameraman_object, create_ops_and_pupil
from master.digest import make_array_digest


def is_interactive():
    # TOOD: add stackoverflow attribution. Also check if this is indeed exact required behavior for CLI / notebook differentiation
    import __main__ as main

    return not hasattr(main, "__file__")


RUN_ALL_ML = False
RUN_ALPHAS = True
TEST_LOOP = False
VARY_PHOTON_INTENSITY = True

logger = logging.getLogger(__name__)


def main():
    jax.config.update("jax_enable_x64", True)
    cdtype = jnp.array(1j).dtype
    jax.devices()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(consoleHandler)
    parser = argparse.ArgumentParser(
        prog="Ptychography stimator statistics simulator",
    )
    parser.add_argument("-d", "--device", type=int, default=1)
    parser.add_argument("-f", "--folder", type=str, default="cdi-data")
    parser.add_argument("-s", "--start_index", type=int, default=0)
    parser.add_argument("-e", "--end_index", type=int, default=128)
    parser.add_argument("-N", "--N_samples", type=int, default=128)
    parser.add_argument("-a", "--algo", type=str, default="rPIE")
    if is_interactive():
        device_id = 0
    else:
        args = parser.parse_args()
        device_id = args.device
        base_folder = args.folder
        start_index = args.start_index
        end_index = args.end_index
        # How many independent realizations of simulated CDI measurements to reconstruct
        N_samples = args.N_samples
        ap_algo = args.algo
    logger.info(f"Using device_id = {device_id}")
    device.setup_device(device_id=device_id)

    COMMON_RANDOM_NUMBERS = False

    # ====== Object parameters ======
    N_crop = 256
    N_pad = 32
    # How many samples to process in parallel on the GPU, function of VRAM available
    N_batch_samples_max = 8
    N_samples_device = end_index - start_index
    N_batch_samples = min(N_batch_samples_max, N_samples_device)

    # ====== Probe + detector parameters =====
    NA_det = 0.475
    NA_spot = 0.5

    frac_x = 0.80
    frac_y = 0.80
    L_x, L_y = 16, 16
    probe_type = "regular"
    probe_seed = 1776
    N_photons = 1e9
    M = 100

    x_obj, obj_support = cameraman_object(N_crop, N_pad)

    int_min = -1
    int_max = 10
    if VARY_PHOTON_INTENSITY:
        photon_intensities = [10.0**k for k in range(int_min, int_max + 1)]
    else:
        photon_intensities = [10.0**int_max]

    print(f"Running photon intensities: {photon_intensities}")
    # ====== Algorithm parameters ======
    N_iter_PIE = 1000
    # TODO: add rPIE parameters here
    epsilon_PIE = 1e-12
    beta_reg_PIE = 0.025
    gamma_PIE = 0.75
    alpha_PIE = 0.0125
    use_flat_init = True
    obj_support_init = obj_support
    obj_support_algo = 1.0
    spectral_support_algo = 1.0

    epsilon_ML = 1e-12
    # 1000 might have diminishing returns in terms of ell gains
    N_iter_ML = 1000
    seed_init = 1776
    seed_sample = 42

    base_key_algo = jax.random.PRNGKey(seed_init)
    base_key_sample = jax.random.PRNGKey(seed_sample)

    probe_params_dict = {
        "NA_spot": NA_spot,
        "frac_x": frac_x,
        "frac_y": frac_y,
        "L_x": L_x,
        "L_y": L_y,
        "probe_type": probe_type,
        "probe_seed": probe_seed,
    }
    probe_params = ProbeParams.from_dict(probe_params_dict)

    pty_sim = PtySim(N_crop, N_pad, NA_det)
    x, y, X, Y, R, k_0, k_x, k_y, K_X, K_Y, k, r_extent, k_extent = (
        pty_sim.create_axes()
    )
    A, A_dagger, F_pupil, pupil_support = pty_sim.get_ops_and_pupil()
    del pupil_support
    probes, N_photons, N_coeff = pty_sim.create_probe_with_average_intensity(
        M,
        probe_params,
        1e6,
        f_probe=create_multispot_array_product_jittered,
        delta_r_coeff=0.50,
        use_shift=True,
        r_frac_jitter=0.1,
    )
    _, spectral_support, P_spectral_support = pty_sim.construct_spectral_support_rel(
        probes[0]
    )
    x_obj_est = P_spectral_support(x_obj)
    canonicalize_est = lambda x_est: np.asarray(
        choose_representative_broadcast(P_spectral_support(x_est), x_obj_est)
    )
    det_sum_unscaled = np.sum(np.abs(A(probes * x_obj)) ** 2)
    N_int_to_photons = lambda N_int_avg: (
        float(N_crop**2 * N_int_avg),
        float(N_crop**2 * N_int_avg / det_sum_unscaled),
    )

    print(
        "Unit intensity probes and object digest: ",
        make_array_digest(probes),
        make_array_digest(x_obj),
    )
    for pixel_intensity in tqdm(photon_intensities):
        max_index = min(end_index, N_samples)
        for start_ix in range(start_index, max_index, N_batch_samples):
            end_ix = min(start_ix + N_batch_samples, max_index)
            N_batch_samples_actual = end_ix - start_ix
            print(start_ix, end_ix, N_batch_samples_actual)
            if TEST_LOOP:
                print("Testing the sample loop, skipping simulation.")
                continue

            probes, N_photons, N_coeff = pty_sim.create_probe_with_average_intensity(
                M,
                probe_params,
                pixel_intensity,
                f_probe=create_multispot_array_product_jittered,
                delta_r_coeff=0.50,
                use_shift=True,
                r_frac_jitter=0.1,
            )
            print(f"Running at N_photons = {N_photons:.3g}, N_coeff = {N_coeff}")
            print(
                "Rescaled intensity probes and N_coeff digest: ",
                make_array_digest(probes),
                make_array_digest(N_coeff),
            )
            ys = A(x_obj * probes)

            exp = int(math.log10(pixel_intensity))
            if COMMON_RANDOM_NUMBERS:
                # Use identical sample and initialization seeds across intensities to isolate effect of absolute scale
                # Reduces variance but is non-standard
                key_algo = base_key_algo
                key_sample = base_key_sample
                fold_data = -1
            else:
                # Standard Monte Carlo simulation: each pixel intensity setting gets its own
                fold_data = exp + 100 + start_ix * 1000
                key_algo = jax.random.fold_in(base_key_algo, fold_data)
                key_sample = jax.random.fold_in(base_key_sample, fold_data)

            measurement_dict = {
                "simulation": {
                    "N_crop": N_crop,
                    "N_pad": N_pad,
                    "N_photons": N_photons,
                    "N_samples": N_samples,
                    "M": M,
                    "NA_det": NA_det,
                },
                "probe": {**probe_params_dict, "N_coeff": N_coeff},
                "rng": {
                    "COMMON_RANDOM_NUMBERS": COMMON_RANDOM_NUMBERS,
                    "seed_init": seed_init,
                    "seed_sample": seed_sample,
                    "base_key_algo": key_data(base_key_algo),
                    "base_key_sample": key_data(base_key_sample),
                    "fold_data": fold_data,
                    "key_algo": key_data(key_algo),
                    "key_sample": key_data(key_sample),
                },
            }

            def create_simulation_dict_and_name(
                ap_algo=None, ml_algo=None, ML_alpha=1.0
            ):
                solver_dict = {
                    "pie_stage": {
                        "algorithm": ap_algo,
                        "alpha_PIE": alpha_PIE,
                        "gamma_PIE": gamma_PIE,
                        "epsilon_PIE": epsilon_PIE,
                        "beta_reg_PIE": beta_reg_PIE,
                        "use_flat_init": use_flat_init,
                    },
                    "ml_stage": {
                        "ml_algo": ml_algo,
                        "epsilon_ML": epsilon_ML,
                        "N_iter_ML": N_iter_ML,
                        "ML_alpha": ML_alpha,
                    },
                }
                if ap_algo is None:
                    solver_dict.pop("ap_stage")
                if ml_algo is None:
                    solver_dict.pop("ml_stage")
                return {
                    "solver": solver_dict,
                    **measurement_dict,
                }

            logger.info(
                f"Running estimator with N_photons={N_photons:.3g} or {pixel_intensity:.3g}/pixel"
            )

            # Sample field
            y_data = sample_from_mu(
                jnp.abs(ys) ** 2,
                batch_dim=N_batch_samples_actual,
                base_key=key_sample,
            )[1]
            d = jax.device_get(y_data**2)
            measurement_hash = config_hasher(measurement_dict)
            save_simulated_measurement_to_h5(
                f"{base_folder}/measurements/CDI-intensity-{pixel_intensity:.3g}-chunk-{start_ix}-{end_ix}.h5",
                x_obj,
                d,
                measurement_dict,
                measurement_hash,
                probes=probes,
                spectral_support=spectral_support,
                x_obj_est=x_obj_est,
            )

            # Select AP-based solver to perform initial solve, finding a log-likelihood 'basin' for FS to work in
            if ap_algo == "rPIE":
                logger.info("Running rPIE algorithm")
                ap_solver = pie_spectral_init
            elif ap_algo == "ORACLE":
                logger.info("Using ground truth oracle")
                ap_solver = lambda *args, **kwargs: x_obj
            else:
                raise ValueError(f"Algorithm {ap_algo} does not exist.")

            def saver(
                algo_name,
                ap_algo,
                ml_algo,
                x_model,
                lls=None,
                x_model_raw=None,
                ML_alpha=1.0,
            ):
                config = create_simulation_dict_and_name(ap_algo, ml_algo, ML_alpha)
                config_hash = config_hasher(config)
                save_estimate_to_h5(
                    f"{base_folder}/estimates/CDI-intensity-{pixel_intensity:.3g}-{algo_name}-chunk-{start_ix}-{end_ix}.h5",
                    x_model,
                    config,
                    config_hash,
                    measurement_hash,
                    lls,
                    x_model_raw,
                )

            # Run AP solver with some assumptions on its parameters
            x_model_pie = jnp.zeros((N_batch_samples_actual,) + x_obj_est.shape, cdtype)
            for sample_ix in tqdm(range(0, end_ix - start_ix)):
                y_complex = jnp.astype(y_data[sample_ix, :], cdtype)
                x_model_pie = x_model_pie.at[sample_ix, :].set(
                    ap_solver(
                        ys=y_complex,
                        probes=probes,
                        base_key=jax.random.fold_in(key_algo, start_ix + sample_ix),
                        alpha=alpha_PIE,
                        gamma=gamma_PIE,
                        epsilon=epsilon_PIE,
                        beta_reg=beta_reg_PIE,
                        N_iter=N_iter_PIE,
                        spectral_support=spectral_support_algo,
                        use_flat_init=use_flat_init,
                        obj_support_init=obj_support_init,
                        obj_support=obj_support_algo,
                        A=A,
                        A_dagger=A_dagger,
                    )
                )
            if ap_algo != "ORACLE":
                # No need to save redundant x_true copies
                saver(
                    ap_algo,
                    ap_algo,
                    None,
                    canonicalize_est(x_model_pie),
                    None,
                    x_model_pie,
                )

            logger.info("Running Fisher scoring starting from rPIE")

            print(x_model_pie.shape, probes.shape, y_data.shape, obj_support.shape)
            x_model_ml_raw, lls_ml_mean = mle_ptycho_second_order_consensus(
                x_model_pie,
                probes,
                y_data,
                obj_support=obj_support_algo,
                epsilon=epsilon_ML,
                N_iter=N_iter_ML,
                track_ll=True,
                A=A,
                A_dagger=A_dagger,
            )
            saver(
                f"{ap_algo}-second-order-ml",
                ap_algo,
                "FS",
                canonicalize_est(x_model_ml_raw),
                lls_ml_mean,
                x_model_ml_raw,
            )

            logger.info("Running first-order ML refinement starting from AP mean")

            N_alphas = 16
            alphas = jnp.logspace(-4, 1, N_alphas)
            logger.info(
                f"Running alphas: {" ".join([f"{alpha:.3g}" for alpha in alphas])}"
            )
            if RUN_ALPHAS:
                for ML_alpha in tqdm(alphas):
                    x_model_ml_raw, lls_ml_mean = mle_ptycho_first_order_consensus(
                        x_model_pie,
                        probes,
                        y_data,
                        obj_support=obj_support_algo,
                        epsilon=epsilon_ML,
                        N_iter=N_iter_ML,
                        track_ll=True,
                        alpha=ML_alpha,
                        A=A,
                        A_dagger=A_dagger,
                    )
                    saver(
                        f"{ap_algo}-first-order-ml-{ML_alpha:.3g}",
                        ap_algo,
                        "FO",
                        canonicalize_est(x_model_ml_raw),
                        lls_ml_mean,
                        x_model_ml_raw,
                        float(ML_alpha),
                    )


if __name__ == "__main__":
    main()
