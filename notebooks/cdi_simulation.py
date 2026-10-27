import jax.numpy as jnp
from master.fft import *
import jax
import master.device as device
from master.symmetry import *
from master.phase_retrieval import *
from master.obj import *
from master.noise import *
from master.mle import *
import logging
import argparse
from master.mle import *
import math
from master.h5_io import *
from tqdm import tqdm


def is_interactive():
    # TOOD: add stackoverflow attribution. Also check if this is indeed exact required behavior for CLI / notebook differentiation
    import __main__ as main

    return not hasattr(main, "__file__")


RUN_ALL_ML = False
TEST_LOOP = False

logger = logging.getLogger(__name__)


def main():
    jax.config.update("jax_enable_x64", True)
    jax.devices()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(consoleHandler)
    parser = argparse.ArgumentParser(
        prog="Estimator statistics simulator",
    )
    parser.add_argument("-d", "--device", type=int, default=1)
    parser.add_argument("-f", "--folder", type=str, default="cdi-data")
    parser.add_argument("-s", "--start_index", type=int, default=0)
    parser.add_argument("-e", "--end_index", type=int, default=0)
    parser.add_argument("-N", "--N_samples", type=int, default=128)
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
    logger.info(f"Using device_id = {device_id}")
    device.setup_device(device_id=device_id)

    COMMON_RANDOM_NUMBERS = False
    N_crop = 256
    N_pad = 128
    # How many samples to process in parallel on the GPU, function of VRAM available
    N_batch_samples = 64
    N_iter_AP = 1000
    beta_AP = 0.9
    epsilon_AP = 1e-7
    epsilon_ML = 1e-12
    # 1000 might have diminishing returns in terms of ell gains
    N_iter_ML = 1000
    # How many independent random solver initializations to apply to each sample
    N_inits = 16
    seed_init = 1776
    seed_sample = 42

    base_key_algo = jax.random.PRNGKey(seed_init)
    base_key_sample = jax.random.PRNGKey(seed_sample)
    ap_algo = "HIO"

    x_cameraman = cameraman() * 0.9 + 0.1
    x_mandrill = as_phase_field(mandrill() ** 2 - 0.5)
    x_obj, obj_support = crop_and_pad(
        x_cameraman * x_mandrill, N_crop=N_crop, N_pad=N_pad
    )

    x_obj = normalize_image_intensity(x_obj)

    # Complex linear forward operator A and adjoint
    A = fft2_shifted
    A_dagger = ifft2_shifted
    yhat = A(x_obj)

    vary_photon_intensity = True
    if vary_photon_intensity:
        photon_intensities = [10.0**k for k in range(-1, 11)]
    else:
        photon_intensities = [10.0**9]
    for pixel_intensity in tqdm(photon_intensities):
        max_index = min(end_index, N_samples)
        for start_ix in range(start_index, max_index, N_batch_samples):
            # print(start_ix)
            end_ix = min(start_ix + N_batch_samples, max_index)
            print(start_ix, end_ix, end_ix - start_ix)
            if TEST_LOOP:
                print("Testing the sample loop, skipping simulation.")
                continue
            N_photons = pixel_intensity * (N_crop**2)
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
                },
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
                ap_algo=None, ml_algo=None, ML_alpha=1.0, averaged=False
            ):
                solver_dict = {
                    "ap_stage": {
                        "algorithm": ap_algo,
                        "N_iter_AP": N_iter_AP,
                        "beta_AP": beta_AP,
                        "epsilon_AP": epsilon_AP,
                        "N_inits": N_inits,
                        "averaged": averaged,
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

            print(
                f"Running estimator with N_photons={N_photons:.3g} or {pixel_intensity:.3g}/pixel"
            )

            # Sample field
            yhat_mu, sqrt_d = sample_field(
                yhat,
                N_photons=N_photons,
                batch_dim=N_batch_samples,
                base_key=key_sample,
            )
            d = jax.device_get(sqrt_d**2)
            x_true = x_obj * jnp.sqrt(N_photons)

            measurement_hash = config_hasher(measurement_dict)
            save_simulated_measurement_to_h5(
                f"{base_folder}/measurements/CDI-intensity-{pixel_intensity:.3g}-chunk-{start_ix}-{end_ix}.h5",
                x_true,
                d,
                measurement_dict,
                measurement_hash,
            )

            # Select AP-based solver to perform initial solve, finding a log-likelihood 'basin' for FS to work in
            if ap_algo == "HIO":
                logger.info("Running Hybrid Input-Output algorithm")
                ap_solver = hio
            elif ap_algo == "DM":
                logger.info("Running Difference Map algorithm")
                ap_solver = difference_map
            elif ap_algo == "ORACLE":
                logger.info("Using ground truth oracle")
                ap_solver = lambda *args, **kwargs: (
                    jnp.broadcast_to(
                        x_true,
                        (N_inits, N_batch_samples) + x_true.shape,
                    ),
                    True,
                    0.0,
                )
            else:
                raise ValueError(f"Algorithm {ap_algo} does not exist.")

            def saver(
                algo_name, ap_algo, ml_algo, averaged, x_model, lls=None, ML_alpha=1.0
            ):
                config = create_simulation_dict_and_name(
                    ap_algo, ml_algo, ML_alpha, averaged
                )
                config_hash = config_hasher(config)
                save_estimate_to_h5(
                    f"{base_folder}/estimates/CDI-intensity-{pixel_intensity:.3g}-{algo_name}-chunk-{start_ix}-{end_ix}.h5",
                    x_model,
                    config,
                    config_hash,
                    measurement_hash,
                    lls,
                )

            # Run AP solver with some assumptions on its parameters
            x_model_ap_raw, converged, err_L2 = ap_solver(
                sqrt_d,
                obj_support,
                A=A,
                A_dagger=A_dagger,
                restarts=1,
                epsilon=epsilon_AP,
                iterations=N_iter_AP,
                stagnate_rel_delta=1e-12,
                beta=beta_AP,
                base_key=key_algo,
                batch_dim=N_inits,
            )
            logger.info("Choosing optimal representatives")
            x_model_ap = choose_representative_broadcast(
                x_model_ap_raw * obj_support, x_true
            )
            del x_model_ap_raw
            if ap_algo != "ORACLE":
                # No need to save redundant x_true copies
                saver(ap_algo, ap_algo, None, False, x_model_ap)
                # Running ML on all initializations also becomes meaningless in this case
                if RUN_ALL_ML:
                    logger.info("Running Fisher scoring to get MLE estimates")
                    print(x_model_ap.shape[:-2])
                    x_model_ml_raw, lls_ml = ml_fisher_scoring_unitary_A(
                        x_model_ap,
                        sqrt_d,
                        obj_support,
                        epsilon=epsilon_ML,
                        N_iter=N_iter_ML,
                        track_ll=True,
                        A=A,
                        A_dagger=A_dagger,
                    )
                    x_model_ml = (
                        choose_representative_broadcast(
                            x_model_ml_raw * obj_support, x_true
                        )
                        * obj_support
                    )

                    saver(
                        f"{ap_algo}-second-order-ml",
                        ap_algo,
                        "FS",
                        False,
                        x_model_ml,
                        lls_ml,
                    )

            logger.info("Averaging AP runs to generate robust initialization")
            x_model_ap_mean = choose_representative_broadcast(
                jnp.mean(x_model_ap, axis=[0]), x_true
            )

            del x_model_ap
            if ap_algo != "ORACLE":
                # No need to save 'average' of ground truth initializations
                saver(f"{ap_algo}-averaged", ap_algo, None, True, x_model_ap_mean)

            logger.info("Running Fisher scoring starting from AP mean")

            x_model_ml_raw_from_mean, lls_ml_mean = ml_fisher_scoring_unitary_A(
                x_model_ap_mean,
                sqrt_d,
                obj_support,
                epsilon=epsilon_ML,
                N_iter=N_iter_ML,
                track_ll=True,
                A=A,
                A_dagger=A_dagger,
            )
            x_model_ml_from_mean = (
                choose_representative_broadcast(
                    x_model_ml_raw_from_mean * obj_support, x_true
                )
                * obj_support
            )
            saver(
                f"{ap_algo}-averaged-second-order-ml",
                ap_algo,
                "FS",
                True,
                x_model_ml_from_mean,
                lls_ml_mean,
            )

            logger.info("Running first-order ML refinement starting from AP mean")

            N_alphas = 16
            alphas = jnp.logspace(-4, 1, N_alphas)
            for ML_alpha in tqdm(alphas):
                x_model_ml_raw_from_mean, lls_ml_mean = ml_refinement_first_order(
                    x_model_ap_mean,
                    sqrt_d,
                    obj_support,
                    epsilon=epsilon_ML,
                    N_iter=N_iter_ML,
                    track_ll=True,
                    alpha=ML_alpha,
                    A=A,
                    A_dagger=A_dagger,
                )
                x_model_ml_from_mean = (
                    choose_representative_broadcast(
                        x_model_ml_raw_from_mean * obj_support, x_true
                    )
                    * obj_support
                )
                saver(
                    f"{ap_algo}-averaged-first-order-ml-{ML_alpha:.3g}",
                    ap_algo,
                    "FO",
                    True,
                    x_model_ml_from_mean,
                    lls_ml_mean,
                    float(ML_alpha),
                )


if __name__ == "__main__":
    main()
