import argparse
import os
import re
import numpy as np
import itertools
import operator
from master.h5_io import *


def main():
    parser = argparse.ArgumentParser(
        prog="Ptychography stimator statistics simulator",
    )
    parser.add_argument(
        "-f", "--folder", type=str, default="/data/multispot_pty/pty-32"
    )
    parser.add_argument("-c", "--chunk_size", type=int, default=8)
    args = parser.parse_args()
    dir = args.folder + "/"
    chunk_size = args.chunk_size
    pattern = re.compile(r"(.*)-chunk-(\d+)-(\d+).h5")
    for subdir in [
        "measurements/",
        "estimates/",
    ]:
        subdir_path = dir + subdir
        files = sorted(os.listdir(subdir_path))
        # Filter out fused chunk files
        filtered_files = list(filter(lambda f: pattern.match(f), files))
        print(filtered_files)
        groups_str = [tuple(pattern.search(f).groups()) + (f,) for f in filtered_files]
        groups_str = [(a, int(b), int(c), d) for (a, b, c, d) in groups_str]

        # Group chunks by name
        def accumulate(l):
            it = itertools.groupby(l, operator.itemgetter(0))
            for key, subiter in it:
                yield key, list(item[1:] for item in subiter)

        measurement_chunks: dict = dict(accumulate(groups_str))
        chunk_count = None
        for measurement_name in measurement_chunks:
            print(measurement_name)
            # Does not check if there are any gaps!
            min_ix = min([el[0] for el in measurement_chunks[measurement_name]])
            max_ix = max([el[1] for el in measurement_chunks[measurement_name]])
            # Ensure that number of chunks (assumed to be equal in size) is consistent
            if chunk_count is None:
                chunk_count = len(measurement_chunks[measurement_name])
                print(f"Chunk count is {chunk_count}")
            else:
                assert chunk_count == len(measurement_chunks[measurement_name])
            N_samples = max_ix - min_ix
            print(N_samples)
            if subdir == "measurements/":
                d_fused = None
                probes_measurement = None
                spectral_support_measurement = None
                x_obj_est_measurement = None
                print(min_ix, max_ix)
                for start_ix, end_ix, path in measurement_chunks[measurement_name]:
                    if end_ix - start_ix != chunk_size:
                        continue
                    print(path)
                    (
                        x_true,
                        d,
                        probes,
                        spectral_support,
                        x_obj_est,
                        measurement_params,
                        measurement_hash,
                    ) = load_measurement_from_h5_with_probes(subdir_path + path)
                    if d_fused is None:
                        first_measurement_params = measurement_params
                        first_measurement_hash = measurement_hash
                        if "M" in measurement_params["simulation"]:
                            # This is a ptychography measurement
                            M = measurement_params["simulation"]["M"]
                            d_fused = np.zeros(
                                (
                                    N_samples,
                                    M,
                                )
                                + d.shape[-2:],
                                dtype=np.float64,
                            )
                        else:
                            d_fused = np.zeros(
                                (N_samples,) + d.shape[-2:], dtype=np.float64
                            )
                    d_fused[start_ix:end_ix, :] = d
                    if probes is not None:
                        # No need to 'fuse' probes. Probes values should all be the same anyway
                        if probes_measurement is None:
                            probes_measurement = probes
                    if spectral_support is not None:
                        if spectral_support_measurement is None:
                            spectral_support_measurement = spectral_support
                    if x_obj_est is not None:
                        if x_obj_est_measurement is None:
                            x_obj_est_measurement = x_obj_est
                    print(x_true.shape, d.shape)
                save_simulated_measurement_to_h5(
                    subdir_path + measurement_name + ".h5",
                    x_true,
                    d_fused,
                    first_measurement_params,
                    first_measurement_hash,
                    probes=probes_measurement,
                    spectral_support=spectral_support_measurement,
                    x_obj_est=x_obj_est_measurement,
                )
            elif subdir == "estimates/":
                x_est_fused = None
                x_est_raw_fused = None
                lls_fused = None
                print(min_ix, max_ix)
                for start_ix, end_ix, path in measurement_chunks[measurement_name]:
                    if end_ix - start_ix != chunk_size:
                        continue
                    print(path)
                    (
                        x_est,
                        simulation_params,
                        lls,
                        x_est_raw,
                        config_hash,
                        measurement_hash,
                    ) = load_estimate_from_h5_with_x_est_raw(subdir_path + path)
                    print(x_est.shape)
                    if x_est_fused is None:
                        first_config_hash = config_hash
                        first_measurement_hash = measurement_hash
                        averaged = len(x_est.shape) == 3
                        if not averaged:
                            x_est_fused = np.zeros(
                                (
                                    x_est.shape[0],
                                    N_samples,
                                )
                                + x_est.shape[-2:],
                                dtype=np.complex128,
                            )
                            x_est_raw_fused = np.zeros(
                                (
                                    x_est_raw.shape[0],
                                    N_samples,
                                )
                                + x_est.shape[-2:],
                                dtype=np.complex128,
                            )
                            if lls is not None:
                                lls_fused = np.zeros(
                                    (
                                        x_est.shape[0],
                                        N_samples,
                                    )
                                    + lls.shape[-1:],
                                    dtype=np.float64,
                                )
                        else:
                            x_est_fused = np.zeros(
                                (N_samples,) + x_est.shape[-2:], dtype=np.complex128
                            )
                            x_est_raw_fused = np.zeros(
                                (N_samples,) + x_est_raw.shape[-2:], dtype=np.complex128
                            )
                            if lls is not None:
                                lls_fused = np.zeros(
                                    (
                                        lls.shape[0],
                                        N_samples,
                                    ),
                                    dtype=np.float64,
                                )
                        # Handle leading N_inits
                    if averaged:
                        x_est_fused[start_ix:end_ix] = x_est
                        x_est_raw_fused[start_ix:end_ix] = x_est_raw
                    else:
                        x_est_fused[:, start_ix:end_ix] = x_est
                        x_est_raw_fused[:, start_ix:end_ix] = x_est_raw
                    if lls is not None:
                        print(lls.shape, x_est.shape, lls_fused.shape)
                        lls_fused[:, start_ix:end_ix] = lls
                save_estimate_to_h5(
                    subdir_path + measurement_name + ".h5",
                    x_est_fused,
                    simulation_params,
                    first_config_hash,
                    first_measurement_hash,
                    lls_fused,
                    x_est_raw_fused,
                )
        # print(chunk_dict)
    print("Fusing chunks in directory")


if __name__ == "__main__":
    main()
