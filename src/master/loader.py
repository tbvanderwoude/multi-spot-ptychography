from .h5_io import (
    load_estimate_from_h5,
    load_estimate_from_h5_with_x_est_raw,
    load_measurement_from_h5,
    load_measurement_from_h5_with_probes,
)
import os


class Loader:
    def __init__(self, base_path, probe_dataset="probes"):
        self.base_path = base_path
        self.est_names = os.listdir(base_path + "estimates")
        self.measurement_names = os.listdir(base_path + "measurements")
        self.probe_dataset = probe_dataset

    def get_measurement_path(self, id_str):
        try:
            path = (
                self.base_path
                + "measurements/"
                + list(
                    filter(
                        lambda x: id_str in x and "chunk" not in x,
                        self.measurement_names,
                    )
                )[0]
            )
            return path
        except:
            raise ValueError(f"Could not get path for {id_str}")

    def get_estimate_files(self, meas_id, est_id):
        joint_id = meas_id + "-" + est_id
        matches = list(
            filter(lambda x: joint_id in x and "chunk" not in x, self.est_names)
        )
        return matches

    def get_estimate_paths(self, meas_id, est_id):
        ids = self.get_estimate_files(meas_id, est_id)
        return [self.base_path + "estimates/" + m for m in ids]

    def get_estimate_path(self, meas_id, est_id):
        matches = self.get_estimate_paths(meas_id, est_id)
        n_matches = len(matches)
        if n_matches > 1:
            print("Warning: multiple matches!")
        return matches[0]

    def get_estimate_path_exact(self, meas_id, est_id):
        joint_id = meas_id + "-" + est_id
        file_path = joint_id + ".h5"
        path = self.base_path + "estimates/" + file_path
        if file_path in self.est_names:
            return path
        else:
            raise ValueError(f"Estimate {path} could not be found!")

    def get_measurement(self, id_str):
        meas_path = self.get_measurement_path(id_str)
        x_true, d, measurement_params, measurement_hash_meas = load_measurement_from_h5(
            meas_path
        )
        return x_true, d, measurement_params, measurement_hash_meas

    def get_measurement_with_probes(self, id_str):
        meas_path = self.get_measurement_path(id_str)
        (
            x_true,
            d,
            probes,
            spectral_support,
            x_obj_est,
            measurement_params,
            measurement_hash_meas,
        ) = load_measurement_from_h5_with_probes(
            meas_path, probe_dataset=self.probe_dataset
        )
        return (
            x_true,
            d,
            probes,
            spectral_support,
            x_obj_est,
            measurement_params,
            measurement_hash_meas,
        )

    def get_estimate(self, meas_id, est_id):
        est_path = self.get_estimate_path_exact(meas_id, est_id)
        return self.get_estimate_from_path(est_path)

    def get_estimate_from_path(self, est_path):
        x_est, simulation_params, lls, config_hash, measurement_hash_est = (
            load_estimate_from_h5(est_path)
        )
        return x_est, simulation_params, lls, config_hash, measurement_hash_est

    def get_estimate_with_x_est_raw(self, meas_id, est_id):
        est_path = self.get_estimate_path_exact(meas_id, est_id)
        return self.get_estimate_from_path_with_x_est_raw(est_path)

    def get_estimate_from_path_with_x_est_raw(self, est_path):
        x_est, simulation_params, lls, x_est_raw, config_hash, measurement_hash_est = (
            load_estimate_from_h5_with_x_est_raw(est_path)
        )
        return (
            x_est,
            simulation_params,
            lls,
            x_est_raw,
            config_hash,
            measurement_hash_est,
        )
