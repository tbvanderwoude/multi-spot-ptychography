import subprocess
import json
import hashlib
from datetime import datetime
import h5py
import jax
from pathlib import Path


def config_hasher(params):
    s = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def get_git_commit_SHA():
    git_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return git_SHA


def key_data(k):
    return jax.random.key_data(k).tolist()


def make_parent(filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)


def save_simulated_measurement_to_h5(
    filename,
    x_true,
    d,
    measurement_params,
    measurement_hash,
    probes=None,
    spectral_support=None,
    x_obj_est=None,
):
    """Save an object estimate together with true object, noisy realization d, and key CDI parameters. Git commit SHA is included for reproducibility."""
    make_parent(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset(
            "x_true",
            data=x_true,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        f.create_dataset(
            "d",
            data=d,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        if probes is not None:
            # Old data has "x_est_raw" here, should be probes. So chunks will have x_est_raw
            f.create_dataset(
                "probes",
                data=probes,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        if spectral_support is not None:
            f.create_dataset(
                "spectral_support",
                data=spectral_support,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        if x_obj_est is not None:
            f.create_dataset(
                "x_obj_est",
                data=x_obj_est,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        f.attrs["measurement_params"] = json.dumps(measurement_params)
        # Crude reproducibility aids
        f.attrs["git_commit"] = get_git_commit_SHA()
        f.attrs["measurement_hash"] = measurement_hash
        f.attrs["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")


def save_estimate_to_h5(
    filename,
    x_hat,
    simulation_params,
    config_hash,
    measurement_hash,
    lls=None,
    x_hat_raw=None,
):
    """Save an object estimate together with true object, noisy realization d, and key CDI parameters. Git commit SHA is included for reproducibility."""
    make_parent(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset(
            "x_est",
            data=x_hat,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        if lls is not None:
            f.create_dataset(
                "lls",
                data=lls,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        if x_hat_raw is not None:
            f.create_dataset(
                "x_est_raw",
                data=x_hat_raw,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        f.attrs["simulation_params"] = json.dumps(simulation_params)
        # Crude reproducibility aids
        f.attrs["git_commit"] = get_git_commit_SHA()
        f.attrs["config_hash"] = config_hash
        f.attrs["measurement_hash"] = measurement_hash
        f.attrs["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")


def load_measurement_from_h5(filename, verbose=False):
    with h5py.File(filename, "r") as f:
        x_true = f["x_true"][()]
        d = f["d"][()]
        measurement_params = json.loads(f.attrs["measurement_params"])
        measurement_hash = f.attrs["measurement_hash"]

        if verbose:
            print(f"[{filename}]")
            print(f"  git_commit       = {f.attrs.get('git_commit')}")
            print(f"  measurement_hash = {measurement_hash}")
            print(f"  timestamp        = {f.attrs.get('timestamp')}")

    return x_true, d, measurement_params, measurement_hash


def load_measurement_from_h5_with_probes(
    filename, verbose=False, probe_dataset="probes"
):
    with h5py.File(filename, "r") as f:
        x_true = f["x_true"][()]
        d = f["d"][()]
        # Bug in saving logic has wrong name for "probes" dataset
        f_get_opt = lambda k: f[k][()] if k in f else None
        probes = f_get_opt(probe_dataset)
        spectral_support = f_get_opt("spectral_support")
        x_obj_est = f_get_opt("x_obj_est")
        measurement_params = json.loads(f.attrs["measurement_params"])
        measurement_hash = f.attrs["measurement_hash"]
        if verbose:
            print(f"[{filename}]")
            print(f"  git_commit       = {f.attrs.get('git_commit')}")
            print(f"  measurement_hash = {measurement_hash}")
            print(f"  timestamp        = {f.attrs.get('timestamp')}")

    return (
        x_true,
        d,
        probes,
        spectral_support,
        x_obj_est,
        measurement_params,
        measurement_hash,
    )


def load_estimate_from_h5(filename, verbose=False):
    with h5py.File(filename, "r") as f:
        x_est = f["x_est"][()]
        lls = f["lls"][()] if "lls" in f else None
        simulation_params = json.loads(f.attrs["simulation_params"])
        config_hash = f.attrs["config_hash"]
        measurement_hash = f.attrs["measurement_hash"]

        if verbose:
            print(f"[{filename}]")
            print(f"  git_commit       = {f.attrs.get('git_commit')}")
            print(f"  config_hash      = {config_hash}")
            print(f"  measurement_hash = {measurement_hash}")
            print(f"  timestamp        = {f.attrs.get('timestamp')}")

    return x_est, simulation_params, lls, config_hash, measurement_hash


def load_estimate_from_h5_with_x_est_raw(filename, verbose=False):
    with h5py.File(filename, "r") as f:
        x_est = f["x_est"][()]
        lls = f["lls"][()] if "lls" in f else None
        x_est_raw = f["x_est_raw"][()] if "x_est_raw" in f else None
        simulation_params = json.loads(f.attrs["simulation_params"])
        config_hash = f.attrs["config_hash"]
        measurement_hash = f.attrs["measurement_hash"]

        if verbose:
            print(f"[{filename}]")
            print(f"  git_commit       = {f.attrs.get('git_commit')}")
            print(f"  config_hash      = {config_hash}")
            print(f"  measurement_hash = {measurement_hash}")
            print(f"  timestamp        = {f.attrs.get('timestamp')}")

    return x_est, simulation_params, lls, x_est_raw, config_hash, measurement_hash
