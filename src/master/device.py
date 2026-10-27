import socket
import jax
import subprocess


def setup_device(device_id=None):
    servername = socket.gethostname().split("-")[0]
    if servername == "pop":
        print("Running locally")
    else:
        print(f"Running on {servername} ImPhys server")

    def choose_gpu_with_nvidia_smi():
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        candidates = []
        for line in result.stdout.strip().splitlines():
            idx, free_mb = [x.strip() for x in line.split(",")]
            candidates.append((int(free_mb), int(idx)))

        free_mb, gpu_idx = max(candidates)
        gpu_count = len(candidates)
        return gpu_idx, free_mb, gpu_count

    gpu_idx, free_mb, gpu_count = choose_gpu_with_nvidia_smi()
    print(f"Using physical GPU {gpu_idx}/{gpu_count} with {free_mb/1024:.2f} GiB free")
    devices = jax.devices()
    print(f"Found {len(devices)}")
    if device_id is None:
        device_id = gpu_idx
    jax.config.update("jax_default_device", devices[device_id])
    return gpu_idx, free_mb, gpu_count
