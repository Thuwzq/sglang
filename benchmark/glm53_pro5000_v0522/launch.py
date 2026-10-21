"""Launch the recorded v0.5.22 minimal profile from an offline image."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/data1/models/GLM-5.3-Flash-fp8")
    )
    parser.add_argument(
        "--runtime-dir", type=Path, default=Path.home() / "glm5_3flash_runtime"
    )
    parser.add_argument("--container", default="glm53-v0522-tp8-ep1-bf16-minimal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    source = root / "python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py"
    runtime = args.runtime_dir.expanduser().resolve()
    model = args.model_dir.expanduser().resolve()
    server = json.loads((here / "server-args.json").read_text())
    command = [
        "docker",
        "run",
        "-d",
        "--pull",
        "never",
        "--name",
        args.container,
        "--gpus",
        "all",
        "--shm-size",
        "32g",
        "--ulimit",
        "nofile=65536:65536",
        "--ulimit",
        "memlock=-1:-1",
        "-p",
        "8000:8000",
        "--mount",
        f"type=bind,src={model},dst=/model,readonly",
        "--mount",
        f"type=bind,src={runtime / 'cache-v0522'},dst=/root/.cache",
        "--mount",
        f"type=bind,src={runtime},dst=/runtime",
    ]
    for value in [
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "PYTHONUNBUFFERED=1",
        "TRITON_CACHE_DIR=/root/.cache/triton",
        "CUDA_CACHE_PATH=/root/.cache/cuda",
        "NCCL_SOCKET_IFNAME=lo",
        "NCCL_IB_DISABLE=1",
        "OMP_NUM_THREADS=8",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM=0",
    ]:
        command.extend(["-e", value])
    command.extend(
        [
            "--mount",
            f"type=bind,src={source},dst=/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py,readonly",
            "--entrypoint",
            "python3",
            "lmsysorg/sglang:v0.5.22",
            *server,
        ]
    )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not model.is_dir() or not source.is_file():
        parser.error("Model directory or patched repository source is missing")
    image = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", "lmsysorg/sglang:v0.5.22"],
        text=True,
    ).strip()
    expected = json.loads((here / "validation-observed.json").read_text())["image_id"]
    if image != expected:
        parser.error(
            "Imported image differs from the recorded image; verify its build first"
        )
    exists = (
        subprocess.run(
            ["docker", "inspect", args.container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if exists:
        parser.error(
            "Container already exists; inspect it and use docker start to resume it"
        )
    (runtime / "cache-v0522").mkdir(parents=True, exist_ok=True)
    logs = runtime / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    profile = runtime / "serving/v0522-tp8-ep1-bf16-minimal"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "launch-command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)
    log = logs / "v0522-tp8-ep1-bf16-minimal-service.log"
    with log.open("ab", buffering=0) as output:
        subprocess.Popen(
            ["docker", "logs", "--timestamps", "-f", args.container],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"Container launched; wait for model warmup. Log: {log}")


if __name__ == "__main__":
    main()
