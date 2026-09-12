"""Native launcher for the prepared Qwen3.8 Flash-Next SSD artifact."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import load_manifest

MODEL_REPO = "garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
SERVED_MODEL = "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
_STATE_VERSION = 1
_RTX_DEFAULT_CONTEXT = 131_072
_SPARK_DEFAULT_CONTEXT = 262_144
_PORTABLE_DEFAULT_CONTEXT = 16_384
_MIN_PORTABLE_GPU_MIB = 24_000
_EXPERT_OFFLOAD_GIB = 64
_HOST_RUNTIME_RESERVE_GIB = 16


@dataclass(frozen=True)
class Hardware:
    architecture: str
    gpu_name: str
    gpu_memory_mib: int
    compute_capability: tuple[int, int]


def _replace_symlink(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"cannot create the private CUDA view over {link}")
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target, target_is_directory=target.is_dir())
    os.replace(temporary, link)


def _packaged_cuda_root() -> Path:
    try:
        import nvidia
    except ImportError as exc:
        raise RuntimeError("the pinned NVIDIA CUDA packages are not installed") from exc

    for package_root in nvidia.__path__:
        candidate = Path(package_root) / "cu13"
        if (candidate / "bin/nvcc").is_file():
            return candidate.resolve()
    raise RuntimeError("cannot locate the pinned NVIDIA CUDA packages")


def _cuda_home() -> Path:
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    packaged = _packaged_cuda_root()
    runtime = Path(sys.prefix).resolve().parent
    view = runtime / "cuda"
    lib64 = view / "lib64"
    lib64.mkdir(parents=True, exist_ok=True)
    cudart = packaged / "lib/libcudart.so.13"
    required = (packaged / "bin", packaged / "include", cudart)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"the pinned CUDA packages are incomplete: {', '.join(missing)}"
        )
    _replace_symlink(view / "bin", packaged / "bin")
    _replace_symlink(view / "include", packaged / "include")
    _replace_symlink(lib64 / "libcudart.so", cudart)
    return view


def _require_compiler() -> None:
    compiler = os.environ.get("CXX", "c++")
    if shutil.which(compiler) is None:
        raise RuntimeError(
            "SGLang requires a C++ compiler for first-run kernels; "
            "install your Linux distribution's standard build tools"
        )


def _data_dir(value: str | None) -> Path:
    configured = value or os.environ.get("SGLANG_SSD_STREAM_HOME")
    root = Path(configured) if configured else Path.home() / ".cache/sglang-ssd-stream"
    return root.expanduser().resolve()


def _state_path(data_dir: Path) -> Path:
    return data_dir / "state.json"


def _read_state(data_dir: Path) -> dict:
    path = _state_path(data_dir)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if state.get("version") != _STATE_VERSION:
        raise RuntimeError(f"unsupported state file format in {path}")
    return state


def _write_state(data_dir: Path, state: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(data_dir)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"version": _STATE_VERSION, **state}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _hub(data_dir: Path):
    os.environ["HF_HOME"] = str(data_dir / "huggingface")
    from huggingface_hub import HfApi, snapshot_download

    return HfApi(), snapshot_download


def _latest_model_revision(data_dir: Path) -> str:
    api, _ = _hub(data_dir)
    revision = api.model_info(MODEL_REPO, revision="main").sha
    if not revision:
        raise RuntimeError(f"Hugging Face did not resolve {MODEL_REPO}@main")
    return revision


def _download_model(data_dir: Path, revision: str) -> Path:
    _, snapshot_download = _hub(data_dir)
    snapshot = Path(snapshot_download(repo_id=MODEL_REPO, revision=revision)).resolve()
    required = (
        snapshot / "ssd-stream.json",
        snapshot / "model.safetensors.index.json",
        snapshot / "mtp" / "model.safetensors.index.json",
    )
    missing = [
        str(path.relative_to(snapshot)) for path in required if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"prepared model snapshot {revision} is missing: {', '.join(missing)}"
        )
    load_manifest(snapshot / "ssd-stream.json")
    return snapshot


def _pinned_revision(data_dir: Path) -> str:
    state = _read_state(data_dir)
    if state.get("model_repo") == MODEL_REPO and state.get("model_revision"):
        return state["model_revision"]
    revision = _latest_model_revision(data_dir)
    _write_state(
        data_dir,
        {"model_repo": MODEL_REPO, "model_revision": revision},
    )
    return revision


def _detect_hardware() -> Hardware:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("NVIDIA driver and nvidia-smi are required") from exc
    gpus = [line.strip() for line in lines.splitlines() if line.strip()]
    if len(gpus) != 1:
        raise RuntimeError(
            f"automatic serving currently expects one GPU, found {len(gpus)}"
        )
    try:
        name, memory, capability = [part.strip() for part in gpus[0].rsplit(",", 2)]
        memory_mib = int(memory.strip())
        major, minor = (int(part) for part in capability.split(".", 1))
    except ValueError as exc:
        raise RuntimeError(f"cannot parse nvidia-smi output: {gpus[0]}") from exc
    return Hardware(platform.machine(), name, memory_mib, (major, minor))


def _is_rtx_pro_6000(hardware: Hardware) -> bool:
    return (
        hardware.architecture == "x86_64"
        and "RTX PRO 6000 Blackwell" in hardware.gpu_name
        and hardware.gpu_memory_mib >= 90_000
    )


def _is_dgx_spark(hardware: Hardware) -> bool:
    return (
        hardware.architecture in {"aarch64", "arm64"}
        and "GB10" in hardware.gpu_name
        and hardware.gpu_memory_mib >= 110_000
    )


def _rtx_pro_args(snapshot: Path, context: int) -> list[str]:
    return [
        "--trust-remote-code",
        "--model-path",
        str(snapshot),
        "--served-model-name",
        SERVED_MODEL,
        "--quantization",
        "modelopt_fp4",
        "--fp4-gemm-backend",
        "flashinfer_cutlass",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--page-size",
        "64",
        "--mamba-radix-cache-strategy",
        "extra_buffer_lazy",
        "--mamba-track-interval",
        "64",
        "--mamba-ssm-dtype",
        "float32",
        "--max-mamba-cache-size",
        "5",
        "--chunked-prefill-size",
        "4096",
        "--max-running-requests",
        "1",
        "--cuda-graph-max-bs-decode",
        "1",
        "--context-length",
        str(context),
        "--max-total-tokens",
        str(context),
        "--mem-fraction-static",
        "0.985",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-draft-model-path",
        str(snapshot / "mtp"),
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--allow-auto-truncate",
        "--enable-multimodal",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "qwen3_coder",
    ]


def _dgx_spark_args(snapshot: Path, context: int) -> list[str]:
    return [
        "--trust-remote-code",
        "--model-path",
        str(snapshot),
        "--served-model-name",
        SERVED_MODEL,
        "--quantization",
        "modelopt_fp4",
        "--fp4-gemm-backend",
        "flashinfer_cutlass",
        "--kv-cache-dtype",
        "bfloat16",
        "--page-size",
        "64",
        "--mamba-radix-cache-strategy",
        "extra_buffer_lazy",
        "--mamba-track-interval",
        "64",
        "--mamba-ssm-dtype",
        "float32",
        "--max-mamba-cache-size",
        "5",
        "--chunked-prefill-size",
        "8192",
        "--max-running-requests",
        "1",
        "--cuda-graph-max-bs-decode",
        "1",
        "--disable-prefill-cuda-graph",
        "--context-length",
        str(context),
        "--max-total-tokens",
        str(context),
        "--mem-fraction-static",
        "0.85",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-draft-model-path",
        str(snapshot / "mtp"),
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--allow-auto-truncate",
        "--enable-multimodal",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
    ]


def _supports_portable_nvfp4(hardware: Hardware) -> bool:
    return (
        hardware.architecture == "x86_64"
        and (8, 0) <= hardware.compute_capability < (13, 0)
    )


def _portable_args(
    hardware: Hardware,
    snapshot: Path,
    context: int,
) -> list[str]:
    kv_dtype = (
        "fp8_e4m3" if hardware.compute_capability >= (10, 0) else "bfloat16"
    )
    args = [
        "--trust-remote-code",
        "--model-path",
        str(snapshot),
        "--served-model-name",
        SERVED_MODEL,
        "--quantization",
        "modelopt_fp4",
        "--kv-cache-dtype",
        kv_dtype,
        "--page-size",
        "64",
        "--mamba-radix-cache-strategy",
        "extra_buffer_lazy",
        "--mamba-track-interval",
        "64",
        "--mamba-ssm-dtype",
        "float32",
        "--max-mamba-cache-size",
        "4",
        "--chunked-prefill-size",
        "1024",
        "--max-running-requests",
        "1",
        "--context-length",
        str(context),
        "--max-total-tokens",
        str(context),
        "--mem-fraction-static",
        "0.80",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--offload-group-size",
        "1",
        "--offload-num-in-group",
        "1",
        "--offload-prefetch-step",
        "1",
        "--offload-mode",
        "cpu",
        "--allow-auto-truncate",
        "--enable-multimodal",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "qwen3_coder",
    ]
    return args


def _profile_args(
    hardware: Hardware,
    snapshot: Path,
    context: int | None,
) -> list[str]:
    if _is_rtx_pro_6000(hardware):
        return _rtx_pro_args(snapshot, context or _RTX_DEFAULT_CONTEXT)
    if _is_dgx_spark(hardware):
        return _dgx_spark_args(snapshot, context or _SPARK_DEFAULT_CONTEXT)
    if _supports_portable_nvfp4(hardware):
        if hardware.gpu_memory_mib < _MIN_PORTABLE_GPU_MIB:
            raise RuntimeError(
                "the grouped CPU-offload profile requires at least 24 GiB of GPU memory"
            )
        return _portable_args(
            hardware,
            snapshot,
            context or _PORTABLE_DEFAULT_CONTEXT,
        )
    raise RuntimeError(
        f"no automatic profile for {hardware.gpu_name} "
        f"({hardware.gpu_memory_mib} MiB, {hardware.architecture})"
    )


def _available_host_memory_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("cannot determine available host memory") from exc
    raise RuntimeError("cannot determine available host memory")


def _require_host_memory(grouped_offload: bool) -> None:
    if not grouped_offload:
        return
    available = _available_host_memory_gib()
    required = _EXPERT_OFFLOAD_GIB + _HOST_RUNTIME_RESERVE_GIB
    if available < required:
        raise RuntimeError(
            f"CPU offload needs about {required} GiB available host memory; "
            f"only {available:.1f} GiB is available"
        )


def _serve(args: argparse.Namespace) -> None:
    _require_compiler()
    data_dir = _data_dir(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    hardware = _detect_hardware()
    grouped_offload = (
        _supports_portable_nvfp4(hardware)
        and not _is_rtx_pro_6000(hardware)
        and not _is_dgx_spark(hardware)
    )
    _require_host_memory(grouped_offload)
    revision = _pinned_revision(data_dir)
    snapshot = _download_model(data_dir, revision)
    profile_args = _profile_args(
        hardware,
        snapshot,
        args.context,
    )
    command = [sys.executable, "-m", "sglang.launch_server"]
    command.extend(profile_args)
    command.extend(["--host", args.host, "--port", str(args.port)])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    forwarded = args.sglang_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    command.extend(forwarded)

    print("Free 48 GB of RAM. Keep the speed.", flush=True)
    print(f"GPU: {hardware.gpu_name}", flush=True)
    print(
        "Compute capability: "
        f"{hardware.compute_capability[0]}.{hardware.compute_capability[1]}",
        flush=True,
    )
    if grouped_offload:
        print("CPU expert offload: enabled (experimental)", flush=True)
    print(f"Model: {MODEL_REPO}@{revision[:12]}", flush=True)
    print(f"Data: {data_dir}", flush=True)
    print(f"API: http://{args.host}:{args.port}/v1", flush=True)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).parent), environment.get("PATH", ""))
    )
    environment.setdefault("MAX_JOBS", "1")
    environment.setdefault("FLASHINFER_NVCC_THREADS", "1")
    environment.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "1")
    environment.setdefault("CUDA_HOME", str(_cuda_home()))
    environment["SGLANG_PLUGINS"] = "ssd_stream"
    os.execve(command[0], command, environment)


def _update_model(args: argparse.Namespace) -> None:
    data_dir = _data_dir(args.data_dir)
    current = _read_state(data_dir).get("model_revision")
    latest = _latest_model_revision(data_dir)
    if current == latest:
        print(f"Model is current: {latest}")
        return
    _download_model(data_dir, latest)
    _write_state(
        data_dir,
        {"model_repo": MODEL_REPO, "model_revision": latest},
    )
    print(f"Model updated: {current or 'none'} -> {latest}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sglang-ssd-stream")
    commands = parser.add_subparsers(dest="command")

    serve = commands.add_parser("serve", help="start the OpenAI-compatible server")
    serve.add_argument("--data-dir")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=30000)
    serve.add_argument("--api-key")
    serve.add_argument("--context", type=int)
    serve.add_argument("sglang_args", nargs=argparse.REMAINDER)
    serve.set_defaults(handler=_serve)

    update_model = commands.add_parser(
        "update-model", help="download and select the latest model artifact"
    )
    update_model.add_argument("--data-dir")
    update_model.set_defaults(handler=_update_model)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["serve", *sys.argv[1:]])
    try:
        args.handler(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"sglang-ssd-stream: {exc}\n")


if __name__ == "__main__":
    main()
