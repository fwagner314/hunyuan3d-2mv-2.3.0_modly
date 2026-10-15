"""Build a release wheel and its ABI/source/checksum metadata on a CUDA build host."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rasterizer_distribution import PROFILES, PYTHONS, source_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--output", type=Path, default=ROOT / "release-dist")
    args = parser.parse_args()
    import torch
    profile = PROFILES[args.profile]
    if sys.platform != "win32" or f"{sys.version_info.major}.{sys.version_info.minor}" not in PYTHONS:
        raise RuntimeError("This release builder requires Windows and Python 3.11.")
    if torch.__version__ != profile["torch"]:
        raise RuntimeError(f"Expected torch {profile['torch']}, got {torch.__version__}")
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = ROOT / "custom_rasterizer"
    shutil.rmtree(bundle / "build", ignore_errors=True)
    # No GPU auto-detection: every published wheel contains all advertised targets.
    env = os.environ.copy()
    arches = [f"{sm // 10}.{sm % 10}" for sm in profile["arches"]]
    arches[-1] += "+PTX"
    env["TORCH_CUDA_ARCH_LIST"] = ";".join(arches)
    env["MAX_JOBS"] = "2"
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["DISTUTILS_USE_SDK"] = "1"
    env["MODLY_RASTERIZER_VERSION"] = "0.2.0+" + args.profile
    subprocess.run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(args.output.resolve())], cwd=bundle, env=env, check=True)
    wheels = list(args.output.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Use an empty output directory for each build.")
    wheel = wheels[0]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
    subprocess.run([sys.executable, "-c", "import torch; import custom_rasterizer_kernel"], check=True)
    kernel_path = subprocess.check_output([sys.executable, "-c", "import torch, custom_rasterizer_kernel; print(custom_rasterizer_kernel.__file__)"], text=True).strip()
    cuobjdump = Path(env["CUDA_HOME"]) / "bin" / "cuobjdump.exe"
    embedded = subprocess.check_output([str(cuobjdump), "-lelf", kernel_path], text=True)
    targets = {int(sm) for sm in re.findall(r"\.sm_(\d+)\.cubin", embedded)}
    if targets != set(profile["arches"]):
        raise RuntimeError(f"Compiled targets {targets} do not match advertised targets {profile['arches']}")
    ptx = subprocess.check_output([str(cuobjdump), "-lptx", kernel_path], text=True)
    if f".sm_{profile['arches'][-1]}.ptx" not in ptx:
        raise RuntimeError("The wheel is missing its advertised PTX fallback.")
    entry = {"filename": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
             "source_sha256": source_digest(bundle), "python": f"{sys.version_info.major}.{sys.version_info.minor}",
             "profile": args.profile, **profile}
    (args.output / (wheel.stem + ".json")).write_text(json.dumps(entry, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
