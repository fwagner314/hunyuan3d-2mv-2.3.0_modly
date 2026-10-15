"""Versioned, verified native wheels for compiler-free Windows installs."""
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "rasterizer-v1"
RELEASE_URL = "https://github.com/iammojogo-sudo/hunyuan3d-2mv-2.3.0_modly/releases/download/" + RELEASE
PYTHONS = ("3.11",)
PROFILES = {
    "cu124": {"torch": "2.6.0+cu124", "arches": [50, 52, 60, 61, 70, 75, 80, 86, 89, 90]},
    "cu128": {"torch": "2.7.0+cu128", "arches": [100, 120]},
}


def profile_for_gpu(gpu_sm):
    for name, profile in PROFILES.items():
        if gpu_sm in profile["arches"]:
            return name
    raise RuntimeError(f"No tested Windows rasterizer for compute capability {gpu_sm}. "
                       "This GPU needs a separately tested release; installing compilers is not required by default.")


def source_digest(root):
    """Reject binaries built from different native or Python wrapper sources."""
    root = Path(root)
    files = [root / "setup.py", *sorted(p for p in (root / "lib").rglob("*") if p.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh")),
             *sorted((root / "custom_rasterizer").rglob("*.py"))]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def runtime_info(python_exe):
    code = "import json,sys,platform; print(json.dumps({'python':f'{sys.version_info.major}.{sys.version_info.minor}','machine':platform.machine(),'bits':64 if sys.maxsize>2**32 else 32}))"
    return json.loads(subprocess.check_output([str(python_exe), "-c", code], text=True))


def select_wheel(index, runtime, profile, digest):
    if runtime["python"] not in PYTHONS or runtime["machine"].lower() not in ("amd64", "x86_64") or runtime["bits"] != 64:
        raise RuntimeError("Prebuilt Windows rasterizers require Modly's 64-bit Python 3.11 runtime. "
                           "Modly must provide a supported Python runtime.")
    if index.get("version") != 1:
        raise RuntimeError("Unsupported rasterizer release index.")
    matches = [entry for entry in index.get("wheels", [])
               if entry.get("python") == runtime["python"] and entry.get("profile") == profile
               and entry.get("torch") == PROFILES[profile]["torch"]
               and entry.get("arches") == PROFILES[profile]["arches"]
               and entry.get("source_sha256") == digest]
    if len(matches) != 1:
        raise RuntimeError("The maintainer has not published exactly one matching rasterizer wheel. "
                           "No local compilation was attempted.")
    entry = matches[0]
    filename = entry.get("filename", "")
    sha = entry.get("sha256", "")
    if not filename.endswith(".whl") or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise RuntimeError("Invalid rasterizer wheel filename.")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise RuntimeError("Invalid rasterizer wheel checksum.")
    return entry


def resolve_wheel(python_exe, ext_dir, gpu_sm):
    runtime = runtime_info(python_exe)
    profile = profile_for_gpu(gpu_sm)
    # Validate platform before making a request or installing dependencies.
    if runtime["python"] not in PYTHONS or runtime["machine"].lower() not in ("amd64", "x86_64") or runtime["bits"] != 64:
        raise RuntimeError("Modly must use 64-bit Python 3.11 for the bundled rasterizer.")
    local_root = Path(ext_dir) / "rasterizer_wheels"
    local_index = local_root / "index.json"
    if local_index.is_file():
        index = json.loads(local_index.read_text(encoding="utf-8"))
        if any(entry.get("python") == runtime["python"] and entry.get("profile") == profile for entry in index.get("wheels", [])):
            entry = select_wheel(index, runtime, profile, source_digest(Path(ext_dir) / "custom_rasterizer"))
            return {**entry, "_bundled": True}
    try:
        with urllib.request.urlopen(RELEASE_URL + "/index.json", timeout=60) as response:
            index = json.load(response)
    except Exception as exc:
        raise RuntimeError("The rasterizer release could not be downloaded. Check the connection; "
                           "the maintainer must publish the matching release before this install can succeed.") from exc
    return select_wheel(index, runtime, profile, source_digest(Path(ext_dir) / "custom_rasterizer"))


def install_wheel(venv, ext_dir, entry):
    py = Path(venv) / "Scripts" / "python.exe"
    # Dependencies must not have upgraded the ABI after initial torch install.
    actual = subprocess.check_output([str(py), "-c", "import torch; print(torch.__version__)"], text=True).strip()
    if actual != entry["torch"]:
        raise RuntimeError(f"Rasterizer requires torch {entry['torch']}, found {actual}.")
    with tempfile.TemporaryDirectory(prefix="modly-rasterizer-") as temporary:
        wheel = Path(temporary) / entry["filename"]
        digest = hashlib.sha256()
        source = (Path(ext_dir) / "rasterizer_wheels" / entry["filename"]).open("rb") if entry.get("_bundled") else urllib.request.urlopen(RELEASE_URL + "/" + entry["filename"], timeout=120)
        with source as response, wheel.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise RuntimeError("Rasterizer download checksum mismatch; installation stopped.")
        with zipfile.ZipFile(wheel) as archive:
            kernels = [name for name in archive.namelist()
                       if "/" not in name and name.startswith("custom_rasterizer_kernel") and name.endswith(".pyd")]
            if len(kernels) != 1:
                raise RuntimeError("Rasterizer wheel must contain exactly one native kernel.")
            site_packages = Path(venv) / "Lib" / "site-packages"
            old_pth = site_packages / "hy3d_custom_rasterizer.pth"
            if old_pth.exists():
                old_pth.unlink()
            old_bundle = site_packages / "custom_rasterizer"
            if (old_bundle / "setup.py").is_file():
                shutil.rmtree(old_bundle)
            subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
            bundle = Path(ext_dir) / "custom_rasterizer"
            for stale in bundle.glob("custom_rasterizer_kernel*.pyd"):
                stale.unlink()
            # Keep the selected release kernel in the extension for bridge repair.
            (bundle / kernels[0]).write_bytes(archive.read(kernels[0]))
    print("[setup] GPU launch validation deferred until generation.", flush=True)
    print("[setup] Installed verified multi-architecture rasterizer; no local compiler needed.")
