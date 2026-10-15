"""Combine build outputs into the extension's distributable wheel bundle."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rasterizer_distribution import source_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    destination = ROOT / "rasterizer_wheels"
    destination.mkdir(exist_ok=True)
    entries = []
    expected_source = source_digest(ROOT / "custom_rasterizer")
    for build_dir in args.build_dirs:
        for metadata in build_dir.glob("*.json"):
            entry = json.loads(metadata.read_text(encoding="utf-8"))
            if entry["source_sha256"] != expected_source:
                raise RuntimeError("Build sources do not match this extension.")
            wheel = build_dir / entry["filename"]
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError("Build checksum mismatch.")
            shutil.copyfile(wheel, destination / wheel.name)
            entries.append(entry)
    if not entries or len({(entry["python"], entry["profile"]) for entry in entries}) != len(entries):
        raise RuntimeError("Expected a nonempty bundle with no duplicate runtime profiles.")
    (destination / "index.json").write_text(json.dumps({"version": 1, "wheels": entries}, indent=2), encoding="utf-8")
    print(f"Bundled {len(entries)} verified rasterizer wheels in {destination}")


if __name__ == "__main__":
    main()
