"""Authenticated release download and SHA verification; no extraction or solves."""

from pathlib import Path
import subprocess, json
import argparse

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.verify_repository import digest


def main():
    manifest = json.loads((ROOT / "data/ASSETS.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument(
        "--asset", action="append", help="Exact asset name; repeat to select several"
    )
    choice.add_argument(
        "--all", action="store_true", help="Download every historical evidence bundle"
    )
    args = parser.parse_args()
    known = {item["name"] for item in manifest["assets"]}
    if args.asset and not set(args.asset) <= known:
        parser.error("Unknown asset: " + ", ".join(sorted(set(args.asset) - known)))
    if not args.asset and not args.all:
        for item in manifest["assets"]:
            print(item["name"], "-", item["bytes"], "bytes")
        print("Select with --asset NAME. No files downloaded.")
        return
    selected = [item for item in manifest["assets"] if args.all or item["name"] in args.asset]
    target = ROOT / "downloads"
    target.mkdir(exist_ok=True)
    for item in selected:
        dest = target / item["name"]
        if not dest.exists():
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    manifest["tag"],
                    "--repo",
                    manifest["repository"],
                    "--pattern",
                    item["name"],
                    "--dir",
                    str(target),
                ],
                check=True,
            )
        if digest(dest) != item["sha256"]:
            raise RuntimeError("Downloaded hash mismatch: " + item["name"])
        print("VERIFIED", item["name"])


if __name__ == "__main__":
    main()
