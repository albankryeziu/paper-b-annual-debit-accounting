"""Record intentional checkout changes after reviewing the Git diff.

This updates repository presentation identities, not any scientific archive or
numerical certificate. Git-ignored files and the inventory itself are excluded.
"""

import csv
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode("utf-8")
        .split("\0")
    )
    destination = ROOT / "reproducibility/REPOSITORY_SHA256.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        count = 0
        for name in sorted(set(names)):
            path = ROOT / name
            if not name or not path.is_file() or path == destination:
                continue
            payload = path.read_bytes()
            writer.writerow(
                {"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            )
            count += 1
    print("Recorded", count, "checkout file identities; no scientific files modified.")


if __name__ == "__main__":
    main()
