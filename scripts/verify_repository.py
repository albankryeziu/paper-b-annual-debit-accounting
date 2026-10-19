"""Read-only checksum and syntax checks; no scientific imports or data downloads."""

from pathlib import Path
import ast, csv, hashlib, json
from scripts.verify_refactor import verify

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def main():
    errors = []
    with (ROOT / "reproducibility/REPOSITORY_SHA256.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        p = ROOT / row["path"]
        if not p.is_file() or digest(p) != row["sha256"]:
            errors.append(row["path"])
    for row in rows:
        p = ROOT / row["path"]
        if p.suffix == ".py" and p.is_file():
            ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
    if errors:
        raise SystemExit("Checksum mismatch: " + ", ".join(errors))
    assets = json.loads((ROOT / "data/ASSETS.json").read_text())
    print("PASS:", len(rows), "committed file identities; Python syntax valid.")
    print("PASS:", verify(), "mapped source structures retained.")
    print("Final code: src/tpwrs; campaigns: src/jobs; engineering: validation.")
    print("Preserved evidence downloads:", len(assets["assets"]))
    print("Scientific status: CONDITIONAL GO. No optimization or AC solve was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
