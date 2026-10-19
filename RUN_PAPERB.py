"""Safe no-argument entry point: verify committed snapshots, never solve."""

from scripts.verify_repository import main

if __name__ == "__main__":
    raise SystemExit(main())
