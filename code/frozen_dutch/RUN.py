"""Run this file in PyCharm. Choose the stage in LOCAL_SETTINGS.json (default 00)."""
import sys

# Keep installed/standard libraries ahead of the project directory. The manifest
# rejects obsolete root-level files before they can shadow imports such as json.
_ROOT = __file__.replace("\\", "/").rsplit("/", 1)[0].rstrip("/")
sys.path[:] = [p for p in sys.path if p and p.replace("\\", "/").rstrip("/").casefold() != _ROOT.casefold()] + [_ROOT]

from tpwrs.orchestrator import main

if __name__ == "__main__":
    raise SystemExit(main())
