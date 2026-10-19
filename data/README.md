# Numerical data and preserved evidence

The files listed in `ASSETS.json` are full immutable snapshot archives, not reduced samples. Ordinary Git contains code, manuscripts, reports and compact tables; bulk evidence is stored on the private `v0.1.0-research-snapshot` GitHub release. Download requires repository access.

```bash
gh auth login
python scripts/download_assets.py
python scripts/download_assets.py --asset paperb-frozen-dutch-current.zip
```

With no arguments the tool lists the available files and downloads nothing. Select exact names with `--asset` (repeat for several); `--all` explicitly downloads all evidence. Code inspection and unit tests require no archive downloads.

| Need | Bundle |
| --- | --- |
| Executed DC source, physical inputs and final policies | `paperb-frozen-dutch-current.zip` |
| Registered original network and upstream inputs | `paperb-registered-upstream-sources.zip` |
| Completed engineering inputs/outputs | `paperb-engineering-v2.zip` |
| Full numerical lower-bound vectors and arrays | `paperb-annual-certificate-2029.zip`, `2030.zip`, `2031.zip` |

The remaining audit/signoff/preflight bundles preserve provenance. They are not alternate implementations to choose between. The main branch contains the single readable final source view.

The tool verifies each SHA-256 and never extracts or runs a model. Restore into a separate workspace, retaining original roots. See [reproduction](../docs/REPRODUCIBILITY.md). Asset hashes determine provenance.
