# Complete data and evidence assets

The files listed in `ASSETS.json` are full immutable snapshot archives, not reduced samples. Ordinary Git contains code, manuscripts, reports and compact tables; bulk evidence is stored on the private `v0.1.0-research-snapshot` GitHub release. Download requires repository access.

```bash
gh auth login
python scripts/download_assets.py
```

The download tool checks existing files and verifies each SHA-256 after download. It never runs a model or extracts over an existing experiment. Unpack assets into a new workspace, retaining their top-level root directories. `reproducibility/README.md` explains the distinctions between the original release, subsequent premium results and later audits. Asset identities, not filenames alone, determine provenance.
