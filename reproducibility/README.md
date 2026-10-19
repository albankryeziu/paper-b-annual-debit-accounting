# Reproducibility and preservation

1. Inspect with `python RUN_PAPERB.py`; this is standard-library-only and read-only.
2. Download and hash-verify all release assets using `scripts/download_assets.py` and `data/ASSETS.json`.
3. Retain complete archive layouts. The current frozen Dutch snapshot includes the subsequent premium family; an older existing ZIP did not, so this release is packaged from the current authoritative archive instead of substituting that old ZIP.
4. `SOURCE_COPY_MANIFEST.csv` identifies the origin label and exact hash of every byte-preserved Git file. The full asset inventories also bind every source file to its archive member. Sources are copied, never rewritten in place.
5. Reconstruct the original Dutch runnable layout using the archive's `FILE_CLASSIFICATION.csv` original-to-canonical mapping and run instructions. Review local settings and upstream source-path records. Restore upstream data at explicitly configured locations; do not guess defaults or change scientific parameters.
6. Original engineering v2 and annual certificate scripts retain absolute local paths as provenance. Their verified reports/outputs are inspectable immediately; fresh numerical execution requires explicit path configuration in a new workspace. Do not present the preserved scripts as an already portable, one-command numerical pipeline.
7. Install the recorded scientific environment and obtain any required Gurobi license separately. The dependency source/range requirements are preserved next to the original code. Reporting PDFs use an additional ReportLab/pypdf/Poppler environment documented in the engineering guide.
8. Keep original Gate 1 certificates, archived trajectories, frozen card, historical AC and the earlier blocked preflight unchanged. The joint participant P/Q map is an explicitly declared engineering-v2 assumption, not a retroactive repair of historical results.

Original package roots: the frozen Dutch archive; final signoff dated 2026-10-17 (annual certificates and locked manuscripts); independent audit dated 2026-10-17; blocked engineering preflight dated 2026-10-18; completed engineering validation v2 dated 2026-10-18. This GitHub distribution is dated 2026-10-19.

No CI job imports the scientific solver or executes original orchestration. Integrity/syntax success is not certification of a new optimization, AC/security or prospective-control result.
