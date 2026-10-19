# Paper B: annual restriction accounting

**Annual Restriction Accounting for Non-Firm Transmission Access: Entitlement Geometry and Network Effects**

Alban Kryeziu, Stuart X. Zhu, Tolga Karabas, and Evrim Ursavas. Research software and reproducibility evidence for a manuscript targeting IEEE Transactions on Power Systems. This repository does not imply journal acceptance or operational deployment.

## Scientific status

The frozen Dutch study compares touched duration (M1), its LP relaxation, proportional depth (M2), and conservative marginal-relief premiums. The engineering assessment is **CONDITIONAL GO for the stated controlled-model claims**. The annual numerical certificates remain certificates for the original soft-thermal DC accounting model, with their documented numerical qualifications.

The version-2 AC experiment uses the final active participant connection for both participant P and Q as an explicitly declared constant-power-factor assumption, rather than a reconstruction of undocumented device locations. The original two-map AC model is preserved separately. There are 648 common-map and 648 historical-map campaign cases, plus 27 historical baseline reconstruction cases. All campaign cases converge, but only **294/648 common-map cases pass thermal limits**. Generator-specific balancing, reserve/ramp feasibility, authentic forecast vintages, prospective implementation and annual AC security are not established. The selected AC advantage of C over U is mixed; ordinary M2 remains stronger in their paired checks. Do not interpret retrospective EC witnesses as rolling controllers or equal-exposure efficiency claims without their explicit controls.

## Quick start: inspect without solving

Python 3.9 or later is sufficient for repository verification; no solver, license or data download is required:

```bash
git clone https://github.com/albankryeziu/paper-b-annual-debit-accounting.git
cd paper-b-annual-debit-accounting
python RUN_PAPERB.py
```

GitHub authentication is required while this repository is private. `RUN_PAPERB.py` checks the committed file identities and Python syntax and prints the evidence status. It never runs an optimization or power-flow experiment. GitHub Actions runs the same lightweight checks.

## Repository map

| Directory | Contents |
| --- | --- |
| `code/frozen_dutch/` | Byte-preserved computational source; original local-path assumptions remain documented |
| `code/engineering_validation_v2/` | Byte-preserved isolated engineering evaluators and audits |
| `paper/locked_main/` | Unmodified Gate 3 manuscript, PDF and bibliography |
| `paper/locked_technical_report/` | Unmodified locked companion and supporting files |
| `paper/proposed_engineering_amendments/` | Proposed exact edits, not an already integrated or recompiled submission |
| `reports/` | Annual certificates, engineering dossier and scientific audit PDFs |
| `results/` | Compact, traceable annual and engineering result tables |
| `data/` | Versioned release-asset registry and data access instructions |
| `scripts/` | Lightweight integrity checking and authenticated asset download |
| `reproducibility/` | Source-copy manifest, environment and restoration instructions |

Full numerical arrays, per-epoch checkpoints and complete raw engineering rows are distributed as private GitHub **release assets**, not omitted and not silently replaced by small samples. `data/ASSETS.json` lists their exact sizes and SHA-256 values. See [data access](data/README.md) and [reproduction](reproducibility/README.md).

## Environments and scientific execution

The validated scientific environment uses Python 3.9.7, NumPy 1.26.4, SciPy 1.11.4, pandas 2.2.2, pandapower 3.2.1, pyarrow 15.0.2 and Gurobi 12.0.3. Historical optimization requires an independently valid Gurobi license; no license file or credential is distributed. The root `requirements.txt` records observed scientific versions, but the repository verification uses only the Python standard library.

The preserved scientific scripts contain machine-specific paths and references to frozen data outside their original directories. **Do not assume that running the old `RUN.py` after cloning reproduces the paper.** Follow the restoration/data-availability guide first. Configure a separate writable reproduction workspace. Keep frozen evidence read-only; do not rerun, refit or overwrite it merely to inspect this repository. No solver rerun was used to prepare this GitHub distribution.

## Citation, licenses and publication

Use [CITATION.cff](CITATION.cff) for authors/title and this repository URL. No unverified paper or report DOI is assigned. Locked manuscript sources remain distinct from the proposed post-engineering changes; their integrated page count has not been validated.

A new blanket software or data license has **not** been selected on behalf of all authors. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) and preserve existing third-party notices. Private storage does not itself approve public redistribution of upstream data, manuscripts or third-party code. Public release and the final code/data license remain author decisions.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md) and [release notes](RELEASE_NOTES.md).
