# Paper B: annual restriction accounting

**Annual Restriction Accounting for Non-Firm Transmission Access: Entitlement Geometry and Network Effects**

Alban Kryeziu, Stuart X. Zhu, Tolga Karabas, and Evrim Ursavas.

This repository presents the final Dutch experiment implementation and manuscript. Start with the mathematical core in `src/tpwrs/`, then the final experiment drivers in `src/jobs/`. Each implementation is kept once. Earlier packages and audit inventories remain in the preserved research release, outside the main source view. See the [code guide](docs/CODE_GUIDE.md) for the methods and data flow.

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

## Read the final code

| Directory | Contents |
| --- | --- |
| [`src/tpwrs/`](src/tpwrs/) | Shared case contracts, DC model, annual support engine, rolling ledger and premium construction |
| [`src/jobs/`](src/jobs/) | Final duration/depth campaign, premium campaign and exposure comparisons |
| [`validation/`](validation/) | Final matched AC, physical-overload, headroom and information checks |
| [`certificates/`](certificates/) | Independent annual lower-bound reconstruction and verification |
| [`paper/main/`](paper/main/) | Final locked manuscript source and PDF |
| [`paper/technical_companion/`](paper/technical_companion/) | Its locked companion |
| `paper/proposed_engineering_amendments/` | Proposed exact edits, not an already integrated or recompiled submission |
| `reports/` | Final engineering dossier and annual certificate report |
| `results/` | Main duration/depth, premium, exposure and engineering tables; frozen premium card |
| `data/` | Versioned release-asset registry and data access instructions |
| `scripts/` | Lightweight integrity checking and authenticated asset download |
| `reproducibility/` | Source mapping and integrity identities |

Read `contracts.py`, `model.py`, `replay.py`, `annual.py`, then `rolling.py` in `src/tpwrs/`. The final campaign drivers are `primary.py`, `debit_rule_family.py`, `m1_exposure_matched.py` and `exposure_diagnostic.py` in `src/jobs/`. `debit_rules.py` implements the relief score and premium law; `cards.py` contains the shared three-rate QP and retained flat-card diagnostic helpers.

Full arrays and raw evidence remain in the [preserved research snapshot](https://github.com/albankryeziu/paper-b-annual-debit-accounting/releases/tag/v0.1.0-research-snapshot), with sizes and SHA-256 values in `data/ASSETS.json`. See [data access](data/README.md) and [reproduction](docs/REPRODUCIBILITY.md).

## Environments and scientific execution

The validated scientific environment uses Python 3.9.7, NumPy 1.26.4, SciPy 1.11.4, pandas 2.2.2, pandapower 3.2.1, pyarrow 15.0.2 and Gurobi 12.0.3. Historical optimization requires an independently valid Gurobi license; no license file or credential is distributed. The root `requirements.txt` records observed scientific versions, but the repository verification uses only the Python standard library.

For the focused mathematical and ledger tests:

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```

GitHub Actions runs integrity checks and these tests using small synthetic inputs, without repeating annual campaigns.

The compact source is a readable edition of the recorded implementation. **Relocated campaign drivers still need their original runtime layout and qualification records.** They are not a newly qualified one-command numerical rerun. Formatting changes source hashes; existing gates are retained rather than bypassed. See [reproduction](docs/REPRODUCIBILITY.md) before scientific execution. Original evidence and experiments remain unchanged.

## Citation, licenses and publication

Use [CITATION.cff](CITATION.cff) for authors/title and this repository URL. No unverified paper or report DOI is assigned. Locked manuscript sources remain distinct from the proposed post-engineering changes; their integrated page count has not been validated.

A new blanket software or data license has **not** been selected on behalf of all authors. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) and preserve existing third-party notices. Private storage does not itself approve public redistribution of upstream data, manuscripts or third-party code. Public release and the final code/data license remain author decisions.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md) and [release notes](RELEASE_NOTES.md).
