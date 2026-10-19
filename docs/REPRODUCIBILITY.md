# Inspection, tests and scientific reproduction

## Inspect and test

`python RUN_PAPERB.py` checks file identities, Python syntax and the recorded source-tree comparison using only the standard library. `python -m pip install -e ".[test]"` makes `tpwrs` and `jobs` importable from `src/`; `python -m pytest tests -q` runs focused mathematical and ledger tests. They do not run an annual optimization or AC campaign.

## Reading edition and executed source identity

This compact source derives from the exact implementation that produced the results. The cleanup changes layout, formatting, comments, module documentation and explicit engineering imports. The source view excludes earlier orchestration and report/audit machinery outside the final source closure.

Formatting changes byte hashes. Moving a file changes `__file__`-derived paths. Drivers retain historical path assumptions and strict source/profile qualification checks. Installing the readable package does not create a newly qualified numerical pipeline:

- The 2028 qualification and annual-solver fingerprints remain tied to original source bytes.
- `jobs.debit_rule_family` still requires exact archive and solver identity matches.
- Engineering scripts expect archived physical/network records and separate signoff sources.
- Independent certificate scripts expect the original signoff directory layout.

No gate was weakened to accept a reformatted hash. Fresh execution of the readable edition needs a separate validated runtime adaptation and numerical qualification. That work and heavy reruns are outside this source cleanup.

## Restore the executed implementation

For exact historical reproduction, use the hash-bound [research snapshot](https://github.com/albankryeziu/paper-b-annual-debit-accounting/releases/tag/v0.1.0-research-snapshot), rather than mixing new source with old qualification records. It preserves complete executed source/input layouts. Follow its file classification, run instructions, source registry and qualification requirements. Resolve upstream inputs explicitly in a separate workspace.

The recorded environment was Python 3.9.7, NumPy 1.26.4, SciPy 1.11.4, pandas 2.2.2, scikit-learn 1.3.2, pyarrow 15.0.2, pandapower 3.2.1 and Gurobi 12.0.3. `requirements.txt` records observed versions. Optimization requires a separate valid Gurobi license. Minimal package dependencies for tests do not replace that environment.

## Preservation

The original local scientific archive, manuscript, annual certificates, 2028 selection, 2029–2031 policies and completed engineering outputs are unchanged. The `v0.1.0-research-snapshot` tag and release are retained. Edits are confined to the GitHub preparation checkout. Archives remain available for provenance but are not the main README's starting point.
