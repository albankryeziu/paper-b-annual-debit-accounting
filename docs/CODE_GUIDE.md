# Final implementation guide

## Organization and reading order

`src/tpwrs/` is the shared mathematical implementation. `src/jobs/` contains the final experiment drivers and their direct dependencies. `validation/` is the completed engineering evaluator. `certificates/` contains the independent annual lower-bound reconstruction and verification. These are different responsibilities, not successive code versions.

Read `contracts.py`, `model.py`, `replay.py`, `annual.py`, then `rolling.py`. Read `debit_rules.py` before the premium driver. The implementation keeps the original `tpwrs` and `jobs` package names so its internal imports and mathematical references remain recognizable.

## Physical and accounting model

`PhysicalCase` in `contracts.py` binds the time grid, participant ordering, capacities, requests, masks, background injections, transaction matrix and network states. Eligible action bounds use the fixed masks and the minimum of request and capacity. The participant transaction convention is evaluated as `p_background - A @ (request - x)`; network flows include the affine PTDF offset. `replay.physical_objectives` independently evaluates the complete monitored-row overload residual and curtailed energy.

`model._build` defines the explicit physical-row formulation. M1 has activation variables and both materiality links; M1-LP has request/action-bound-normalized debit; M2 has capacity-normalized debit. The weighted interface is called `M3` inside the original numerical contract. The conservative premium uses that interface with its own frozen positive rates. This internal name does not make the relief score the older budget-dual method.

`annual.py` implements the continuous annual engine: screening, coordinate compression, hourly epigraph support generation, staged objectives and replay acceptance. `linear.py` builds sparse solver models. `execution.py` contains solver profiles, identity checks, resource controls and numerical acceptance. M1's mixed-integer engine and the continuous annual support engine remain distinct.

`rolling.solve_rolling` receives the current balance, solves the look-ahead window, commits only the implemented block and subtracts only that block's debit. Primary campaigns use 48 hours of look-ahead and 24 hours of commitment. Tests check carry-over, window coverage and annual resets.

## Main duration/depth experiment

`jobs.primary` constructs annual M1-LP/M2 and rolling M1/M2 experiments for 2029–2031 at rho=0.15. `_policy_metrics` and `_rolling_metrics` distinguish debit, FDE and physical energy. `jobs.certificates` forms conservative saved-action witnesses; `jobs.primary_freeze` records evidence identities. `jobs.common` supplies shared campaign helpers.

The campaign retains its qualified solver settings, replay tolerances and identity gates. Earlier source reconstruction, qualification orchestration, alternate M3 campaigns, gamma/bound runners and report-export runners are absent from this source view. Their historical inputs and outputs remain in the preserved release.

## Conservative premium experiment

`debit_rules.hinge_relief_scores` computes the requested-state right derivative using the complete row evaluator, including boundary directional hinge terms. It uses no optimized action to construct the score. This remains a retrospective deterministic-input experiment; exogenous scores do not establish authentic forecast availability.

`debit_rules.premium_card` uses the three-rate convex QP helper in `cards.py`. That module now contains only the QP, bucket assignment, rate lookup, matched-decomposition and held-out-audit helpers used by final dependencies. Unused teacher construction, candidate selection, gamma and ordering-control functions are removed. `features.py` retains only `PredictorBundle`, required when loading saved models for the original flat-card diagnostic. These are compatibility dependencies, not another adopted premium policy.

`jobs.debit_rule_family.prepare` audits the original card and constructs the fitting-designed candidate. `run_campaign` checks the frozen identity, evaluates 2028, applies its predeclared validation gate, then evaluates the frozen policies on 2029–2031. Constants remain gamma=0.5, xi=0.25 and penalty=1. The uniform control uses the reference premium. `results/frozen_premium_card.json` and `premium_validation_gate.json` are exact saved records.

## Exposure controls

`jobs.m1_exposure_matched` implements the retrospective duration-matched rolling M2 comparison. It measures energy and FDE separately, checks source identities and uses exact rational sums of saved binary64 coordinates for ceilings. `jobs.exposure_diagnostic` constructs the post-solve premium witness capped by each participant's uniform-reference energy. Its source originally lived beside the premium results; it is now visible with the other experiments.

These comparisons answer different questions. Neither a repaired annual witness nor hindsight budgets define a prospective chronological controller. The cleanup does not change that interpretation.

## Independent annual certificates

`certificates/prepare_gate1.py` reconstructs the archived restricted master, `solve_gate1.py` retains independent primal/dual vectors, and `verify_certificate.py` and `verify_endpoint_identity.py` check lower bounds and upper endpoints. `gate1_common.py` supplies independent arithmetic and physical-input reconstruction.

These scripts retain the original signoff-folder execution context. They are included once as the final verification source and are not imported by the default entry point. Do not run annual solves merely to inspect them. Saved certificates remain in `results/annual_certificates/` and the research release.

## Final engineering evaluation

`validation/engineering_common.py` contains physical input loading, transaction and participant P/Q reconstruction, power-factor scenarios, unchanged AC solver settings and protected-output writing. Imports from it are explicit in the other modules.

| Module | Responsibility |
| --- | --- |
| `stage_preflight.py` | Historical baseline reproduction and input/mapping checks |
| `stage_physical.py` | Full-year DC physical-overload metrics |
| `ac_evaluation.py` | Individual AC case construction and metrics |
| `stage_ac.py` | Identical selected-case replay under both P/Q mappings |
| `stage_headroom.py` | Conditional generator/headroom audit |
| `stage_information.py` | Frozen parameters and information timing |
| `stage_qc.py`, `verify_common_mapping.py` | Independent numerical and mapping checks |

The common-map assumption is C_Q=C_P, with historical C_Q retained separately. Completed numerical reports are retained; report-formatting, amendment-generation and archive-sealing scripts are absent from the scientific source view.

## What the cleanup verifies

`reproducibility/SOURCE_MAP.json` links retained sources to original paths and byte hashes. `scripts/verify_refactor.py` compares parsed structures, excluding module documentation and the documented expansion of engineering wildcard imports. For trimmed `cards.py` and `features.py`, it compares exactly the listed retained definitions/constants and separately verifies imports and the absence of other executable statements. Retained function bodies, coefficients, comparison signs, objective order and numerical settings are unchanged. Code is formatted at 100 columns; obsolete concatenation banners are removed.

This does not assert that relocation leaves filesystem-dependent behavior unchanged. Drivers using `__file__`, absolute paths or historical release manifests need their recorded context. No source-hash guard has been relaxed and no certificate is reassigned to reformatted source. See [reproduction](REPRODUCIBILITY.md).
