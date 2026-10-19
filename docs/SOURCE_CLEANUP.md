# Final source cleanup — 19 October 2026

The default repository now contains one readable edition of the final implementation. The original `v0.1.0-research-snapshot` commit, tag and release remain unchanged.

## Changes

- Consolidated the source presentation into `src/tpwrs` (shared mathematics), `src/jobs` (final campaigns), `validation` (final engineering checks) and `certificates` (independent annual bounds).
- Retained 39 scientific Python files across those locations, with no second frozen copy in the main branch. Earlier job runners and report/archive generators are outside this source view.
- Reduced the rate-card module to 316 formatted lines containing final dependencies. Removed unused candidate selection, gamma and ordering-control helpers. Kept the saved-predictor compatibility class required by the flat-card diagnostic.
- Promoted the post-solve exposure-witness implementation from the results directory into the experiment source tree.
- Expanded dense statements, standardized 100-column formatting, removed obsolete concatenation banners, added module descriptions and made engineering imports explicit.
- Added a manuscript-to-code guide, package metadata and directly browsable final result tables/card records. A normal inspection or test run downloads no historical bundles.

## Verification

- **42 retained source mappings pass structural checks against the original definitions.** For trimmed modules, only the documented retained functions/constants are selected; imports are checked separately. Other scientific module structures are retained in full, apart from module documentation and explicit engineering imports. Function bodies, numerical constants and arithmetic order are unchanged.
- **39 focused tests pass** on Python 3.9 with the recorded NumPy/SciPy/pandas versions. They cover debit laws, rate bounds/normalization, marginal relief, exact participant exposure ceilings, rolling balance carry-over/reset, identity/tamper rejection and saved-predictor dispatch.
- Undefined-name checks and formatting checks pass.
- **37 scientific artifacts are byte-identical** to their original sources, including the locked manuscript/companion, saved result tables, frozen premium card/validation decision and retained reports. See `reproducibility/ARTIFACT_SOURCES.csv`.
- No optimization, AC campaign, rate fitting or numerical result regeneration was run. The original local archive, scientific data and completed checkpoints were not changed.

## Limits of this cleanup

This is a readable source edition, not a newly qualified portable numerical pipeline. Relocation changes filesystem context and formatting changes source hashes. Historical runtime assumptions and qualification guards remain explicit and unchanged; they have not been bypassed to accept the cleaned source. Use the preserved snapshot for exact execution context, or qualify a separate runtime adaptation before fresh scientific runs.

The locked manuscript remains separate from the proposed engineering amendments. The cleanup does not integrate those edits, certify a new page count, or change any scientific conclusion.
