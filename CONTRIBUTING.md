# Contributing

Keep frozen scientific snapshots and result files immutable. Propose changes on a branch with a clear scientific rationale. Separate model changes from file organization or presentation changes. Never overwrite checkpoints, refit rates from held-out results, silently repair actions, or claim a physical/security result unsupported by its data.

Run `python RUN_PAPERB.py` and `python -m pytest tests -q` before submitting changes. Regenerate the checksum inventory only after reviewing intentional edits. Preserve `reproducibility/SOURCE_MAP.json`: it records the source origin and retained-definition structure, rather than certifying new numerical runs. Heavy solver and AC runs are not part of CI. Report the exact configuration, environment, source identities and numerical checks for any separately authorized new experiment.
