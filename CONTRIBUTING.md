# Contributing

Keep frozen scientific snapshots and result files immutable. Propose changes on a branch with a clear scientific rationale. Separate model changes from file organization or presentation changes. Never overwrite checkpoints, refit rates from held-out results, silently repair actions, or claim a physical/security result unsupported by its data.

Run `python RUN_PAPERB.py` before submitting changes. Regenerate the repository checksum inventory only after reviewing intentional edits; preserve the upstream-copy manifest. Heavy solver and AC runs are not part of CI. Report the exact configuration, environment, source identities, scope and numerical checks when a new experiment is separately authorized.
