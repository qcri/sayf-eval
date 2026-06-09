# Contributing to sayf-eval

Thanks for your interest in improving sayf-eval! This guide covers the dev setup,
the quality gate, and how to add a new benchmark task.

## Dev setup

```bash
git clone https://github.com/qcri/sayf-eval.git
cd sayf-eval
make install          # pip install -e ".[dev]"
pre-commit install    # optional: run ruff on every commit
```

## Quality gate

CI runs the same two checks; please run them before opening a PR:

```bash
make style     # auto-fix: ruff format + ruff check --fix
make quality   # CI gate: ruff format --check + ruff check
make test      # pytest (unit tests)
```

Style: ruff (line length 119, double quotes, isort with `sayf_eval` first-party).

## Tests

`make test` runs the unit suite (`tests/test_*.py`) — no network or API keys.
The live/parity scripts (`tests/live_smoke_azure.py`, `judge_agreement.py`,
`parity_*.py`) need provider credentials and a cluster; they are excluded from
CI and documented in [VALIDATION.md](VALIDATION.md).

## Adding a benchmark task

Tasks are **data**, not subclasses. To add one:

1. **Loader** (`src/sayf_eval/datasets.py`): write a function returning
   `list[Sample]` (or reuse `make_hf_loader` / `make_athena_loader`). Build the
   exact prompt the benchmark intends; normalize the gold answer into `target`.
2. **Judge rules** (`src/sayf_eval/judge_prompts.py`): if the task needs a new
   answer family, add a `(format_hint, compare_rule)` entry keyed by `task_type`.
   Reuse an existing `task_type` (`mcq`, `seceval`, `rcm`, `vsp`, `ate`, `rms`,
   `taa`, `ckt`) when possible.
3. **Metric** (`src/sayf_eval/metrics.py`): only if the task needs aggregation
   beyond accuracy (e.g. a new structured metric like VSP MAD / ATE micro-F1).
4. **Register** (`src/sayf_eval/tasks/__init__.py`): `register(Task(...))` with
   the task name, `task_type`, loader, optional system prompt, and token budget.
5. **Test**: add a loader-helper unit test and confirm
   `sayf-eval run --tasks <name> --max-samples 3 …` produces a valid summary.

## Invariants to preserve

These standardized choices are load-bearing — do not silently change them:

- All attempted items count toward the denominator; unparseable/empty answers
  are **incorrect**. Only judge-API failures are `skipped` (excluded from both
  numerator and denominator).
- Reasoning models: strip the `<think>` block before judging, then apply any
  stop sequence to the answer portion only.
- Per-task scoring rules (set match, CWE/MITRE IDs, CVSS, alias-aware names) live
  in the judge prompt, not in regex post-processing.

## License

By contributing, you agree your contributions are licensed under the
[MIT License](LICENSE).
