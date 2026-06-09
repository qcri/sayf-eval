#!/usr/bin/env bash
#SBATCH --job-name=sayf_release_check
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/export/home/aberriche/BenchBench/sayf-eval/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/sayf-eval/slurm/logs/%x_%j.err
set -uo pipefail

# Full release gate for sayf-eval v0.1.0 — install, lint, test, build, verify.
# Does NOT publish to PyPI (that needs a token); prints the upload command at the end.

ROOT=/export/home/aberriche/BenchBench/sayf-eval
ENV_BIN=/export/home/aberriche/miniconda3/envs/vllm/bin
export PATH="$ENV_BIN:$PATH"
PY="$ENV_BIN/python"
mkdir -p "$ROOT/slurm/logs"
cd "$ROOT"

rc=0
step() { echo; echo "==================== $* ===================="; }

step "1. editable install (.[dev])"
"$PY" -m pip install -q -e ".[dev]" || { echo "INSTALL FAILED"; exit 1; }
"$PY" -c "import sayf_eval; print('sayf_eval', sayf_eval.__version__)"

step "2. ruff format --check"
"$PY" -m ruff format --check src tests || rc=1

step "3. ruff check"
"$PY" -m ruff check src tests || rc=1

step "4. pytest"
"$PY" -m pytest || rc=1

step "5. build sdist + wheel"
rm -rf dist build ./*.egg-info src/*.egg-info
"$PY" -m build || { echo "BUILD FAILED"; rc=1; }
ls -la dist/ 2>/dev/null

step "6. twine check (PyPI metadata)"
"$PY" -m twine check dist/* || rc=1

step "7. clean-venv install from wheel + entry-point smoke"
rm -rf /tmp/sayf_relcheck
"$PY" -m venv /tmp/sayf_relcheck
/tmp/sayf_relcheck/bin/python -m pip install -q dist/*.whl && \
  /tmp/sayf_relcheck/bin/sayf-eval --help >/dev/null 2>&1 && \
  /tmp/sayf_relcheck/bin/python -c "import sayf_eval, sayf_eval.tasks; from sayf_eval.registry import available_tasks; print('installed wheel OK —', len(available_tasks()), 'tasks; CLI present')" \
  || { echo "WHEEL INSTALL/CLI CHECK FAILED"; rc=1; }

echo
echo "============================================================"
if [ "$rc" -eq 0 ]; then
  echo "RELEASE GATE: PASS — v0.1.0 is ready."
  echo "To publish (needs a PyPI token, run interactively):"
  echo "    python -m twine upload dist/*"
else
  echo "RELEASE GATE: FAILURES above (rc=$rc)"
fi
echo "============================================================"
exit "$rc"
