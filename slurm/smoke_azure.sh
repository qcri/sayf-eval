#!/usr/bin/env bash
#SBATCH --job-name=seceval_azure_smoke
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/export/home/aberriche/BenchBench/sayf-eval/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/sayf-eval/slurm/logs/%x_%j.err
set -uo pipefail

# Live validation of the LiteLLM transport against the project's Azure OpenAI
# deployment (gpt-5.4), CPU-only. Validates Model adapter -> litellm -> Azure,
# judge-as-Model, pipeline, real dataset load, and corpus metrics. Also runs the
# real pytest suite (deps install fine on compute nodes, unlike the login node).

ROOT=/export/home/aberriche/BenchBench/sayf-eval
ORIG=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
ENV_BIN=/export/home/aberriche/miniconda3/envs/vllm/bin
export PATH="$ENV_BIN:$PATH"
PY="$ENV_BIN/python"
mkdir -p "$ROOT/slurm/logs"

echo "============================================================"
echo "Job ${SLURM_JOB_ID:-local}  Node ${SLURMD_NODENAME:-$(hostname)}  $(date)"
echo "============================================================"

# Azure key from the original repo's .env
set -a; source "$ORIG/.env"; set +a
[ -n "${AZURE_API_KEY:-}" ] || { echo "AZURE_API_KEY missing"; exit 1; }

echo ">> installing litellm + pytest"
"$PY" -m pip install -q litellm pytest || { echo "pip install FAILED"; exit 1; }
"$PY" -c "import litellm, pytest, cvss, datasets; from importlib.metadata import version; print('deps ok: litellm', version('litellm'))"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

echo ">> pytest"
"$PY" -m pytest tests/ -q || echo "WARN: pytest reported failures"

echo ">> live Azure smoke"
# Override with: sbatch --export=ALL,TASKS="mcq rcm ...",SAMPLES=2 slurm/smoke_azure.sh
TASKS="${TASKS:-mcq seceval}"
SAMPLES="${SAMPLES:-3}"
"$PY" tests/live_smoke_azure.py --samples "$SAMPLES" --tasks $TASKS

echo "Done: $(date)"
echo "============================================================"
