#!/usr/bin/env bash
#SBATCH --job-name=seceval_judge_agree
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/export/home/aberriche/BenchBench/sayf-eval/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/sayf-eval/slurm/logs/%x_%j.err
set -uo pipefail

# Tier-4: judge-agreement parity between the original scoring pipeline and the
# new scorer, same Azure gpt-5.4 judge, on identical stored model responses.

ROOT=/export/home/aberriche/BenchBench/sayf-eval
ORIG=/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks
ENV_BIN=/export/home/aberriche/miniconda3/envs/vllm/bin
export PATH="$ENV_BIN:$PATH"
PY="$ENV_BIN/python"
mkdir -p "$ROOT/slurm/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

set -a; source "$ORIG/.env"; set +a
[ -n "${AZURE_API_KEY:-}" ] || { echo "AZURE_API_KEY missing"; exit 1; }

"$PY" -m pip install -q litellm 2>/dev/null || true   # ensure transport present

RESP_DIR="${RESP_DIR:-$ORIG/outputs/responses_Llama-3.3-70B-Instruct}"
echo "Job ${SLURM_JOB_ID:-local}  Node ${SLURMD_NODENAME:-$(hostname)}  $(date)"
echo "Responses: $RESP_DIR  per-task: ${PER_TASK:-15}"

"$PY" tests/judge_agreement.py --response-dir "$RESP_DIR" --per-task "${PER_TASK:-15}"
echo "Done: $(date)"
