#!/usr/bin/env bash
#SBATCH --job-name=seceval_parity2
#SBATCH --partition=cpu-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/export/home/aberriche/BenchBench/seceval-harness/slurm/logs/%x_%j.out
#SBATCH --error=/export/home/aberriche/BenchBench/seceval-harness/slurm/logs/%x_%j.err
set -uo pipefail

# Authoritative Tier-1 parity: new loaders vs the CURRENT original collectors
# (generation stubbed). Deterministic, no API; needs datasets + HF/GitHub network.

ROOT=/export/home/aberriche/BenchBench/seceval-harness
ENV_BIN=/export/home/aberriche/miniconda3/envs/vllm/bin
export PATH="$ENV_BIN:$PATH"
PY="$ENV_BIN/python"
mkdir -p "$ROOT/slurm/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "Job ${SLURM_JOB_ID:-local}  Node ${SLURMD_NODENAME:-$(hostname)}  $(date)"
"$PY" tests/parity_vs_current.py --samples "${SAMPLES:-5}"
echo "Done: $(date)"
