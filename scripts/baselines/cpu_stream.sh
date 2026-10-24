#!/bin/bash
# RQ1 pure-CPU stream: TF-IDF+SVM and LightGCN. No GPU, runs alongside the GPU streams.
# Restart-safe: skips any cell whose results/baselines/<ds>/seed<s>/<method>.json exists.
# Threads capped at 2: higher thread counts + concurrency triggers SIGBUS (exit 135) on this host.
# Usage: bash scripts/baselines/cpu_stream.sh <parity> [mod]
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
PY=/data/miniconda3/envs/ys310/bin/python
LOG="$ROOT/results/baselines_runner.log"
PAR=${1:-0}
MOD=${2:-2}
export CUDA_VISIBLE_DEVICES=""
export HXL_DEVICE=R
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$ROOT/results/baselines"
n=0
for ds in mf mc gc; do for s in 42 43 44; do
  for m in tfidf_svm lightgcn; do
    n=$((n+1)); [ $((n % MOD)) -ne "$PAR" ] && continue
    dst="$ROOT/results/baselines/$ds/seed$s/$m.json"
    [ -f "$dst" ] && continue
    echo "[$(date '+%m-%d %H:%M:%S')] [cpu$PAR] START $m $ds s$s" >> "$LOG"
    set +e
    "$PY" "run_$m.py" "$ds" "$s" < /dev/null >> "$LOG" 2>&1
    ec=$?
    set -e
    echo "[$(date '+%m-%d %H:%M:%S')] [cpu$PAR] END $m $ds s$s exit=$ec" >> "$LOG"
  done
done; done
echo "[$(date '+%m-%d %H:%M:%S')] [cpu$PAR] stream done"
