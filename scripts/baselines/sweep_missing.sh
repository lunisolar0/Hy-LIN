#!/bin/bash
# 补跑 results/missing.txt（device bug 修后）：4 GPU worker，单卡 2 并发+显存门控
set -u
cd "$(dirname "$0")"
PY=/data/miniconda3/envs/ys310/bin/python
LOG=../../results/baselines_runner.log
MIN_FREE_MB=2000
free_mb() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
worker() {
  export CUDA_VISIBLE_DEVICES="$1"
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  local n=0
  while IFS='|' read -r m ds s; do
    n=$((n+1)); [ $((n % 4)) -ne "$2" ] && continue
    [ -f "../../results/baselines/$ds/seed$s/$m.json" ] && { echo "[sweep$2 SKIP] $m $ds s$s" >> "$LOG"; continue; }
    local w=0
    while [ "$(free_mb "$1")" -lt "$MIN_FREE_MB" ]; do sleep 30; w=$((w+1)); [ "$w" -ge 480 ] && return 1; done
    echo "[$(date '+%m-%d %H:%M:%S')] [sweep$2] START $m $ds s$s" >> "$LOG"
    set +e; $PY "run_$m.py" "$ds" "$s" < /dev/null >> "$LOG" 2>&1; ec=$?; set -e
    echo "[$(date '+%m-%d %H:%M:%S')] [sweep$2] END $m $ds s$s exit=$ec" >> "$LOG"
    sleep 5
  done < ../../results/missing.txt
}
for attempt in 1 2 3; do
  echo "[$(date '+%m-%d %H:%M:%S')] SWEEP pass $attempt" >> "$LOG"
  worker 0 0 & worker 1 1 & worker 0 2 & worker 1 3 &
  wait
done
echo "[$(date '+%m-%d %H:%M:%S')] SWEEP DONE" | tee -a "$LOG"
