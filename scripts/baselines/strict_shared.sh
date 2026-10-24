#!/bin/bash
# 严格版共享队列：4 worker 抢单（flock 原子认领），无 strict 标记即跑，3 pass。
# 解决 mod 分区下快失败 worker 早退、慢任务堵 pass 的问题。日志接 baselines_runner.log。
set -u
cd "$(dirname "$0")"
PY=/data/miniconda3/envs/ys310/bin/python
LOG=../../results/baselines_runner.log
LOCKD=../../results/strict_locks
MIN_FREE_MB=2000
free_mb() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
mkdir -p "$LOCKD"

build_tasks() {
  : > ../../results/strict_tasks.txt
  for ds in gc mc mf; do for s in 42 43 44; do
    for m in deeptriage triagerx cbtmf sclbt; do
      grep -q '"strict": true' "../../results/baselines/$ds/seed$s/$m.json" 2>/dev/null || echo "$m|$ds|$s" >> ../../results/strict_tasks.txt
    done
  done; done
}

worker() { # $1=gpu $2=wid
  export CUDA_VISIBLE_DEVICES="$1"
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  while IFS='|' read -r m ds s; do
    [ -z "$m" ] && continue
    grep -q '"strict": true' "../../results/baselines/$ds/seed$s/$m.json" 2>/dev/null && continue
    mkdir "$LOCKD/${m}_${ds}_${s}" 2>/dev/null || continue
    local w=0
    while [ "$(free_mb "$1")" -lt "$MIN_FREE_MB" ]; do sleep 30; w=$((w+1)); [ "$w" -ge 480 ] && { rmdir "$LOCKD/${m}_${ds}_${s}"; return 1; }; done
    echo "[$(date '+%m-%d %H:%M:%S')] [q$2] START run_${m}_strict $ds s$s" >> "$LOG"
    set +e; $PY "run_${m}_strict.py" "$ds" "$s" < /dev/null >> "$LOG" 2>&1; ec=$?; set -e
    echo "[$(date '+%m-%d %H:%M:%S')] [q$2] END run_${m}_strict $ds s$s exit=$ec" >> "$LOG"
    rmdir "$LOCKD/${m}_${ds}_${s}" 2>/dev/null
    sleep 5
  done < ../../results/strict_tasks.txt
}

echo "[$(date '+%m-%d %H:%M:%S')] SHARED QUEUE start" | tee -a "$LOG"
for attempt in 1 2 3; do
  build_tasks
  left=$(wc -l < ../../results/strict_tasks.txt)
  echo "[$(date '+%m-%d %H:%M:%S')] pass $attempt: $left 格待跑" | tee -a "$LOG"
  [ "$left" -eq 0 ] && break
  worker 0 0 & worker 1 1 & worker 0 2 & worker 1 3 &
  wait
  $PY ../../monitor/mailer.py "Hy-LIN 严格版 pass$attempt 完" "剩 $(wc -l < ../../results/strict_tasks.txt) 格（重建前）" >> "$LOG" 2>&1
done
$PY make_table.py >> "$LOG" 2>&1
$PY ../../monitor/mailer.py "Hy-LIN RQ1：最终大表就绪" "tables/rq1_big.csv（严格口径）已就绪。" >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] SHARED QUEUE ALL DONE" | tee -a "$LOG"
