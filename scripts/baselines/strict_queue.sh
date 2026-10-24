#!/bin/bash
# 严格版：等 Hy-LIN 9/9 齐后跑 36 格（4 方法x3数据集x3种子），覆盖旧数据。最终口径即严格版，无版本之分。
# 代价追加记入中央表（json 内 cost 双份保留）。日志接 baselines_runner.log。
set -u
cd "$(dirname "$0")"
PY=/data/miniconda3/envs/ys310/bin/python
LOG=../../results/baselines_runner.log
MIN_FREE_MB=2000
free_mb() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }

echo "[$(date '+%m-%d %H:%M:%S')] STRICT排队：等 hylin 9/9" | tee -a "$LOG"
while [ "$(find ../../results/baselines -path '*seed*/hylin.json' 2>/dev/null | wc -l)" -lt 9 ]; do sleep 60; done
$PY ../../monitor/mailer.py "Hy-LIN RQ1：hylin齐，严格版开跑" "开始36格严格版（4方法），覆盖旧数据。最终口径即严格版。" >> "$LOG" 2>&1

QT=$(mktemp)
for ds in gc mc mf; do for s in 42 43 44; do
  for m in run_deeptriage_strict run_triagerx_strict run_cbtmf_strict run_sclbt_strict; do echo "$m|$ds|$s" >> "$QT"; done
done; done

worker() {
  export CUDA_VISIBLE_DEVICES="$1"
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  local n=0 short
  while IFS='|' read -r m ds s; do
    n=$((n+1)); [ $((n % 4)) -ne "$2" ] && continue
    short="${m#run_}"; short="${short%_strict}"
    if grep -q '"strict": true' "../../results/baselines/$ds/seed$s/$short.json" 2>/dev/null; then
      echo "[strict$2 SKIP] $m $ds s$s" >> "$LOG"; continue
    fi
    local w=0
    while [ "$(free_mb "$1")" -lt "$MIN_FREE_MB" ]; do sleep 30; w=$((w+1)); [ "$w" -ge 480 ] && return 1; done
    echo "[$(date '+%m-%d %H:%M:%S')] [strict$2] START $m $ds s$s" >> "$LOG"
    set +e; $PY "$m.py" "$ds" "$s" < /dev/null >> "$LOG" 2>&1; ec=$?; set -e
    echo "[$(date '+%m-%d %H:%M:%S')] [strict$2] END $m $ds s$s exit=$ec" >> "$LOG"
    cnt=0
    for d in gc mc mf; do for x in 42 43 44; do
      grep -q '"strict": true' "../../results/baselines/$d/seed$x/$short.json" 2>/dev/null && cnt=$((cnt+1))
    done; done
    if [ "$cnt" -eq 9 ]; then $PY ../../monitor/mailer.py "Hy-LIN RQ1严格版：$short 9/9" "$short 严格版三数据集x三种子完成。" >> "$LOG" 2>&1; fi
    sleep 5
  done < "$QT"
}
for attempt in 1 2 3; do
  echo "[$(date '+%m-%d %H:%M:%S')] STRICT pass $attempt" >> "$LOG"
  worker 0 0 & worker 1 1 & worker 1 2 & worker 0 3 &
  wait
done
$PY make_table.py >> "$LOG" 2>&1
$PY ../../monitor/mailer.py "Hy-LIN RQ1严格版：36格完成" "最终大表已出 tables/rq1_big.csv（严格口径）。" >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] STRICT ALL DONE" | tee -a "$LOG"
