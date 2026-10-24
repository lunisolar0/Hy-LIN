#!/bin/bash
# 严格版共享 worker（独立进程，供 executor 启停）：抢 strict_tasks.txt，flock 认领，干完即退。
# 用法：strict_worker.sh <gpu> <wid>
set -u
# 必须先锚定绝对路径：原来 ROOT 用相对路径二次 dirname，只有以绝对路径调用时
# 才正确，改成先解析脚本目录再逐级上溯，任何调用方式都一致。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPU="$1"; WID="$2"
PY=/data/miniconda3/envs/ys310/bin/python
LOG=../../results/baselines_runner.log
LOCKD="$ROOT/results/strict_locks"
QFILE="$ROOT/results/quarantined.txt"
MIN_FREE_MB=2000
MIN_MEM_GB=8      # 可用内存低于此：暂不起新任务（防共享机被其他组吃干导致OOM）
avail_gb() { awk '/MemAvailable/{printf "%d", $2/1024/1024}' /proc/meminfo; }
# 暴露全部 2 张卡：triagerx 提取在单进程内把 batch 交替分到双卡（吃满双卡），$GPU 仅用于资源检查
export HXL_DEVICE=R
export CUDA_VISIBLE_DEVICES="$1"
free_mb() { nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$LOCKD"; touch "$QFILE"
# 每个worker错峰 + 任务shuffle：避免4 worker全挤在同一个方法上导致单卡爆满
sleep $((RANDOM % 15))
# 任务表可用 STRICT_TASKS 覆盖：两个 GPU 想各跑一份不相交的任务时用
TASKS="${STRICT_TASKS:-$ROOT/results/strict_tasks.txt}"
_SHUF="$ROOT/results/strict_tasks.$WID.tmp"
shuf "$TASKS" > "$_SHUF" 2>/dev/null || cp "$TASKS" "$_SHUF"
trap 'rm -f "$_SHUF"' EXIT
while IFS='|' read -r m ds s; do
  [ -z "$m" ] && continue
  grep -q '"strict": true' "../../results/baselines/$ds/seed$s/$m.json" 2>/dev/null && continue
  grep -qxF "$m|$ds|$s" "$QFILE" 2>/dev/null && continue
  _lk="$LOCKD/${m}_${ds}_${s}"
  if [ -d "$_lk" ]; then
    _op=$(cat "$_lk/pid" 2>/dev/null)
    if [ -z "$_op" ] || ! kill -0 "$_op" 2>/dev/null; then rm -rf "$_lk"; fi
  fi
  if mkdir "$_lk" 2>/dev/null; then echo "$$" > "$_lk/pid"; else continue; fi
  w=0
  while [ "$(free_mb)" -lt "$MIN_FREE_MB" ]; do sleep 30; w=$((w+1)); [ "$w" -ge 480 ] && break; done
  # 内存准入按数据集分级：原来统一 MIN_MEM_GB=8，只够 mf(约 7G 特征)。
  # gc 单格就要约 73G，用 8G 门控等于没门控——共享机上其他组一吃内存，
  # 它照样起跑再被 SIGBUS(135) 杀掉。这里按真实需求 + 余量设阈值。
  case "$ds" in
    gc) _need_mem=88 ;;
    mc) _need_mem=32 ;;
    *)  _need_mem=16 ;;
  esac
  mw=0
  while [ "$(avail_gb)" -lt "$_need_mem" ]; do
    sleep 30; mw=$((mw+1))
    echo "[$(date '+%m-%d %H:%M:%S')] [q$WID] WAIT mem $ds s$s need ${_need_mem}G avail $(avail_gb)G" >> "$LOG"
    [ "$mw" -ge 240 ] && break
  done
  # 重活全局互斥：gc/mc 的 triagerx 单格吃 70G+，多个 worker 各自"看到 118G 还够"
  # 就会同时起跑，合起来超过物理内存 → SIGBUS(135) / 被 OOM killer 杀(137)。
  # 实测三个 gc 并发时一个吃到 68G、另两个卡在 1G 等待，可用内存只剩 50G。
  # 用全局锁保证同一时刻只有一个重活格在跑（锁内再核一次内存余量）。
  _heavy_lk="$LOCKD/_heavy"
  case "$m" in
    triagerx|deeptriage)
      if [ -d "$_heavy_lk" ]; then
        _hp=$(cat "$_heavy_lk/pid" 2>/dev/null)
        if [ -z "$_hp" ] || ! kill -0 "$_hp" 2>/dev/null; then rm -rf "$_heavy_lk"; fi
      fi
      hw=0
      while ! mkdir "$_heavy_lk" 2>/dev/null; do
        sleep 30; hw=$((hw+1)); [ "$hw" -ge 240 ] && break
      done
      echo "$$" > "$_heavy_lk/pid"
      # 拿到锁后再等一次：前面那个重活可能刚释放，内存尚未回落
      wait2=0
      while [ "$(avail_gb)" -lt "$_need_mem" ]; do sleep 30; wait2=$((wait2+1)); [ "$wait2" -ge 160 ] && break; done
      ;;
  esac
  # 临跑前复查结果是否已被别的 worker 产出。原因：上面"等内存/等重活锁"可能阻塞
  # 十几分钟，期间另一 worker 可能已跑完同一格并写出 json；若不复查，本 worker
  # 醒来仍会重复整个训练（实测 gc/s42 被重算过）。这里复查即释放锁退出。
  if grep -q '"strict": true' "../../results/baselines/$ds/seed$s/$m.json" 2>/dev/null; then
    case "$m" in triagerx|deeptriage) rm -rf "$_heavy_lk" 2>/dev/null ;; esac
    rm -rf "$LOCKD/${m}_${ds}_${s}" 2>/dev/null
    echo "[$(date '+%m-%d %H:%M:%S')] [q$WID] DUP-SKIP $m $ds s$s（等待期间已被完成）" >> "$LOG"
    continue
  fi
  # 间歇性失败（这台共享机上出现过 exit=135 SIGBUS、内存抢占导致的 OOM）重试若干次，
  # 避免一次抖动把一格永久判死。
  ec=1
  for try in 1 2 3; do
    echo "[$(date '+%m-%d %H:%M:%S')] [q$WID] START run_${m}_strict $ds s$s try=$try" >> "$LOG"
    set +e; $PY "run_${m}_strict.py" "$ds" "$s" < /dev/null >> "$LOG" 2>&1; ec=$?; set -e
    echo "[$(date '+%m-%d %H:%M:%S')] [q$WID] END run_${m}_strict $ds s$s try=$try exit=$ec" >> "$LOG"
    [ "$ec" -eq 0 ] && break
    case "$ec" in 135|137|1) ;; *) break ;; esac
    rw=0
    while [ "$(avail_gb)" -lt "$MIN_MEM_GB" ]; do sleep 30; rw=$((rw+1)); [ "$rw" -ge 40 ] && break; done
    sleep 20
  done
  rmdir "$LOCKD/${m}_${ds}_${s}" 2>/dev/null || rm -rf "$LOCKD/${m}_${ds}_${s}" 2>/dev/null
  # 释放重活全局锁，让下一个重活格进场
  case "$m" in
    triagerx|deeptriage) rm -rf "$LOCKD/_heavy" 2>/dev/null ;;
  esac
  sleep 5
done < "$_SHUF"
