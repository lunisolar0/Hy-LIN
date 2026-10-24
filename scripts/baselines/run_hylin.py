import os
"""Hy-LIN RQ1 行：锁定配置跑 9 格（3 数据集×3 种子），输出转 baselines 口径 + 代价进中央表。
输出：results/hylin_runs/<ds>_s<seed>/{metrics.json,best.pt}（与探究 code/outputs 隔离），
      results/baselines/<ds>/seed<seed>/hylin.json（供 make_table.py），代价追加 results/baselines_cost.csv。
等 6 基线 54 格齐后才开跑（串行，不抢卡），跑完发邮件。"""
import json
import sys, subprocess, sys, time, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
# 锁点必须已落地且字段完整，否则明确失败——见 scripts/analysis/lock_guard.py
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from lock_guard import load_lock
CFG = load_lock(ROOT)
_OUT = os.environ.get("HXL_BASELINE_OUT", "")
RES = (Path(_OUT) / "baselines") if _OUT else (ROOT / "results" / "baselines")
RUNS = (Path(_OUT) / "hylin_runs") if _OUT else (ROOT / "results" / "hylin_runs")
COST_CSV = (Path(_OUT) / "cost.csv") if _OUT else (ROOT / "results" / "baselines_cost.csv")
# 用启动本脚本的解释器拉子进程：写死本机路径会让 RQ2/RQ3 在 A800 上直接 FileNotFoundError
PY = os.environ.get("HYLIN_PYTHON") or sys.executable

def free_min():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
        return min(int(x) for x in o.replace("MiB", "").split() if x.strip().isdigit())
    except Exception:
        return 10**9

def wait_baselines():
    # 原设计是"等 54 格基线全齐再开跑"，为本机 24G 卡不与基线抢显存。
    # A800 有 80G，且 RQ4 要求 Hy-LIN 的代价也来自 A800，两者并行才成立。
    if os.environ.get("HXL_HYLIN_WAIT_BASELINES", "1") == "0":
        print("[hylin] 跳过等基线（A800 模式）", flush=True)
        return
    while True:
        n = sum(1 for ds in ("gc", "mc", "mf") for s in (42, 43, 44)
                for m in ("tfidf_svm", "deeptriage", "triagerx", "lightgcn", "cbtmf", "sclbt")
                if (RES / ds / f"seed{s}" / f"{m}.json").exists())
        if n >= 54: return
        print(f"[hylin] 等基线 {n}/54，60s 后再查", flush=True)
        time.sleep(60)

def to_baselines(ds, seed, mpath):
    mj = json.loads((mpath / "metrics.json").read_text())
    tall = mj["test"]["all"]
    d = RES / ds / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    out = {"method": "hylin", "dataset": ds, "seed": seed, "n_dev": mj["n_dev"], "test": tall,
           "extra": {"locked": True},
           "cost": {"wall_s": mj["cost"]["wall_seconds"], "params": mj["cost"]["parameter_count"],
                    "gpu_peak_mib": mj["cost"]["gpu_max_allocated_mib"], "device": mj["cost"].get("device_name", "")}}
    (d / "hylin.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # 把 --save-scores 落盘的打分矩阵/标签/训练频数一并搬到 baselines 目录，
    # 与其它基线同名同放。缺了这三份，§2.2 显著性、§2.3 硬负、§3.3 案例、
    # §4 容量仿真全部无法计算（reviewer_extras/operational_analysis 都读这里），
    # 而它们事后无法补算——必须在这一轮就搬。
    import shutil as _shutil
    for fname in ("test_scores.npy", "test_labels.npy", "train_counts.npy"):
        src = mpath / fname
        if src.exists():
            _shutil.copyfile(src, d / f"hylin_{fname}")
        else:
            print(f"[warn] {mpath} 缺 {fname}（--save-scores 未生效？）", flush=True)
    # 多 worker 并发追加同一 CSV：必须与 common.save_scores 共用同一把锁，
    # 否则会出现交错行；字段也必须完全一致（12 列，含 device_tag/device_name）。
    import fcntl
    lock = open(str(COST_CSV) + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        new = not COST_CSV.exists()
        with open(COST_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ts", "method", "dataset", "seed", "wall_s", "params",
                                              "gpu_peak_mib", "rss_peak_gb", "n_dev", "test_n",
                                              "device_tag", "device_name"])
            if new: w.writeheader()
            w.writerow({"ts": time.strftime("%m-%d %H:%M"), "method": "hylin", "dataset": ds, "seed": seed,
                        "wall_s": mj["cost"]["wall_seconds"], "params": mj["cost"]["parameter_count"],
                        "gpu_peak_mib": mj["cost"]["gpu_max_allocated_mib"], "rss_peak_gb": "",
                        "n_dev": mj["n_dev"], "test_n": mj["n_test"],
                        "device_tag": mj["cost"].get("device_tag", ""),
                        "device_name": mj["cost"].get("device_name", "")})
            f.flush(); os.fsync(f.fileno())
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

def run_one(ds, seed, tag):
    import os
    out = RUNS / f"{ds}_s{seed}"
    if (RES / ds / f"seed{seed}" / "hylin.json").exists():
        print(f"[{tag} SKIP] hylin {ds} s{seed}", flush=True); return
    while free_min() < 2000:
        print(f"[{tag}] 显存不足，等 30s", flush=True); time.sleep(30)
    cfg = CFG
    cmd = [PY, "-m", "hy_lin.train", "--data", str(ROOT / "data" / "processed" / f"{ds}.json"),
           "--cache", str(ROOT / "code" / "cache" / f"graph_{ds}_minilm_s{seed}_kb{cfg['k_bug']}_kd{cfg['k_dev']}_centroid.pt"),
           "--out", str(out), "--seed", str(seed), "--encoder", cfg["encoder"],
           "--dev-graph", cfg["dev_graph"], "--hidden", str(cfg.get("hidden", 384)),
           "--k-bug", str(cfg["k_bug"]), "--k-dev", str(cfg["k_dev"]),
           "--exp-layers", str(cfg["exp_layers"]), "--imp-layers", str(cfg["imp_layers"]),
           "--dropout", str(cfg["dropout"]), "--lr", str(cfg["lr"]),
           "--weight-decay", str(cfg.get("weight_decay", 0.0001)),
           "--batch-size", str(cfg.get("batch_size", 2048)), "--epochs", str(cfg.get("epochs", 40)),
           "--patience", str(cfg.get("patience", 10)), "--tau", str(cfg.get("tau", 0.0)),
           "--freq-exponent", str(cfg.get("freq_exponent", 0.5)), "--fusion", cfg["fusion"],
           "--story", str(cfg.get("story", 0.5)), "--low-coef", str(cfg.get("low_coef", 0.35)),
           "--imp-query", cfg.get("imp_query", "prop"), "--gate-mode", cfg["gate_mode"],
           "--valve", cfg.get("valve", "bug"),
           # RQ1 九格必须落盘打分矩阵与 train_counts：这是 R1-C14 逐案例比较
           # 和 RQ4 容量仿真唯一的原料，事后无法补算，必须在这一轮就存下来。
           "--save-scores"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=tag[-1])
    print(f"[{tag}] START hylin {ds} s{seed}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT / "code"), env=env)
    print(f"[{tag}] END hylin {ds} s{seed} exit={r.returncode}", flush=True)
    if r.returncode == 0:
        to_baselines(ds, seed, out)

def _acquire_singleton():
    """整机只允许一个 RQ1 实例。

    为什么必须：多个 run_hylin 实例会**同时写同一个输出目录**
    （results/hylin_runs/<ds>_s<seed>），互相覆盖 best.pt 与 metrics.json；
    而 watchdog 在"没检测到实例"的瞬间会误启动第二个（SSH 命令失败、
    进程尚未出现在 ps 里，都会被判成"没在跑"）。
    实测出现过 5 个实例同时写一个目录，训练成果全部作废。

    这里用 flock 做整机互斥：后来的实例立刻退出，不产生任何副作用。
    """
    import fcntl
    lock_dir = ROOT / "results" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    fh = (lock_dir / "hylin_run.lock").open("a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[hylin] 已有 RQ1 实例在跑，本实例退出", flush=True)
        raise SystemExit(0)
    fh.write(str(os.getpid()))
    fh.flush()
    return fh  # 保持引用，进程退出时自动释放


def main():
    import os
    _lock_fh = _acquire_singleton()
    wait_baselines()
    tasks = [(ds, s) for ds in ("gc", "mc", "mf") for s in (42, 43, 44)]
    from multiprocessing import Process
    def worker(tag, parity, _n):
        os.environ["CUDA_VISIBLE_DEVICES"] = tag[-1]
        for i, (ds, s) in enumerate(tasks):
            # 分片必须按**实际起的 worker 数**切，不能硬编码 2。
            # 原来写死 `i % 2`，而 worker 数 _n 取自 GPU 数量：
            # 单卡机器只起 parity=0 一个 worker -> parity=1 的 4 格
            # **永远不会被任何 worker 领走**，主表 9 格会缺一半。
            # 实测：迁到单卡 5090 后只产出 2/9，正是 parity 0 的那几格。
            if i % _n != parity:
                continue
            run_one(ds, s, tag)
    # 单卡实例上起两个 worker 会让第二个指向不存在的卡 1，
    # 并把两格挤到同一张卡上——RQ4 的代价测量会因此失真。
    import subprocess as _sp
    try:
        _n = len([l for l in _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                     timeout=20).stdout.splitlines() if l.strip()])
    except Exception:
        _n = 1
    _n = max(1, _n)
    ps = [Process(target=worker, args=(f"gpu{i}", i, _n)) for i in range(_n)]
    [x.start() for x in ps]; [x.join() for x in ps]
    subprocess.run([PY, str(ROOT / "scripts" / "baselines" / "make_table.py")])
    subprocess.run([PY, str(ROOT / "monitor" / "mailer.py"),
                    "Hy-LIN RQ1：hylin 9/9 完成", "大表 63 格齐，tables/rq1_big.csv 已更新。"])

if __name__ == "__main__":
    main()
