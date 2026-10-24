"""RQ4 代价表：在独占窗口重测被污染的格子。

为什么需要重测
--------------
代价测量必须在**无竞争**条件下做。实测同一方法、同一数据集的三个种子
耗时最多差 2.3 倍（sclbt/gc: 3356s vs 7610s vs 7278s）——这不可能是方法差异，
只能是多任务抢同一张卡所致。审稿人看到这种波动会质疑整张代价表。

重测规则（事先定死，不看结果再改）
--------------------------------
1. 对每个 (方法, 数据集) 取三个种子的 wall_s；
2. 若 (max-min)/min > 40%，判定该组被污染；
3. 组内**只重测最慢的那一格**：污染表现为"变慢"，快的几格测量本身可信；
4. 重测**严格串行**，一次一格，独占整张卡；
5. 重测会覆盖该格 json —— 这是刻意的：重跑后该格的指标与代价**同源自洽**，
   比"旧指标配新代价"更严谨；
6. 原值备份到 results/cost_clean/backup/，中央代价 CSV 追加新行，
   RQ4 代价表按 (method,dataset,seed) 取**最后一行**。

用法：
    python scripts/baselines/remeasure_cost.py            # 只报告
    python scripts/baselines/remeasure_cost.py --apply    # 真正重测
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("HXL_ROOT", "/data/projects/ys/Hy-LIN"))
BASE = ROOT / "results" / "baselines"
OUT = ROOT / "results" / "cost_clean"
BACKUP = OUT / "backup"
VARIANCE_TRIGGER = 0.40
COST_SEED = 42
DATASETS = ("mf", "mc", "gc")
SEEDS = (42, 43, 44)
METHODS = ("tfidf_svm", "deeptriage", "triagerx", "lightgcn", "cbtmf", "sclbt")

# 与 remote_runner.sh 同一套脚本命名：tfidf_svm/lightgcn 没有 _strict 后缀
def script_of(method: str) -> str:
    return f"run_{method}.py" if method in ("tfidf_svm", "lightgcn") else f"run_{method}_strict.py"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cells() -> dict:
    cells: dict = {}
    for ds in DATASETS:
        for s in SEEDS:
            for m in METHODS:
                p = BASE / ds / f"seed{s}" / f"{m}.json"
                if not p.exists():
                    continue
                try:
                    cells.setdefault((m, ds), {})[s] = (p, json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    continue
    return cells


def contaminated(cells: dict) -> list[tuple[str, str, int]]:
    out = []
    for (m, ds), per_seed in sorted(cells.items()):
        walls = [(s, float((d.get("cost") or {}).get("wall_s")))
                 for s, (_p, d) in per_seed.items() if (d.get("cost") or {}).get("wall_s")]
        if len(walls) < 2:
            continue
        vals = [w for _s, w in walls]
        if (max(vals) - min(vals)) / min(vals) > VARIANCE_TRIGGER:
            # 组内波动过大 = 整组都不可信。统一用同一个种子（COST_SEED）重测，
            # 让代价表里每个方法×数据集都出自同源、同条件的一次干净运行，
            # 而不是"组内挑一个看起来不慢的"。
            if COST_SEED in per_seed:
                out.append((m, ds, COST_SEED))
    # deeptriage 例外：它的格子在 11 路争抢下单格跑了 7-11 小时，
    # 方差判定对它失效（只测到一个种子）。但它是最慢的基线，
    # 代价列恰恰最吃它的数，所以无条件重测。
    for ds in DATASETS:
        key = ("deeptriage", ds)
        if key in cells and COST_SEED in cells[key] and (key[0], key[1], COST_SEED) not in out:
            out.append(("deeptriage", ds, COST_SEED))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--skip", default="", help="逗号分隔的方法名；其污染格本次不重测")
    args = ap.parse_args()

    cells = load_cells()
    todo_all = contaminated(cells)
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    todo = [t for t in todo_all if t[0] not in skip]
    if skip:
        kept = sorted({t[0] for t in todo_all if t[0] in skip})
        log(f"按 --skip 跳过：{','.join(sorted(skip))}（实际涉及 {','.join(kept) or '无'}）")
    log(f"被污染的格共 {len(todo_all)} 个（格间耗时波动 > {VARIANCE_TRIGGER:.0%}），本次重测 {len(todo)} 个")
    for m, ds, s in todo:
        per = {k: round(float((d.get('cost') or {}).get('wall_s') or 0), 1) for k, (_p, d) in cells[(m, ds)].items()}
        log(f"  {m:11s} {ds:3s} -> 重测 s{s}   现有: {per}")
    if not todo:
        log("无需重测")
        return 0
    if not args.apply:
        log("（仅报告，未实际重测；加 --apply 执行）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    for m, ds, s in todo:
        src, old = cells[(m, ds)][s]
        bak = BACKUP / f"{m}_{ds}_s{s}.json"
        if not bak.exists():
            shutil.copy2(src, bak)
        log(f"重测 {m} {ds} s{s} …")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["HYLIN_MEM_FRACTION"] = "0.45"
        cmd = [sys.executable, str(ROOT / "scripts" / "baselines" / script_of(m)), ds, str(s)]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT / "code"), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
        log(f"  exit={rc} 用时 {time.time()-t0:.0f}s")
        # 编排器靠这个文件数进度；它同时是这次干净运行的独立留痕
        fresh = BASE / ds / f"seed{s}" / f"{m}.json"
        if fresh.exists():
            try:
                payload = json.loads(fresh.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            (OUT / f"{m}_{ds}_s{s}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "_done.json").write_text(json.dumps(
        {"done_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n": len(todo),
         "cells": [f"{m}_{ds}_s{s}" for m, ds, s in todo]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log("代价重测完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
