"""RQ4 代价会话：在**独占且安静**的机器上，串行重测全部方法的代价。

为什么需要独立会话
------------------
代价是唯一会被资源竞争污染的量：多任务抢同一张卡时，wall_s 只会被拉长。
实测同一方法同一数据集的三个种子耗时最多差 2.3 倍（sclbt/gc：
3356s vs 7610s vs 7278s），这只能是竞争造成的。RQ4 要论证"代价可接受"，
这张表一旦被指认受污染，整节的可部署性论证就塌了。

协议（事先定死）
----------------
1. **严格串行**：一次只跑一格，整机无其他 GPU 任务；
2. **同一台机器**：全部方法（6 基线 + Hy-LIN）在同一台机器上测，
   满足"单一机器原则"——表内数字可直接横向比较；
3. **独立输出目录**：只取代价，**不覆盖主结果**（主结果是 RQ1 表的原料，
   覆盖等于篡改实验记录）；
4. **同种子同配置**：seed 42，与主实验同一份锁点、同一套代码；
5. **从便宜到贵**：先跑便宜的，早点拿到完整表；最贵的 deeptriage 排在最后，
   即使时间不够，缺的也只是一格而不是整行。

为什么放在本机而不是 A800
------------------------
A800 只有 1 张卡，而 RQ2/RQ3 的 519 格也要用它，代价会话若排在其后，
会成为关键路径的尾巴。本机 2×RTX 4090 平时空着（只跑同步与监管），
把代价会话放这里，等于把它从 A800 关键路径上完全摘掉。
代价表本身是"同一台机器上全部方法"的横向比较，不要求必须是 A800——
而且 4090 是工业界真实的部署机型，比云卡更贴近"可部署"这个论断。

用法：
    python scripts/baselines/cost_session.py            # 报告计划
    python scripts/baselines/cost_session.py --run      # 实际执行
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("HXL_ROOT", "/data/projects/ys/Hy-LIN"))
OUT = ROOT / "results" / "cost_session"
SEED = 42
DATASETS = ("mf", "mc", "gc")

# 脚本名与 remote_runner.sh 同一套：tfidf_svm / lightgcn 没有 _strict 后缀
SCRIPT = {
    "tfidf_svm": "run_tfidf_svm.py",
    "lightgcn": "run_lightgcn.py",
    "cbtmf": "run_cbtmf_strict.py",
    "sclbt": "run_sclbt_strict.py",
    "triagerx": "run_triagerx_strict.py",
    "deeptriage": "run_deeptriage_strict.py",
}
# 从便宜到贵：贵的方法排后面，时间不够时缺的只是一格
ORDER = ["lightgcn", "cbtmf", "hylin", "tfidf_svm", "sclbt", "triagerx", "deeptriage"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cells() -> list[tuple[str, str]]:
    out = []
    for m in ORDER:
        for ds in DATASETS:
            out.append((m, ds))
    return out


def run_cell(method: str, dataset: str, dest_root: Path | None = None) -> dict | None:
    """串行跑一格，返回 {method, dataset, seed, cost{...}} 或 None。"""
    tag = f"{method}_{dataset}_s{SEED}"
    dest = (dest_root / tag) if dest_root else (OUT / tag)
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HXL_BASELINE_OUT"] = str(dest)
    env["HXL_DEVICE"] = "R"
    env["HYLIN_MEM_FRACTION"] = "0.45"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    if method == "hylin":
        # Hy-LIN 走锁点配置；复用 run_hylin.run_one，保证与 RQ1 九格同配置
        cmd = [sys.executable, str(ROOT / "scripts" / "baselines" / "cost_session_hylin.py"),
               dataset, str(SEED)]
        cwd = str(ROOT)
    else:
        cmd = [sys.executable, str(ROOT / "scripts" / "baselines" / SCRIPT[method]), dataset, str(SEED)]
        cwd = str(ROOT / "code")

    t0 = time.time()
    with (dest / "run.log").open("w", encoding="utf-8") as h:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=h, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0:
        log(f"  !! {tag} exit={rc}")
        return None

    src = dest / "baselines" / dataset / f"seed{SEED}" / f"{method}.json"
    if not src.exists():
        log(f"  !! {tag} 未产出 {method}.json")
        return None
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return None
    cost = payload.get("cost") or {}
    if not cost.get("wall_s"):
        cost = dict(cost)
        cost["wall_s"] = round(wall, 1)
    payload["cost"] = cost
    payload["cost"]["measured_in"] = "cost_session"
    payload["cost"]["device_tag"] = "R"
    (dest / "cost.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload



def probe() -> int:
    """污染探针：第二张卡跑 RQ2/RQ3 时，代价会话的测量还准吗？

    代价会话跑在 GPU 0，RQ2/RQ3 worker 跑在 GPU 1——两张卡不共享显存与算力，
    但共享 CPU 与 PCIe。线程封顶到 8+8（72 核）后理论上干扰很小，
    但"理论上"不算数：这里在同一台机器上重测一格已测过的廉价格，
    与安静窗口的基线逐项对比，把影响**量化**出来写进协议。

    判据：若 wall_s 相对偏差 > 10%，则代价表不可信，
    必须改为"RQ2 worker 与代价会话在时间上错开"（牺牲并行度换可信度）。
    """
    target = ("hylin", "mf")
    base_file = OUT / f"{target[0]}_{target[1]}_s{SEED}" / "cost.json"
    if not base_file.exists():
        log(f"探针需要基线 {base_file}，请先跑过一格 {target[0]}_{target[1]}")
        return 1
    base = json.loads(base_file.read_text(encoding="utf-8")).get("cost") or {}
    log(f"探针：重测 {target[0]}_{target[1]}（基线 wall_s={base.get('wall_s')}）")
    p = run_cell(target[0], target[1], dest_root=OUT / "_probe")
    if p is None:
        log("探针格子跑失败")
        return 1
    c = p.get("cost") or {}
    b, n = float(base.get("wall_s") or 0), float(c.get("wall_s") or 0)
    if b <= 0:
        log("基线无效")
        return 1
    delta = (n - b) / b
    verdict = "可接受" if abs(delta) <= 0.10 else "不可接受（需让两卡时间错开）"
    log(f"探针结果：基线 {b:.1f}s -> 现在 {n:.1f}s，相对偏差 {delta:+.1%}，判定：{verdict}")
    (OUT / "_probe" / "verdict.json").write_text(json.dumps({
        "cell": f"{target[0]}_{target[1]}_s{SEED}",
        "baseline_wall_s": b, "probed_wall_s": n,
        "relative_delta": round(delta, 4), "verdict": verdict,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="实际执行（默认只报告计划）")
    ap.add_argument("--probe", action="store_true", help="污染探针：重测一格并与安静基线对比")
    args = ap.parse_args()

    if args.probe:
        return probe()

    plan = cells()
    log(f"代价会话计划：{len(plan)} 格（{len(ORDER)} 方法 × {len(DATASETS)} 数据集 × seed {SEED}），严格串行")
    for m, ds in plan:
        log(f"  {m:11s} {ds}")

    if not args.run:
        log("（仅报告，未执行；加 --run 实际跑）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    done_marker = OUT / "_done.json"
    summary = []
    for m, ds in plan:
        tag = f"{m}_{ds}_s{SEED}"
        cp = OUT / tag / "cost.json"
        if cp.exists():
            log(f"跳过 {tag}（已有）")
            try:
                summary.append(json.loads(cp.read_text(encoding="utf-8")))
            except Exception:
                pass
            continue
        log(f"=== 测 {tag} ===")
        p = run_cell(m, ds)
        if p is None:
            continue
        c = p.get("cost") or {}
        log(f"    wall_s={c.get('wall_s')}  gpu_peak={c.get('gpu_peak_mib')}MiB  params={c.get('params')}")
        summary.append(p)

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    done_marker.write_text(json.dumps(
        {"done_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n": len(summary)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log(f"代价会话完成：{len(summary)}/{len(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
