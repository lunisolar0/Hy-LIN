"""基线公共：划分 / 文本 / 评测 / 落盘。所有基线统一走 data/processed + 时序位置切分 + ranking_metrics。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from hy_lin.data import load_records, chronological_split, build_chronological_split, TEXT_CHARS
from hy_lin.metrics import ranking_metrics, KS

ROOT = Path(__file__).resolve().parents[2]
import os as _os
# 输出位置可重定向：代价会话（cost_session.py）要在**独立目录**里重跑一遍基线，
# 只取代价，不许覆盖主结果（主结果是 RQ1 表的原料，覆盖了等于篡改实验记录）。
_OUT = _os.environ.get("HXL_BASELINE_OUT", "")
RES = (Path(_OUT) / "baselines") if _OUT else (ROOT / "results" / "baselines")
MODELS = (Path(_OUT) / "models") if _OUT else (ROOT / "results" / "baselines_models")
COST_CSV = (Path(_OUT) / "cost.csv") if _OUT else (ROOT / "results" / "baselines_cost.csv")

def load_split(dataset: str, seed: int, split: str = "chronological"):
    owners_s, texts = load_records(ROOT / "data" / "processed" / f"{dataset}.json")
    dev_names_all = sorted(set(owners_s))
    d2i = {n: i for i, n in enumerate(dev_names_all)}
    owners_all = np.asarray([d2i[n] for n in owners_s], dtype=np.int64)
    if split != "chronological":
        raise ValueError(f"基线只支持时序位置切分，收到 {split!r}；随机分层已作废（实验设置 §1.4）")
    owners, tr, va, te = build_chronological_split(owners_all)
    dev_names = [dev_names_all[d] for d in np.unique(owners_all[tr])]
    return owners, texts, dev_names, tr, va, te

def model_path(name: str, dataset: str, seed: int, suffix: str = "pt") -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    return MODELS / f"{name}_{dataset}_s{seed}.{suffix}"


def device_info() -> dict:
    """算力来源标记：本项目跨两台异构机器（R=2×RTX4090 本机，A=A800 云机）。

    代价必须分开记录——两台卡的单卡显存（24G vs 80G）与吞吐差异很大，
    混在一起算出来的"代价"没有意义。约定用环境变量 HXL_DEVICE 传 "R"/"A"。
    """
    import os, platform
    info: dict = {}
    tag = os.environ.get("HXL_DEVICE", "").strip()
    if tag:
        info["device_tag"] = tag
    name = ""
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception:
        name = ""
    info["device_name"] = name or platform.processor() or platform.machine()
    if tag == "A":
        info["machine"] = "A800"
    elif tag == "R":
        info["machine"] = "RTX4090"
    return info


def collect_cost(t0: float, model=None) -> dict:
    """wall_s / params / gpu_peak / rss_peak + 设备来源，快照式采集，永不抛错。"""
    import time, resource
    out: dict = {"wall_s": time.time() - t0}
    out.update(device_info())
    try:
        out["params"] = sum(p.numel() for p in model.parameters()) if model is not None else -1
    except Exception:
        out["params"] = -1
    try:
        out["gpu_peak_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0
    except Exception:
        out["gpu_peak_mib"] = -1
    try:
        out["rss_peak_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)
    except Exception:
        out["rss_peak_gb"] = -1
    return out


def save_scores(name: str, dataset: str, seed: int, test_scores: np.ndarray, test_labels: np.ndarray,
                val_scores: np.ndarray | None = None, val_labels: np.ndarray | None = None, extra: dict | None = None,
                cost: dict | None = None):
    import csv, time
    d = RES / dataset / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}_test_scores.npy", test_scores.astype(np.float32))
    np.save(d / f"{name}_test_labels.npy", test_labels.astype(np.int64))
    m = ranking_metrics(torch.from_numpy(test_scores), torch.from_numpy(test_labels))
    out = {"method": name, "dataset": dataset, "seed": seed, "n_dev": int(test_scores.shape[1]),
           "test": m, "extra": extra or {}, "cost": cost or {}}
    if val_scores is not None:
        out["val"] = ranking_metrics(torch.from_numpy(val_scores), torch.from_numpy(val_labels))
    (d / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # 中央代价表（RQ4 直接用）：method,dataset,seed,wall_s,params,gpu_peak_mib,rss_peak_gb
    # 多 worker 会同时追加同一文件；不锁会出现交错写，实测曾把 CSV 写出 NUL 块。
    c = cost or {}
    import fcntl
    COST_CSV.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(COST_CSV) + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        new = not COST_CSV.exists()
        with open(COST_CSV, "a", newline="") as f:
            # 与 run_hylin.py 保持完全相同的 schema：两边字段不一致会写出错列/错位行，
            # 这是之前 CSV 出现 NUL 块与列错位的根因之一。
            w = csv.DictWriter(f, fieldnames=["ts", "method", "dataset", "seed", "wall_s", "params",
                                              "gpu_peak_mib", "rss_peak_gb", "n_dev", "test_n",
                                              "device_tag", "device_name"])
            if new: w.writeheader()
            w.writerow({"ts": time.strftime("%m-%d %H:%M"), "method": name, "dataset": dataset, "seed": seed,
                        "wall_s": round(float(c.get("wall_s", -1)), 1), "params": c.get("params", -1),
                        "gpu_peak_mib": c.get("gpu_peak_mib", -1), "rss_peak_gb": c.get("rss_peak_gb", -1),
                        "n_dev": int(test_scores.shape[1]), "test_n": int(test_labels.shape[0]),
                        "device_tag": c.get("device_tag", ""), "device_name": c.get("device_name", "")})
            f.flush(); _os.fsync(f.fileno())
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print(f"[{name}/{dataset}/s{seed}] " + " ".join(f"{k}={m[k]:.4f}" for k in ["MRR"] + [f"HR@{k}" for k in KS] + [f"NDCG@{k}" for k in KS]))
    return out
