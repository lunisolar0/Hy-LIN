"""从各基线结果 JSON 重建中央代价台账 results/baselines_cost.csv。

为什么需要：
- `common.save_scores` 与 `run_hylin.py` 曾用两套字段写同一个 CSV，
  且多 worker 无锁并发追加，实测把文件写出了 NUL 块与列错位；
- 仍在跑的旧进程（如 deeptriage 补格）用的是加载时的旧 schema，
  它们落盘后 CSV 会再次出现缺列行。
本脚本按 12 列 schema 从 JSON 重建（JSON 才是唯一真相源），并把
device_tag 从 device_name 推断补齐。坏文件请先手工备份。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ.get("HXL_ROOT", "/data/projects/ys/Hy-LIN"))
OUT = ROOT / "results" / "baselines_cost.csv"
FIELDS = ["ts", "method", "dataset", "seed", "wall_s", "params",
          "gpu_peak_mib", "rss_peak_gb", "n_dev", "test_n",
          "device_tag", "device_name"]


def tag_from_name(name: str) -> str:
    if "A800" in name:
        return "A"
    if "5090" in name:
        return "N"
    if "4090" in name:
        return "R"
    return ""


def main() -> int:
    rows = []
    for path in glob.glob(str(ROOT / "results" / "baselines" / "*" / "seed*" / "*.json")):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = d.get("cost") or {}
        parts = Path(path).parts
        dataset, seed = parts[-3], int(parts[-2][4:])
        method = Path(path).stem
        test = d.get("test") or {}
        test_n = test.get("n") or (test.get("all") or {}).get("n")
        rows.append({
            "ts": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path))),
            "method": method, "dataset": dataset, "seed": seed,
            "wall_s": round(float(c.get("wall_s", c.get("wall_seconds", -1))), 1),
            "params": c.get("params", c.get("parameter_count", -1)),
            "gpu_peak_mib": c.get("gpu_peak_mib", c.get("gpu_max_allocated_mib", -1)),
            "rss_peak_gb": c.get("rss_peak_gb", ""),
            "n_dev": d.get("n_dev", -1),
            "test_n": test_n if test_n is not None else -1,
            "device_tag": c.get("device_tag", ""),
            "device_name": c.get("device_name", ""),
        })
    for r in rows:
        if not r["device_tag"]:
            r["device_tag"] = tag_from_name(str(r["device_name"]))
    rows.sort(key=lambda r: (r["method"], r["dataset"], r["seed"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"rebuild -> {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
