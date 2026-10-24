"""代价会话里 Hy-LIN 的单格运行器：复用 run_hylin.run_one，保证与 RQ1 九格同配置。

为什么要单独一个脚本：run_hylin.py 的 main() 会把 9 格全跑一遍，
而代价会话需要**严格串行**地一格一格跑（同时也是安静窗口的前提）。
这里只借用它的 run_one（含锁点校验与打分矩阵落盘），不做并发。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("HXL_ROOT", "/data/projects/ys/Hy-LIN"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))


def main() -> int:
    if len(sys.argv) < 3:
        print("用法: cost_session_hylin.py <dataset> <seed>")
        return 2
    dataset, seed = sys.argv[1], int(sys.argv[2])
    # lock_guard 会在锁点缺失/字段不全时明确失败——这是有意的 fail-fast
    from lock_guard import load_lock
    load_lock(ROOT)
    import run_hylin
    # CUDA_VISIBLE_DEVICES 已在环境里设为 0；tag 用 "gpu0" 保持与 run_one 一致
    run_hylin.run_one(dataset, seed, "gpu0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
