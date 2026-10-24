"""锁点守卫：读 configs/locked_model.json 并强制 fail fast。

为什么需要
----------
configs/locked_model.json 在探究未结束前只有
`{"status": "PENDING_EXPLORATION", "note": ...}`，里面没有 k_bug / exp_layers 等字段。
此时下游脚本如果直接 `CFG["k_bug"]`，会抛一个语焉不详的 KeyError；
更糟的情形是有人为了"让它跑起来"而临时回退到旧锁点——
那等于在用旧随机划分的锁点跑时序口径的实验，结果无法自证。

所以这里统一：锁点未落地就明确失败，不给"先跑起来再说"留口子。
"""
from __future__ import annotations

import json
from pathlib import Path

# 锁点必须包含的字段。缺任何一个都说明锁点没写完整，不能拿去跑。
REQUIRED = (
    "k_bug", "k_dev", "exp_layers", "imp_layers",
    "dropout", "lr", "gate_mode", "fusion", "encoder",
)


class LockNotReady(RuntimeError):
    """锁点尚未落地，或内容不完整。"""


def load_lock(root: Path) -> dict:
    """读锁点；未落地或字段不全则抛 LockNotReady。"""
    path = Path(root) / "configs" / "locked_model.json"
    if not path.exists():
        raise LockNotReady(
            f"锁点文件不存在：{path}\n"
            "探究尚未产出锁点时不得跑下游实验——否则用的是旧锁点，结果无法自证。"
        )
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LockNotReady(f"锁点文件不是合法 JSON：{path}（{exc}）") from exc
    status = cfg.get("status")
    if status == "PENDING_EXPLORATION" or not isinstance(cfg, dict) or not cfg.get("k_bug"):
        raise LockNotReady(
            f"锁点尚未落地（status={status!r}）：{path}\n"
            "先等 hy_lin.study 完成两阶段门控并写入锁点，再跑 RQ1/RQ2/RQ3。"
        )
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise LockNotReady(
            f"锁点字段不完整，缺 {missing}：{path}\n"
            "锁点不完整时跑出来的格子无法与锁点对齐（层网格/敏感性网格会静默复用错格子）。"
        )
    return cfg
