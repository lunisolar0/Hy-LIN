"""在处理后的最小数据集 Firefox（mf）上训练并报告 HR、NDCG、MRR。

用法（在 code 目录下，使用 ys310）:
    python -m hy_lin.train --data ../data/processed/mf.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hy_lin.data import prepare, split_views
from hy_lin.guides import node2vec_knn, random_knn
from hy_lin.metrics import (
    KS,
    _self_check,
    centroid_scores,
    format_block,
    popularity_scores,
    sliced_metrics,
)
from hy_lin.model import HyLIN


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_i, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_i, best_free = i, free
    return torch.device(f"cuda:{best_i}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="训练 Hy-LIN 缺陷负责人预测")
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "mf.json")
    parser.add_argument("--cache", type=Path, default=root / "code" / "cache" / "mf_graph.pt")
    parser.add_argument("--out", type=Path, default=root / "code" / "outputs" / "mf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", choices=("minilm", "bert", "roberta"), default="minilm")
    parser.add_argument("--dev-graph", choices=("centroid", "overlap"), default="centroid")
    parser.add_argument(
        "--split",
        choices=("chronological",),
        default="chronological",
        help="固定为对齐前人的 80/10/10 交付顺序位置切分；随机分层已作废（实验设置 §1.4）",
    )
    parser.add_argument(
        "--bug-graph",
        choices=("text", "random", "node2vec"),
        default="text",
        help="图增强候选图的引导来源：text 为锁定用文本近邻；random / node2vec 为非文本对照（C5 消融）",
    )
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--k-bug", type=int, default=10)
    parser.add_argument("--k-dev", type=int, default=15)
    parser.add_argument("--exp-layers", type=int, default=2)
    parser.add_argument("--imp-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.0, help="logit adjustment 强度，0 表示不校正频率")
    parser.add_argument(
        "--freq-exponent",
        type=float,
        default=0.0,
        help="训练样本权重为 1/count^exponent。0 表示每条缺陷等权，0.5 会抬低频、压整体",
    )
    parser.add_argument(
        "--fusion",
        choices=("repr", "score", "zscore", "mixture", "residual"),
        default="zscore",
        help="门控之后如何比较两路。zscore 先把每条路在开发者之间标准化，再按频率混合",
    )
    parser.add_argument("--story", type=float, default=0.5, help="分路损失权重：高频监督明线，低频监督暗线")
    parser.add_argument("--low-coef", type=float, default=0.35, help="早停分数里低频验证 MRR 的权重")
    parser.add_argument("--uniform-valve", action="store_true", help="阀门固定为均匀权重，用于消融图增强")
    parser.add_argument("--force-gate", choices=("explicit", "implicit"), default=None)
    parser.add_argument("--save-scores", action="store_true",
                        help="落盘 test_scores/test_labels/train_counts（定性分析原料，见 docs/定性实验设计.md）")
    parser.add_argument("--gate-mode", choices=("free", "bounded", "freq"), default="freq")
    parser.add_argument(
        "--gate-decay",
        type=float,
        default=0.0,
        help="时间衰减门（§6.1）：λ>0 时用按交付顺序指数衰减的计数替换 log1p(计数)；0 即原频率门",
    )
    parser.add_argument("--imp-keep", type=float, default=None, help="暗线保留自身画像的初始强度，越大越少借邻居")
    parser.add_argument("--imp-query", choices=("prop", "text"), default="prop")
    parser.add_argument(
        "--valve",
        choices=("bug", "hetero", "none"),
        default="bug",
        help="bug 是缺陷–缺陷软边；hetero 是缺陷–其他开发者的异质软边；none 表示不增强",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, tensors, query, dev_knn, dev_w, log_freq, prior, tau: float, seen_mask=None) -> torch.Tensor:
    model.eval()
    logits, _gate = model.encode_queries(
        tensors["emb"],
        tensors["owner"],
        tensors["knn"],
        query["emb"],
        query["knn"],
        dev_knn,
        dev_w,
        log_freq,
    )
    logits = logits - tau * prior
    if seen_mask is not None:
        logits = torch.where(seen_mask.view(1, -1), logits, torch.full_like(logits, float("-inf")))
    return logits


def heterogeneous_candidates(
    bug_emb: torch.Tensor,
    train_emb: torch.Tensor,
    train_owner: torch.Tensor,
    n_dev: int,
    k: int,
    row_owner: torch.Tensor | None,
) -> torch.Tensor:
    """每个缺陷连向文本质心最接近的 k 个开发者。训练行会去掉真实负责人，阀门不与原水渠重叠。"""
    summed = torch.zeros(n_dev, train_emb.size(1), device=train_emb.device, dtype=train_emb.dtype)
    summed.index_add_(0, train_owner, F.normalize(train_emb, dim=-1))
    counts = torch.bincount(train_owner, minlength=n_dev).clamp(min=1).unsqueeze(-1).float()
    centroids = F.normalize(summed / counts, dim=-1)
    take = k + 1 if row_owner is not None else k
    sim = F.normalize(bug_emb, dim=-1) @ centroids.T
    idx = torch.topk(sim, take, dim=-1).indices
    if row_owner is None:
        return idx
    is_owner = idx == row_owner.unsqueeze(1)
    has_owner = is_owner.any(dim=1)
    drop = is_owner.clone()
    drop[~has_owner, -1] = True
    return idx[~drop].view(-1, k)


def story_loss(s_exp, s_imp, target, counts, prior, tau: float, weight: float) -> torch.Tensor:
    """高频样本只督促明线认人，低频样本只督促暗线认人。"""
    extra = s_exp.new_zeros(())
    if weight <= 0:
        return extra
    adjusted_exp = s_exp - tau * prior
    adjusted_imp = s_imp - tau * prior
    high = counts[target] >= 20
    low = counts[target] < 5
    terms = 0
    if int(high.sum()) > 0:
        extra = extra + F.cross_entropy(adjusted_exp[high], target[high])
        terms += 1
    if int(low.sum()) > 0:
        extra = extra + F.cross_entropy(adjusted_imp[low], target[low])
        terms += 1
    if terms == 0:
        return extra
    return weight * extra


def gate_summary(gate: torch.Tensor, counts: torch.Tensor) -> dict[str, float]:
    out = {}
    bands = {
        "low(<5)": counts < 5,
        "mid(5-19)": (counts >= 5) & (counts < 20),
        "high(>=20)": counts >= 20,
        "all": torch.ones_like(counts, dtype=torch.bool),
    }
    for name, mask in bands.items():
        if int(mask.sum()) == 0:
            continue
        out[name] = float(gate[mask].mean())
    return out


def repro_info():
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, cwd="/data/projects/ys/Hy-LIN").stdout.strip() or None
    except Exception:
        commit = None
    return {"git_commit": commit, "hardware": "2x RTX 4090 24GB", "protocol": "EXPERIMENT_PROTOCOL.md v2.0"}


def main() -> None:
    args = parse_args()
    started = time.time()
    _self_check()
    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        # 卡有约 24GB。限制在 75%，进程自己报错退出，避免显存打满后整机重启。
        fraction = float(os.environ.get("HYLIN_MEM_FRACTION", "0.75"))
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
        print(f"设备 {device} {torch.cuda.get_device_name(device)}，显存上限 {fraction:.0%}")
    else:
        print("设备 cpu")

    bundle = prepare(
        args.data,
        args.cache,
        args.seed,
        args.k_bug,
        args.k_dev,
        device,
        encoder_key=args.encoder,
        dev_graph=args.dev_graph,
        split=args.split,
    )
    train = split_views(bundle, "train")
    val = split_views(bundle, "val")
    test = split_views(bundle, "test")
    if args.bug_graph != "text":
        n_train = train["knn"].size(0)
        if args.bug_graph == "random":
            train["knn"] = random_knn(n_train, n_train, args.k_bug, args.seed)
            val["knn"] = random_knn(val["knn"].size(0), n_train, args.k_bug, args.seed + 1)
            test["knn"] = random_knn(test["knn"].size(0), n_train, args.k_bug, args.seed + 2)
            print(f"引导来源 random：增强候选图改为随机训练缺陷（种子 {args.seed}）")
        else:
            print(f"引导来源 node2vec：在缺陷–负责人二部图上取结构近邻（种子 {args.seed}）")
            train["knn"], val["knn"], test["knn"] = node2vec_knn(bundle, args.k_bug, args.seed)
    counts = bundle["train_counts"].float()
    n_dev = int(bundle["n_dev"])
    print(
        f"缺陷 train/val/test = {len(train['owner'])}/{len(val['owner'])}/{len(test['owner'])}，"
        f"负责人 {n_dev}，训练频数中位数 {int(counts.median())}，"
        f"<5 的负责人 {(counts < 5).sum().item()}，>=20 的负责人 {(counts >= 20).sum().item()}"
    )

    tensors = {key: value.to(device) for key, value in train.items()}
    val_t = {key: value.to(device) for key, value in val.items()}
    test_t = {key: value.to(device) for key, value in test.items()}
    dev_knn = bundle["dev_knn"].to(device)
    dev_w = bundle["dev_w"].to(device)
    counts_d = counts.to(device)
    if args.gate_decay > 0:
        # 时间衰减门（§6.1）：交付顺序即时间的代理。age 归一化到 [0,1]，
        # 权重 exp(-λ·age)，最近一条记 1、最老一条记 e^{-λ}。
        tr_idx = bundle["train_idx"].float()
        span = (tr_idx.max() - tr_idx.min()).clamp(min=1.0)
        age = (tr_idx.max() - tr_idx) / span
        w = torch.exp(-args.gate_decay * age).to(device)
        decayed = torch.bincount(tensors["owner"], weights=w, minlength=n_dev)
        log_freq = torch.log1p(decayed.clamp(min=0.0))
        print(f"时间衰减门 λ={args.gate_decay}: 频数均值 {decayed.mean():.2f}（原始 {counts_d.mean():.2f}）")
    else:
        log_freq = torch.log1p(counts_d)
    counts_floor = counts_d.clamp(min=1.0)
    prior = torch.log(counts_floor / counts_floor.sum())

    model = HyLIN(
        n_dev=n_dev,
        text_dim=tensors["emb"].size(1),
        hidden=args.hidden,
        n_exp=args.exp_layers,
        n_imp=args.imp_layers,
        dropout=args.dropout,
        fusion=args.fusion,
    ).to(device)
    model.uniform_valve = args.uniform_valve
    model.force_gate = args.force_gate
    model.gate_mode = args.gate_mode
    model.imp_query = args.imp_query
    model.valve_type = args.valve
    if args.valve == "hetero":
        tensors["knn"] = heterogeneous_candidates(
            tensors["emb"], tensors["emb"], tensors["owner"], n_dev, args.k_bug, tensors["owner"]
        )
        val_t["knn"] = heterogeneous_candidates(
            val_t["emb"], tensors["emb"], tensors["owner"], n_dev, args.k_bug, None
        )
        test_t["knn"] = heterogeneous_candidates(
            test_t["emb"], tensors["emb"], tensors["owner"], n_dev, args.k_bug, None
        )
    if args.imp_keep is not None:
        with torch.no_grad():
            for layer in model.imp_layers:
                layer.keep.fill_(args.imp_keep)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_state = None
    best_select = -1.0
    best_mrr = -1.0
    best_epoch = -1
    stall = 0
    history = []
    n_train = tensors["emb"].size(0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n_train, device=device)
        total_loss = 0.0
        seen = 0
        last_gate = None
        t0 = time.time()
        for start in range(0, n_train, args.batch_size):
            batch = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, gate, _w, s_exp, s_imp = model.encode_train_batch(
                tensors["emb"],
                tensors["owner"],
                tensors["knn"],
                batch,
                dev_knn,
                dev_w,
                log_freq,
            )
            logits = logits - args.tau * prior
            target = tensors["owner"][batch]
            per = F.cross_entropy(logits, target, reduction="none")
            if args.freq_exponent > 0:
                sample_w = counts_d[target].clamp(min=1).pow(-args.freq_exponent)
                sample_w = sample_w / sample_w.mean().clamp(min=1e-6)
                loss = (per * sample_w).mean()
            else:
                loss = per.mean()
            loss = loss + story_loss(s_exp, s_imp, target, counts_d, prior, args.tau, args.story)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch.numel()
            seen += batch.numel()
            last_gate = gate.detach()
        scheduler.step()

        seen_mask = counts_d > 0
        val_scores = evaluate(model, tensors, val_t, dev_knn, dev_w, log_freq, prior, args.tau, seen_mask)
        val_m = sliced_metrics(val_scores, val_t["owner"], counts_d)
        val_mrr = val_m["all"]["MRR"]
        val_low = val_m.get("low(<5)", {}).get("MRR", 0.0)
        select = val_mrr + args.low_coef * val_low
        gates = gate_summary(last_gate, counts_d)
        history.append(
            {"epoch": epoch, "loss": total_loss / seen, "val_mrr": val_mrr, "val_low_mrr": val_low, "select": select, "gate": gates}
        )
        print(
            f"epoch {epoch:02d}  loss {total_loss / seen:.4f}  val_MRR {val_mrr:.4f}  "
            f"val_HR@1 {val_m['all']['HR@1']:.4f}  val_NDCG@10 {val_m['all']['NDCG@10']:.4f}  "
            f"gate L/M/H {gates.get('low(<5)', float('nan')):.2f}/"
            f"{gates.get('mid(5-19)', float('nan')):.2f}/"
            f"{gates.get('high(>=20)', float('nan')):.2f}  "
            f"val_low {val_m.get('low(<5)', {}).get('MRR', float('nan')):.3f}  "
            f"{time.time() - t0:.1f}s"
        )
        if select > best_select:
            best_select = select
            best_mrr = val_mrr
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= args.patience:
                print(f"验证 MRR 连续 {args.patience} 轮未提升，停止于 epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("没有得到可用的模型参数")
    model.load_state_dict(best_state)
    test_scores = evaluate(model, tensors, test_t, dev_knn, dev_w, log_freq, prior, args.tau, seen_mask)
    test_raw = evaluate(model, tensors, test_t, dev_knn, dev_w, log_freq, prior, tau=0.0, seen_mask=seen_mask)
    test_m = sliced_metrics(test_scores, test_t["owner"], counts_d)
    test_raw_m = sliced_metrics(test_raw, test_t["owner"], counts_d)

    # 验证集指标随最佳轮次一并落盘：探究的变体/超参裁决一律看验证集，测试集只在最终报告用一次。
    val_scores = evaluate(model, tensors, val_t, dev_knn, dev_w, log_freq, prior, args.tau, seen_mask)
    val_m = sliced_metrics(val_scores, val_t["owner"], counts_d)

    pop = popularity_scores(counts_d, test_t["emb"].size(0))
    cen = centroid_scores(tensors["emb"], tensors["owner"], test_t["emb"], n_dev)
    neginf = torch.full_like(pop, float("-inf"))
    pop = torch.where(seen_mask.view(1, -1), pop, neginf)
    cen = torch.where(seen_mask.view(1, -1), cen, torch.full_like(cen, float("-inf")))
    pop_m = sliced_metrics(pop, test_t["owner"], counts_d)
    cen_m = sliced_metrics(cen, test_t["owner"], counts_d)

    # 最佳轮次的门控用当前参数再算一次，不依赖最后一个训练 batch。
    with torch.no_grad():
        _logits, gate, _w, _s_exp, _s_imp = model.encode_train_batch(
            tensors["emb"],
            tensors["owner"],
            tensors["knn"],
            torch.arange(min(64, n_train), device=device),
            dev_knn,
            dev_w,
            log_freq,
        )
        gates = gate_summary(gate, counts_d)

    print()
    print(format_block(f"Hy-LIN 测试集（按验证 MRR 选择的 epoch {best_epoch}，含频率校正）", test_m))
    print()
    print(format_block("Hy-LIN 测试集（同一模型，解码时不做频率校正）", test_raw_m))
    print()
    print(format_block("对照：训练集频率", pop_m))
    print()
    print(format_block("对照：训练缺陷文本质心", cen_m))
    low_m = test_m.get("low(<5)", {})
    print(
        f"门控均值（明线权重） low {gates.get('low(<5)', float('nan')):.3f}  "
        f"high {gates.get('high(>=20)', float('nan')):.3f}"
    )
    print(
        f"SUMMARY fusion={args.fusion} story={args.story} tau={args.tau} "
        f"MRR={test_m['all']['MRR']:.4f} HR@1={test_m['all']['HR@1']:.4f} "
        f"HR@10={test_m['all']['HR@10']:.4f} N@10={test_m['all']['NDCG@10']:.4f} "
        f"lowMRR={low_m.get('MRR', float('nan')):.4f} lowHR@1={low_m.get('HR@1', float('nan')):.4f} "
        f"lowHR@10={low_m.get('HR@10', float('nan')):.4f} "
        f"gateL={gates.get('low(<5)', float('nan')):.3f} "
        f"gateM={gates.get('mid(5-19)', float('nan')):.3f} "
        f"gateH={gates.get('high(>=20)', float('nan')):.3f}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "data": str(args.data),
        "best_epoch": best_epoch,
        "best_val_mrr": best_mrr,
        "n_train": int(n_train),
        "n_val": int(val_t["emb"].size(0)),
        "n_test": int(test_t["emb"].size(0)),
        "n_dev": n_dev,
        "tau": args.tau,
        "gate_explicit_weight": gates,
        "split": args.split,
        "val": val_m,
        "test": test_m,
        "test_without_prior": test_raw_m,
        "baseline_popularity": pop_m,
        "baseline_text_centroid": cen_m,
        "history": history,
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "cost": {
            "wall_seconds": round(time.time() - started, 1),
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "device": str(device),
            "device_tag": os.environ.get("HXL_DEVICE", ""),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "gpu_max_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1) if device.type == "cuda" else None,
            "gpu_max_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1) if device.type == "cuda" else None,
        },
        "repro": repro_info(),
    }
    out_file = args.out / "metrics.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model": best_state, "best_epoch": best_epoch}, args.out / "best.pt")

    # 落盘逐样本打分与训练频带计数。基线每格都存了 *_test_scores.npy，
    # Hy-LIN 之前只存 metrics.json，导致锁点后无法在同一批 bug 上做
    # Hy-LIN vs 基线的逐案例比较（R1-C14），也无法做容量约束仿真——
    # ���是定性分析唯一的原料。用开关控制是因为 504 格全存会占满磁盘。
    if args.save_scores:
        # 转 CPU 后再转 numpy：test_scores 在 GPU 上，直接 .numpy() 会因设备不匹配报错
        np.save(args.out / "test_scores.npy", test_scores.detach().float().cpu().numpy())
        np.save(args.out / "test_labels.npy", test_t["owner"].detach().cpu().numpy())
        # 频带划分口径见 metrics.py::sliced_metrics（按训练集样本数），没有这个向量
        # 就无法还原"被推荐的开发者属于哪个频带"，事后补不回来
        np.save(args.out / "train_counts.npy", counts_d.detach().cpu().numpy())

    print(f"结果已写入 {out_file}")


if __name__ == "__main__":
    main()
