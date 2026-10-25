"""读取处理后的缺陷数据，并建好训练图用的近邻与划分。

初始交互图的真实边只来自训练集里的「缺陷 → 负责人」。
阀门候选是训练缺陷之间的文本 top-k，查询缺陷只连向训练缺陷。
开发者同质图由训练缺陷文本的质心相似度连边，与阀门分开构建。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TEXT_CHARS = 1500
ENCODERS = {
    "minilm": {"kind": "st", "name": "sentence-transformers/all-MiniLM-L6-v2", "batch": 128},
    "bert": {"kind": "hf", "name": "bert-base-uncased", "batch": 32},
    "roberta": {"kind": "hf", "name": "FacebookAI/roberta-base", "batch": 32},
}


def load_records(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    owners, texts = [], []
    for row in rows:
        owners.append(row["owner"])
        title = (row.get("issue_title") or "").strip()
        desc = (row.get("description") or "").strip()[:TEXT_CHARS]
        texts.append(f"{title}\n{desc}" if desc else title)
    return owners, texts


def chronological_split(
    n_records: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """时序切分：按记录在交付文件中的先后顺序做位置切分。

    对齐前人（TriagerX / SCL-BT / CBT-MF）在 DeepTriage benchmark 上的
    chronological split 口径——benchmark 按报告时间（bug id 单调）交付，
    故交付顺序即时间顺序：前 80% 训练、中 10% 验证、后 10% 测试。
    不分层、不补齐：新开发者可能只出现在验证/测试里，这正是要暴露的冷启动。
    """
    idx = np.arange(n_records, dtype=np.int64)
    n_test = int(round(n_records * test_ratio))
    n_val = int(round(n_records * val_ratio))
    n_train = n_records - n_val - n_test
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def build_chronological_split(
    owners: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """时序切分并把候选开发者限定为训练集出现过的负责人。

    返回 (owners_new, train_idx, val_idx, test_idx)。owners_new 把负责人重映射到
    0..n_dev-1，只在训练集出现的负责人才有编号；只出现在验证/测试里的负责人
    （冷启动）对应记录被丢弃，保证候选集与训练集一致——这正是前人“活跃开发者”
    过滤的等价做法，也让各方法输出维度对齐。
    """
    idx = np.arange(len(owners), dtype=np.int64)
    tr, va, te = chronological_split(len(owners), val_ratio, test_ratio)
    train_devs = np.unique(owners[tr])
    pos = np.searchsorted(train_devs, owners)
    pos_c = np.clip(pos, 0, len(train_devs) - 1)
    in_train = train_devs[pos_c] == owners
    owners_new = np.where(in_train, pos, -1).astype(np.int64)
    va = va[in_train[va]]
    te = te[in_train[te]]
    return owners_new, tr, va, te


def _topk_knn(src: torch.Tensor, dst: torch.Tensor, k: int, exclude_self: bool) -> torch.Tensor:
    """src 在 dst 中的 top-k 下标。向量应已做 L2 归一化。"""
    n_src = src.size(0)
    k_eff = k + 1 if exclude_self else k
    if k_eff > dst.size(0):
        raise ValueError(f"近邻数 {k_eff} 超过候选节点数 {dst.size(0)}")
    idx_out = torch.empty(n_src, k, dtype=torch.long, device=src.device)
    step = 2048
    for start in range(0, n_src, step):
        end = min(start + step, n_src)
        sim = src[start:end] @ dst.T
        if exclude_self:
            rows = torch.arange(end - start, device=src.device)
            sim[rows, start + rows] = -1e9
        idx_out[start:end] = torch.topk(sim, k, dim=-1).indices
    return idx_out


def _encode_texts(texts: list[str], device: torch.device, encoder_key: str) -> np.ndarray:
    spec = ENCODERS[encoder_key]
    if spec["kind"] == "st":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(spec["name"], device=str(device), local_files_only=True)
        encoder.max_seq_length = 256
        emb = encoder.encode(
            texts,
            batch_size=spec["batch"],
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        del encoder
    else:
        emb = _encode_hf(texts, device, spec["name"], spec["batch"])
        emb = _remove_top_component(emb, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(emb, dtype=np.float32)


def _remove_top_component(emb: np.ndarray, device: torch.device) -> np.ndarray:
    """原始 BERT / RoBERTa 向量方向很集中，去掉第一主成分后近邻才分得开。"""
    matrix = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(matrix, q=1)
    matrix = matrix - (matrix @ components) @ components.T
    return F.normalize(matrix, dim=-1).cpu().numpy()


def _encode_hf(texts: list[str], device: torch.device, name: str, batch_size: int) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModel.from_pretrained(name, local_files_only=True).to(device)
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pieces.append(F.normalize(pooled, dim=-1).cpu())
            if start % (batch_size * 50) == 0:
                print(f"  编码进度 {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    del model
    return torch.cat(pieces, dim=0).numpy()


def centroid_dev_graph(summed: torch.Tensor, counts: torch.Tensor, k_dev: int, temperature: float = 0.2):
    """由缺陷向量之和与条数构造开发者中心图。

    counts 为 0 的开发者没有中心：不作为任何人的邻居，自身的边权也置 0。
    全量训练集上没有这种开发者，结果与一次性构图相同。
    """
    counts = counts.to(device=summed.device, dtype=summed.dtype)
    centroids = F.normalize(summed / counts.clamp(min=1).unsqueeze(-1), dim=-1)
    has = counts > 0
    sim = centroids @ centroids.T
    sim.fill_diagonal_(-1e9)
    empty = ~has
    if bool(empty.any()):
        sim = sim.clone()
        sim[empty, :] = -1e9
        sim[:, empty] = -1e9
        sim.fill_diagonal_(-1e9)
    values, index = torch.topk(sim, k_dev, dim=-1)
    weight = torch.softmax(values / temperature, dim=-1)
    if bool(empty.any()):
        weight = weight.clone()
        weight[empty] = 0
    return index, weight


def _dev_graph_overlap(train_knn: torch.Tensor, train_owner: torch.Tensor, n_dev: int, k_dev: int, device: torch.device):
    """两个开发者若彼此的训练缺陷落在文本近邻里，就连一条同质边。"""
    src = train_owner.unsqueeze(1).expand_as(train_knn).reshape(-1).to(device)
    dst = train_owner.to(device)[train_knn.reshape(-1).to(device)]
    keep = src != dst
    flat = src[keep] * n_dev + dst[keep]
    acc = torch.bincount(flat, minlength=n_dev * n_dev).float().view(n_dev, n_dev)
    acc.fill_diagonal_(0)
    values, index = torch.topk(acc, k_dev, dim=-1)
    weight = values.clamp(min=0)
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    empty = values.sum(dim=-1) <= 0
    if bool(empty.any()):
        weight[empty] = 1.0 / k_dev
    return index.cpu(), weight.cpu()


def prepare(
    data_path: Path,
    cache_path: Path,
    seed: int,
    k_bug: int,
    k_dev: int,
    device: torch.device,
    encoder_key: str = "minilm",
    dev_graph: str = "centroid",
    split: str = "chronological",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    if encoder_key not in ENCODERS:
        raise ValueError(f"未知编码器 {encoder_key}")
    if dev_graph not in {"centroid", "overlap"}:
        raise ValueError(f"未知同质图 {dev_graph}")
    if split != "chronological":
        raise ValueError(f"本项目只支持时序位置切分，收到 {split!r}；随机分层已作废（实验设置 §1.4）")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "data": str(data_path.resolve()),
        "mtime": data_path.stat().st_mtime,
        "seed": seed,
        "k_bug": k_bug,
        "k_dev": k_dev,
        "encoder": encoder_key,
        "dev_graph": dev_graph,
        "split": split,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "version": 3,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("signature") == signature:
            print(f"使用缓存 {cache_path}")
            return cached
        print("缓存与当前设置不一致，重新构建")

    print(f"读取 {data_path}")
    owner_names, texts = load_records(data_path)
    dev_names_all = sorted(set(owner_names))
    name_to_id_all = {name: i for i, name in enumerate(dev_names_all)}
    owners_all = np.asarray([name_to_id_all[name] for name in owner_names], dtype=np.int64)
    if split == "chronological":
        owners, train_idx, val_idx, test_idx = build_chronological_split(
            owners_all, val_ratio, test_ratio
        )
        n_dev = int(owners.max()) + 1 if owners.size else 0
        n_drop = len(dev_names_all) - n_dev
        dev_names = [dev_names_all[d] for d in np.unique(owners_all[train_idx])]
        print(
            f"时序切分 80/10/10：train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}，"
            f"候选负责人 {n_dev}（丢弃冷启动负责人 {n_drop} 个）"
        )
    emb_cache = cache_path.parent / f"emb_{data_path.stem}_{encoder_key}.pt"
    emb_signature = {"data": signature["data"], "mtime": signature["mtime"], "encoder": encoder_key, "version": 2}
    if emb_cache.exists():
        emb_payload = torch.load(emb_cache, map_location="cpu", weights_only=False)
        if emb_payload.get("signature") == emb_signature:
            print(f"使用文本缓存 {emb_cache}")
            emb = emb_payload["emb"]
        else:
            emb = None
    else:
        emb = None
    if emb is None:
        print(f"编码文本：{len(texts)} 条，编码器 {encoder_key}")
        emb = torch.from_numpy(_encode_texts(texts, device, encoder_key))
        torch.save({"signature": emb_signature, "emb": emb}, emb_cache)
        print(f"文本缓存已写入 {emb_cache}")
    train_emb = F.normalize(emb[train_idx], dim=-1).to(device)
    print("构建缺陷阀门候选与开发者同质图")
    train_knn = _topk_knn(train_emb, train_emb, k_bug, exclude_self=True).cpu()
    val_knn = _topk_knn(F.normalize(emb[val_idx], dim=-1).to(device), train_emb, k_bug, exclude_self=False).cpu()
    test_knn = _topk_knn(F.normalize(emb[test_idx], dim=-1).to(device), train_emb, k_bug, exclude_self=False).cpu()

    train_owner = torch.from_numpy(owners[train_idx].astype(np.int64))
    counts = torch.bincount(train_owner, minlength=n_dev).float()
    dim = train_emb.size(1)
    summed = torch.zeros(n_dev, dim, device=device)
    summed.index_add_(0, train_owner.to(device), train_emb)
    if dev_graph == "overlap":
        dev_knn, dev_w = _dev_graph_overlap(train_knn, train_owner, n_dev, k_dev, device)
    else:
        dev_knn, dev_w = centroid_dev_graph(summed, counts.to(device), k_dev)
        dev_knn, dev_w = dev_knn.cpu(), dev_w.cpu()

    payload = {
        "signature": signature,
        "n_dev": n_dev,
        "emb": emb,
        "owners": torch.from_numpy(owners),
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "test_idx": torch.from_numpy(test_idx),
        "train_knn": train_knn,
        "val_knn": val_knn,
        "test_knn": test_knn,
        "dev_knn": dev_knn,
        "dev_w": dev_w,
        "train_counts": counts.cpu(),
    }
    torch.save(payload, cache_path)
    print(f"缓存已写入 {cache_path}")
    return payload


def split_views(bundle: dict, split: str) -> dict[str, torch.Tensor]:
    """取出某个划分的缺陷表示。近邻下标指向训练集局部下标。"""
    idx = bundle[f"{split}_idx"]
    owners = bundle["owners"][idx]
    emb = bundle["emb"][idx]
    knn_key = "train_knn" if split == "train" else f"{split}_knn"
    return {"emb": emb, "owner": owners, "knn": bundle[knn_key]}
