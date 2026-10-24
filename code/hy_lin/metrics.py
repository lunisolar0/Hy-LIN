"""单正例排序指标：HR、NDCG @ k，以及 MRR。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


KS = (1, 3, 5, 10)


def ranking_metrics(scores: torch.Tensor, labels: torch.Tensor, ks=KS) -> dict[str, float]:
    """scores: [N, D]，越大越靠前。labels: [N]，每条查询恰好一个正确开发者。

    并列名次按固定种子(12345)在并列段内均匀随机分位，期望等于平均名次；
    实测并列只占查询的 0.00–0.23%，对指标无可感影响。
    单正例时 IDCG = 1，因此 NDCG@k = 1/log2(rank+1)（rank 落在 k 以内）。
    """
    if scores.ndim != 2:
        raise ValueError("scores 应为 [N, D]")
    labels = labels.long().view(-1)
    true = scores.gather(1, labels.view(-1, 1))
    n_better = (scores > true).sum(dim=1)
    n_tie = (scores == true).sum(dim=1) - 1
    gen = torch.Generator(device=scores.device).manual_seed(12345)
    rank = (n_better + torch.floor(torch.rand(labels.numel(), device=scores.device, generator=gen) * (n_tie + 1)).long()).float() + 1.0
    out = {"MRR": (1.0 / rank).mean().item(), "n": int(labels.numel())}
    for k in ks:
        hit = (rank <= k).float()
        out[f"HR@{k}"] = hit.mean().item()
        gain = torch.where(rank <= k, 1.0 / torch.log2(rank + 1.0), torch.zeros_like(rank))
        out[f"NDCG@{k}"] = gain.mean().item()
    return out


def sliced_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    train_counts: torch.Tensor,
    ks=KS,
) -> dict[str, dict]:
    """按正确负责人在训练集中的样本数切分。低频 <5，高频 >=20。"""
    counts = train_counts[labels.long()]
    n_raw = int(labels.numel())
    seen_q = counts > 0
    n_eval = int(seen_q.sum())
    labels, scores, counts = labels[seen_q], scores[seen_q], counts[seen_q]
    bands = {
        "all": torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device),
        "low(<5)": counts < 5,
        "mid(5-19)": (counts >= 5) & (counts < 20),
        "high(>=20)": counts >= 20,
    }
    out = {}
    for name, mask in bands.items():
        if int(mask.sum()) == 0:
            continue
        out[name] = ranking_metrics(scores[mask], labels[mask], ks)
    if "all" in out:
        out["all"]["n_raw"] = n_raw
        out["all"]["n_eval"] = n_eval
    return out


def popularity_scores(train_counts: torch.Tensor, n_query: int) -> torch.Tensor:
    return torch.log1p(train_counts).view(1, -1).expand(n_query, -1)


def centroid_scores(
    train_emb: torch.Tensor,
    train_owner: torch.Tensor,
    query_emb: torch.Tensor,
    n_dev: int,
) -> torch.Tensor:
    """每个开发者取其训练缺陷文本的均值方向，查询与该方向做内积。"""
    dim = train_emb.size(1)
    summed = torch.zeros(n_dev, dim, device=train_emb.device, dtype=train_emb.dtype)
    summed.index_add_(0, train_owner, F.normalize(train_emb, dim=-1))
    counts = torch.bincount(train_owner, minlength=n_dev).clamp(min=1).unsqueeze(-1).float()
    centroids = F.normalize(summed / counts, dim=-1)
    queries = F.normalize(query_emb, dim=-1)
    return queries @ centroids.T


def format_block(title: str, blocks: dict[str, dict], ks=KS) -> str:
    lines = [title]
    head = f"{'slice':<12}{'n':>8}{'MRR':>8}"
    for k in ks:
        head += f"{'HR@'+str(k):>8}{'N@'+str(k):>8}"
    lines.append(head)
    for name, m in blocks.items():
        row = f"{name:<12}{m['n']:>8}{m['MRR']:>8.4f}"
        for k in ks:
            row += f"{m[f'HR@{k}']:>8.4f}{m[f'NDCG@{k}']:>8.4f}"
        lines.append(row)
    return "\n".join(lines)


def _self_check() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.2], [5.0, 0.0, 1.0]])
    labels = torch.tensor([1, 0])
    perfect = ranking_metrics(scores, labels, ks=(1, 3))
    assert abs(perfect["HR@1"] - 1.0) < 1e-6
    assert abs(perfect["MRR"] - 1.0) < 1e-6
    assert abs(perfect["NDCG@1"] - 1.0) < 1e-6

    second = ranking_metrics(torch.tensor([[0.1, 0.9, 0.2]]), torch.tensor([2]), ks=(1, 3))
    assert abs(second["HR@1"]) < 1e-6
    assert abs(second["HR@3"] - 1.0) < 1e-6
    assert abs(second["MRR"] - 0.5) < 1e-6
    assert abs(second["NDCG@3"] - (1.0 / torch.log2(torch.tensor(3.0)).item())) < 1e-5
