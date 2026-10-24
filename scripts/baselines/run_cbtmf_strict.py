"""CBT-MF 忠实版 v2（按论文：软件学报 2025 / 谢生龙等）。

论文要点与本文实现的对应：
1. bug 节点初始化为 word2vec 文本向量 p0（论文 §2.2/3.1；w2v 维度原文未给确值，取 300，披露）；
2. 开发者节点初始化为 E-mail 单热向量 q0（论文 §3.1，"用表示其 ID 的 E-mail 初始化为单向量"），
   再由线性层映射到卷积维度；
3. 数据增强（论文 §3.2）：K-means 原型聚类 + 正/负采样 s=5，得到增强分派记录
   t_ij ∈ {3,2,1}（原始/正样本/负样本），增强记录只出现在训练阶段；
4. 二部图卷积（论文 §3.3 式 6-11）：边权重 η_t 加权，消息 ψ = η_t (R1 p + R2 (p ⊙ q))，
   开发者与 bug 两侧交替卷积，L=2 层（论文敏感性实验 1–3 层，2 层最好）；
5. 表示拼接 + 内积打分（式 12）：p* = p0||p1||…||pL，q* = q0||q1||…||qL，t̂ = p*·q*；
6. BPR 损失（式 13-14）：三元组 (正样本 bug i, 开发者 j, 负样本 bug k)，
   L = -logσ(t̂_ij − t̂_kj) + λ||Ξ||²，Adam，batch 8192（论文 §4 参数给出的值）；
7. 冷启动：时序切分下测试 bug 不在训练图中，卷积退化为自环（p^l = p^{l-1}），
   打分 = p0·(Σ_l q^l)，全部使用**训练得到的参数**，不用外部质心（当前 v1 的缺陷已修）。

原文未给确值、这里按披露口径固定的超参：
- w2v 维度 300、图卷积维度 d=128、K-means 簇数 200（原文给出 k%∈{10..50}，口径不明）、
  负样本池取"其他簇的原型 bug"（论文 §3.2 "负采样收集除 bi 以外的所有原型 bug"）。
- 早停：验证集 MRR（与其它基线一致）；测试只跑一次。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost


def tokenize(s: str) -> list[str]:
    return s.lower().split()[:400]


class CBTMF(nn.Module):
    """二部图卷积 + 层拼接内积打分（论文 §3.3）。"""

    def __init__(self, n_bug: int, n_dev: int, p_dim: int, d: int, layers: int, w2v_init: np.ndarray):
        super().__init__()
        self.n_bug, self.n_dev, self.d, self.layers = n_bug, n_dev, d, layers
        # p0: w2v 文本向量（固定初始化，可微调）；q0: E-mail 单热 → 线性映射
        self.p0 = nn.Parameter(torch.from_numpy(w2v_init), requires_grad=True)
        self.dev_onehot = torch.eye(n_dev)
        self.proj_dev = nn.Linear(n_dev, d)
        self.proj_bug = nn.Linear(p_dim, d)
        # 每层两个方向各两条消息分支：R1/R2 走 bug→dev，R3/R4 走 dev→bug
        self.r_blocks = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(4 * layers)])
        self.w_dev = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.w_bug = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, edges_src, edges_dst, edge_w, dev_onehot):
        """edges_src: bug 下标（0..n_bug-1）在 train 局部编号；edges_dst: dev 下标。"""
        p = self.proj_bug(self.p0)                 # [n_bug, d]
        q = self.proj_dev(dev_onehot)              # [n_dev, d]
        ps, qs = [p], [q]
        dev = p.device
        src = edges_src.to(dev)
        dst = edges_dst.to(dev)
        w = edge_w.to(dev).unsqueeze(-1)           # η_t
        for l in range(self.layers):
            # bug → dev：m = η (R1 p_b + R2 (p_b ⊙ q_d))
            pb = p[src]
            qd = q[dst]
            msg_d = w * (self.r_blocks[2 * l](pb) + self.r_blocks[2 * l + 1](pb * qd))
            agg_d = torch.zeros_like(q)
            cnt_d = torch.zeros(self.n_dev, 1, device=dev, dtype=q.dtype)
            agg_d.index_add_(0, dst, msg_d)
            cnt_d.index_add_(0, dst, w)
            q_new = self.w_dev[l](agg_d / cnt_d.clamp(min=1.0) + q)
            # dev → bug：m = η (R3 q_d + R4 (p_b ⊙ q_d))（用更新后的 q）
            pb2 = p[src]
            qd2 = q_new[dst]
            msg_b = w * (self.r_blocks[2 * (self.layers + l)](qd2) + self.r_blocks[2 * (self.layers + l) + 1](pb2 * qd2))
            agg_b = torch.zeros_like(p)
            cnt_b = torch.zeros(self.n_bug, 1, device=dev, dtype=p.dtype)
            agg_b.index_add_(0, src, msg_b)
            cnt_b.index_add_(0, src, w)
            p = self.w_bug[l](agg_b / cnt_b.clamp(min=1.0) + p)
            q = q_new
            ps.append(p)
            qs.append(q)
        p_star = torch.cat(ps, dim=-1)
        q_star = torch.cat(qs, dim=-1)
        return p_star, q_star


def build_augmented(tr, owners_tr, n_dev, seed, s=5, k_clusters=200, dim=300):
    """K-means 原型聚类 + 正负采样，返回增强 (bug, dev) 三元组与增强边。"""
    from gensim.models import Word2Vec
    from sklearn.cluster import MiniBatchKMeans

    texts_tok = tr["tokens"]
    w2v = Word2Vec(sentences=texts_tok, vector_size=dim, window=5, min_count=3, sg=1,
                   workers=8, epochs=5, seed=seed)
    def bvec(toks):
        vs = [w2v.wv[w] for w in toks if w in w2v.wv]
        return np.mean(vs, 0) if vs else np.zeros(dim, np.float32)
    Bv = np.stack([bvec(t) for t in texts_tok]).astype(np.float32)
    k = max(2, min(k_clusters, len(Bv) // 50))
    clu = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=seed, batch_size=4096).fit_predict(Bv)
    rng = np.random.default_rng(seed)
    pos_pairs = []  # (bug_i, dev_y) 正样本（含原始）
    neg_triples = []  # (bug_i, dev_y, neg_bug_k)
    for r in range(len(Bv)):
        same = np.flatnonzero((clu == clu[r]) & (owners_tr == owners_tr[r]) & (np.arange(len(Bv)) != r))
        if len(same):
            take = rng.choice(same, size=min(s, len(same)), replace=False)
            for t in take:
                pos_pairs.append((int(t), int(owners_tr[r])))
        # 负样本池 = 其他簇的文件（"原型 bug"）
        other = np.flatnonzero(clu != clu[r])
        if len(other) >= s:
            negs = rng.choice(other, size=s, replace=False)
        else:
            negs = other
        for ng in negs:
            if owners_tr[ng] != owners_tr[r]:
                neg_triples.append((r, int(owners_tr[r]), int(ng)))
    return w2v, Bv, clu, np.asarray(pos_pairs, dtype=np.int64), np.asarray(neg_triples, dtype=np.int64)


def main(dataset: str, seed: int, epochs: int = 60, bs: int = 8192, d: int = 128,
         layers: int = 2, lam: float = 1e-4, s: int = 5, w2v_dim: int = 300, patience: int = 10, lr: float = 2e-3):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    np.random.seed(seed)
    owners, texts, _, tr_idx, va_idx, te_idx = load_split(dataset, seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dev = int(owners.max()) + 1
    n_bug = len(tr_idx)
    owners_tr = owners[tr_idx]
    tr_pack = {"tokens": [tokenize(texts[i]) for i in tr_idx]}
    w2v, Bv, clu, pos_pairs, neg_triples = build_augmented(
        tr_pack, owners_tr, n_dev, seed, s=s, k_clusters=200, dim=w2v_dim)
    p0 = Bv / (np.linalg.norm(Bv, axis=1, keepdims=True) + 1e-8)

    # 边：原始 t=3（η=3）+ 增强正样本 t=2（η=2）
    src_raw = np.arange(n_bug)
    dst_raw = owners_tr
    src_aug = pos_pairs[:, 0]
    dst_aug = pos_pairs[:, 1]
    edges_src = torch.from_numpy(np.concatenate([src_raw, src_aug]).astype(np.int64))
    edges_dst = torch.from_numpy(np.concatenate([dst_raw, dst_aug]).astype(np.int64))
    edge_w = torch.from_numpy(np.concatenate([np.full(n_bug, 3.0), np.full(len(src_aug), 2.0)]).astype(np.float32))

    dev_onehot = torch.eye(n_dev, device=dev)
    model = CBTMF(n_bug, n_dev, p0.shape[1], d, layers, p0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def score_bugs(p_star, q_star):
        return p_star @ q_star.T

    def split_vec(idx):
        toks = [tokenize(texts[i]) for i in idx]
        vec = np.stack([np.mean([w2v.wv[w] for w in t if w in w2v.wv], 0) if any(w in w2v.wv for w in t)
                        else np.zeros(w2v_dim, np.float32) for t in toks]).astype(np.float32)
        vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
        return vec

    val_vec = torch.from_numpy(split_vec(va_idx)).to(dev)
    test_vec = torch.from_numpy(split_vec(te_idx)).to(dev)

    def eval_split(vec):
        # 查询 bug 无图边：卷积退化 → p^l = p^{l-1}；等价于 p0 与 Σ_l q^l 内积
        p0q = model.proj_bug(vec)
        p_star_q = torch.cat([p0q] * (layers + 1), dim=-1)
        _, q_star = model(edges_src, edges_dst, edge_w, dev_onehot)
        return (p_star_q @ q_star.T).detach().cpu().numpy()

    best_val, best_state, stall = -1.0, None, 0
    from hy_lin.metrics import ranking_metrics
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(neg_triples))
        for st in range(0, len(neg_triples), bs):
            b = torch.from_numpy(neg_triples[perm[st:st + bs]]).to(dev)
            p_star, q_star = model(edges_src, edges_dst, edge_w, dev_onehot)
            pos_b, d_j, neg_b = b[:, 0], b[:, 1], b[:, 2]
            t_pos = (p_star[pos_b] * q_star[d_j]).sum(-1)
            t_neg = (p_star[neg_b] * q_star[d_j]).sum(-1)
            loss = -F.logsigmoid(t_pos - t_neg).mean() + lam * sum(p.pow(2).sum() for p in model.parameters())
            opt.zero_grad(); loss.backward(); opt.step()
        sve = eval_split(val_vec)
        mrr = ranking_metrics(torch.from_numpy(sve), torch.from_numpy(owners[va_idx]))["MRR"]
        print(f"[cbtmf {dataset} s{seed}] ep{ep} loss={float(loss):.4f} valMRR={mrr:.4f}", flush=True)
        if mrr > best_val + 1e-5:
            best_val, stall = mrr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ste = eval_split(test_vec)
        sve = eval_split(val_vec)
    torch.save({"state": model.state_dict()}, model_path("cbtmf", dataset, seed))
    save_scores("cbtmf", dataset, seed, ste, owners[te_idx], sve, owners[va_idx],
                {"strict_v2": True, "layers": layers, "d": d, "w2v_dim": w2v_dim, "s": s,
                 "lam": lam, "bs": bs, "dev_init": "onehot-email",
                 "note": "paper-faithful: bipartite GCN + interaction msg + concat + BPR triplet"},
                cost=collect_cost(t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("seed", type=int)
    a = ap.parse_args()
    main(a.dataset, a.seed)
