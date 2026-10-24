"""SCL-BT 忠实版 v2（按论文：Automated Software Engineering 2025, 32:51）。

论文结构 → 本实现：
1. 语义初始化（§3.2.1）：p0 = W(BiRNN(W2V))。论文明确说明编码器"可被 BERT 等替换"，
   本文统一预算下用**冻结 BERT CLS** 特征 + 线性 W（披露）；开发者 q0 = onehot(E-mail) 线性映射（§3.2.2）。
2. 二部图卷积骨干（§4.1.2）：bug↔developer 交替卷积，L=2 层。
3. 自适应加边视图（§4.2.1 式 10-12）：G1、G2 = 原图 + 按比例 α_ae 随机增加异质边；
   论文的 α_ae = β|T|/(|B||D|) 未给 β 与"条数"的确切口径，本实现取 β=0.05（加边数 = 5%|T|），披露。
4. 双视图 InfoNCE（§4.2.2 式 14-20）：同一节点在两视图上的表示互为正对，τ=0.5。
5. 超图同质视图（§4.2.1 (2) 式 13/21）：二阶同质邻居的均值表示，与骨干表示做 InfoNCE。
6. 多任务输出（§4.2 开头）：链接预测（内积 + BCE）+ 开发者分类（CE）联合训练。
7. 打分：测试 bug 无图边 → 链接预测分 + 分类分相加（论文"multiple task learning"的线性组合口径，披露）。

原文未给确值/口径不明处（披露）：加边比例 β=0.05、对比温度 τ=0.5（论文给定）、
GCN 维度 d=128、batch 1024、lr 1e-3、BMI 早停按验证集 MRR。
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


class BipartiteGCN(nn.Module):
    def __init__(self, p_dim: int, n_dev: int, d: int, layers: int):
        super().__init__()
        self.layers = layers
        self.proj_bug = nn.Linear(p_dim, d)
        self.proj_dev = nn.Linear(n_dev, d)
        self.w_b = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.w_d = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])

    def forward(self, p0, q0, src, dst, n_bug, n_dev):
        p, q = p0, q0
        for l in range(self.layers):
            # bug → dev（均值聚合）
            agg = torch.zeros_like(q).index_add_(0, dst, p[src])
            cnt = torch.zeros(n_dev, 1, device=q.device, dtype=q.dtype).index_add_(0, dst, torch.ones(len(src), 1, device=q.device, dtype=q.dtype))
            q = F.normalize(self.w_d[l](agg / cnt.clamp(min=1)) + q, dim=-1)
            # dev → bug
            agg = torch.zeros_like(p).index_add_(0, src, q[dst])
            cnt = torch.zeros(n_bug, 1, device=p.device, dtype=p.dtype).index_add_(0, src, torch.ones(len(src), 1, device=p.device, dtype=p.dtype))
            p = F.normalize(self.w_b[l](agg / cnt.clamp(min=1)) + p, dim=-1)
        return p, q


def info_nce(z1, z2, tau=0.5):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau
    labels = torch.arange(len(z1), device=z1.device)
    return F.cross_entropy(logits, labels)


def main(dataset: str, seed: int, epochs: int = 60, bs: int = 1024, d: int = 128,
         layers: int = 2, lr: float = 1e-3, tau: float = 0.5, add_ratio: float = 0.05,
         lam_cls: float = 1.0, lam_c: float = 0.05, patience: int = 10):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    np.random.seed(seed)
    owners, texts, _, tr_idx, va_idx, te_idx = load_split(dataset, seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dev = int(owners.max()) + 1
    n_bug = len(tr_idx)
    owners_tr = owners[tr_idx]

    # 语义初始化：冻结 BERT CLS（与其它 PLM 类基线统一预算）
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    bert = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True).to(dev).eval()
    for prm in bert.parameters():
        prm.requires_grad = False

    @torch.no_grad()
    def encode(idx, bsz=64):
        outs = []
        for s in range(0, len(idx), bsz):
            b = tk([texts[i] for i in idx[s:s + bsz]], padding=True, truncation=True,
                   max_length=256, return_tensors="pt").to(dev)
            outs.append(bert(**b).last_hidden_state[:, 0, :].cpu())
        return torch.cat(outs)

    print("[sclbt] BERT 编码训练/验证/测试文本 …", flush=True)
    E_tr = encode(tr_idx).to(dev)
    E_va = encode(va_idx).to(dev)
    E_te = encode(te_idx).to(dev)
    del bert
    torch.cuda.empty_cache()

    src0 = torch.arange(n_bug, device=dev)
    dst0 = torch.from_numpy(owners_tr.astype(np.int64)).to(dev)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    def augment(src, dst):
        n_add = int(round(add_ratio * len(src)))
        s_add = torch.randint(0, n_bug, (n_add,), generator=rng)
        d_add = torch.randint(0, n_dev, (n_add,), generator=rng)
        return torch.cat([src, s_add.to(dev)]), torch.cat([dst, d_add.to(dev)])

    src1, dst1 = augment(src0, dst0)
    src2, dst2 = augment(src0, dst0)

    model = BipartiteGCN(E_tr.size(1), n_dev, d, layers).to(dev)
    clf = nn.Linear(d, n_dev).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=lr)
    eye_dev = torch.eye(n_dev, device=dev)

    def init_reps():
        # onehot(Email) → q0 等价于权重转置；p0 由冻结 BERT CLS 线性投影
        q0 = model.proj_dev.weight.t() + model.proj_dev.bias
        p0 = F.normalize(model.proj_bug(E_tr), dim=-1)
        return p0, q0

    # 二阶同质邻接（超图视图用，train-only）
    dev_dev = torch.zeros(n_dev, n_dev, device=dev)
    dev_dev.index_put_((dst0, dst0), torch.ones(n_bug, device=dev), accumulate=True)
    dev_dev.fill_diagonal_(0)
    row = dev_dev / dev_dev.sum(1, keepdim=True).clamp(min=1)

    from hy_lin.metrics import ranking_metrics

    def forward_all():
        p0_all, q0 = init_reps()
        p1, q1 = model(p0_all, q0, src1, dst1, n_bug, n_dev)
        p2, q2 = model(p0_all, q0, src2, dst2, n_bug, n_dev)
        pg, qg = model(p0_all, q0, src0, dst0, n_bug, n_dev)
        return p1, q1, p2, q2, pg, qg

    def eval_split(E):
        with torch.no_grad():
            p0_all, q0 = init_reps()
            _, qg = model(p0_all, q0, src0, dst0, n_bug, n_dev)
            pg = F.normalize(model.proj_bug(E), dim=-1)
            return (pg @ qg.T + clf(pg)).cpu().numpy()

    best_val, best_state, stall = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm_b = torch.randperm(n_bug)
        tot, nb = 0.0, 0
        for st in range(0, n_bug, bs):
            bi = perm_b[st:st + bs].to(dev)
            # 开发者节点少（千级），每步随机采一批，避免尾部空 batch 让 InfoNCE 变成 NaN
            di = torch.randint(0, n_dev, (min(bs, n_dev),), device=dev)
            p1, q1, p2, q2, pg, qg = forward_all()
            # ①双视图 InfoNCE（异质结构）
            l_he = info_nce(p1[bi], p2[bi], tau) + info_nce(q1[di], q2[di], tau)
            # ②超图同质对比（开发者：骨干表示 vs 二阶邻域均值）
            q_h = row[di] @ qg.detach()
            l_hom = info_nce(qg[di], q_h, tau)
            # ③多任务：链接预测 BCE（正边 + 随机负边）
            pos = (pg[src0] * qg[dst0]).sum(-1)
            neg_s = torch.randint(0, n_bug, (4096,), device=dev)
            neg_d = torch.randint(0, n_dev, (4096,), device=dev)
            neg = (pg[neg_s] * qg[neg_d]).sum(-1)
            l_link = F.binary_cross_entropy_with_logits(
                torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
            # ④开发者分类
            l_cls = F.cross_entropy(clf(pg[bi]), torch.from_numpy(owners_tr).to(dev)[bi])
            loss = l_link + lam_cls * l_cls + lam_c * (l_he + l_hom)
            if not torch.isfinite(loss):
                print(f"[sclbt] ep{ep} 非有限损失：link={l_link} cls={l_cls} he={l_he} hom={l_hom}", flush=True)
                raise SystemExit(3)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(clf.parameters()), 5.0)
            opt.step()
            tot += float(loss) * len(bi); nb += len(bi)
        sve = eval_split(E_va)
        mrr = ranking_metrics(torch.from_numpy(sve), torch.from_numpy(owners[va_idx]))["MRR"]
        print(f"[sclbt {dataset} s{seed}] ep{ep} loss={tot / max(1, nb):.4f} valMRR={mrr:.4f}", flush=True)
        if mrr > best_val + 1e-5:
            best_val, stall = mrr, 0
            best_state = {k: v.detach().clone() for k, v in list(model.state_dict().items()) + list(clf.state_dict().items())}
        else:
            stall += 1
            if stall >= patience:
                break
    model.load_state_dict({k: best_state[k] for k in model.state_dict()})
    clf.load_state_dict({k: best_state[k] for k in clf.state_dict()})
    model.eval(); clf.eval()
    ste = eval_split(E_te)
    sve = eval_split(E_va)
    torch.save({"model": model.state_dict(), "clf": clf.state_dict()}, model_path("sclbt", dataset, seed))
    save_scores("sclbt", dataset, seed, ste, owners[te_idx], sve, owners[va_idx],
                {"strict_v2": True, "encoder": "bert-frozen-cls", "layers": layers, "d": d,
                 "tau": tau, "add_ratio": add_ratio, "lam_cls": lam_cls, "lam_c": lam_c,
                 "note": "paper-faithful: bipartite GCN + 2-view InfoNCE + hypergraph homophily + multitask"},
                cost=collect_cost(t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("seed", type=int)
    a = ap.parse_args()
    main(a.dataset, a.seed)
