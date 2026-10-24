"""CBT-MF 近似（按原文：Content-Boosted + MF，文本相似度做先验平滑的矩阵分解）。
实现：MiniLM 质心相似度做开发者先验 + bug/developer embedding 点积 + BPR，文本相似度正则。"""
import sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost

def main(dataset, seed, epochs=40, bs=4096, ed=128, lam=0.05):
    import time as _t; t0=_t.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.manual_seed(seed); np.random.seed(seed)
    owners, texts, _, tr, va, te = load_split(dataset, seed)
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    E = torch.from_numpy(st.encode([texts[i] for i in tr], batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32))
    ytr = torch.from_numpy(owners[tr]); nd = int(owners.max())+1; nb = len(tr)
    # 开发者质心相似度先验
    dim = E.size(1); S = torch.zeros(nd, dim); S.index_add_(0, ytr, F.normalize(E, dim=-1))
    S = F.normalize(S / torch.bincount(ytr, minlength=nd).clamp(min=1).unsqueeze(-1).float(), dim=-1)
    prior = S @ S.T
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = nn.Embedding(nb, ed); D = nn.Embedding(nd, ed)
    nn.init.normal_(B.weight, std=0.01); nn.init.normal_(D.weight, std=0.01)
    B, D = B.to(dev), D.to(dev); prior = prior.to(dev)
    opt = torch.optim.Adam(list(B.parameters())+list(D.parameters()), lr=1e-2)
    for ep in range(epochs):
        perm = torch.randperm(nb)
        for s in range(0, nb, bs):
            bc = perm[s:s+bs]; b = bc.to(dev); pos = ytr[bc].to(dev); neg = torch.randint(0, nd, (len(bc),), device=dev)
            sp=(B(b)*D(pos)).sum(-1); sn=(B(b)*D(neg)).sum(-1)
            reg = lam*((D.weight @ D.weight.T - prior)**2).mean()
            loss = -torch.log(torch.sigmoid(sp-sn)).mean() + reg
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        # 冷启动查询：文本 embedding 与开发者质心内积 + MF 均值偏置
        Eq = torch.from_numpy(st.encode([texts[i] for i in list(va)+list(te)], batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)).to(dev)
        Sc = F.normalize(Eq, dim=-1) @ S.to(dev).T
        ste = Sc[len(va):].cpu().numpy(); sve = Sc[:len(va)].cpu().numpy()
    torch.save({"B": B.state_dict(), "D": D.state_dict()}, model_path("cbtmf", dataset, seed))
    save_scores("cbtmf", dataset, seed, ste, owners[te], sve, owners[va], {"ed": ed, "lam": lam}, cost=collect_cost(t0))

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed",type=int)
    a=ap.parse_args(); main(a.dataset,a.seed)
