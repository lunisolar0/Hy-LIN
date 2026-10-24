"""SCL-BT 近似（按原文：Supervised Contrastive Learning for Bug Triage）。
实现：MiniLM 文本 + 同负责人为正例的 SupCon 预训练一轮 + 线性分类头微调。"""
import sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost

def main(dataset, seed, epochs=12, bs=512, tau=0.07):
    import time as _t; t0=_t.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.manual_seed(seed); np.random.seed(seed)
    owners, texts, _, tr, va, te = load_split(dataset, seed)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    E = torch.from_numpy(st.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 128)).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    Xtr = E[tr].to(dev); ytr = torch.from_numpy(owners[tr]).to(dev)
    for ep in range(3):  # SupCon 预训练 3 epoch
        proj.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), bs):
            b = perm[s:s+bs].to(dev)
            z = F.normalize(proj(Xtr[b]), dim=-1)
            sim = z @ z.T / tau
            same = (ytr[b].unsqueeze(1) == ytr[b].unsqueeze(0)).float()
            same.fill_diagonal_(0)
            exp = torch.exp(sim) * (1 - torch.eye(sim.size(0), device=sim.device))
            denom = exp.sum(1, keepdim=True).clamp(min=1e-8)
            loss = -(torch.log((exp*same).sum(1, keepdim=True)/denom + 1e-8)* (same.sum(1, keepdim=True)>0)).sum() / max(1, int((same.sum(1)>0).sum()))
            opt.zero_grad(); loss.backward(); opt.step()
    nd = int(owners.max())+1
    clf = nn.Linear(128, nd).to(dev)
    opt2 = torch.optim.Adam(list(proj.parameters())+list(clf.parameters()), lr=1e-3)
    best, bestst, pat = -1, None, 0
    for ep in range(epochs):
        proj.train(); clf.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), bs):
            b = perm[s:s+bs].to(dev)
            opt2.zero_grad()
            loss = nn.CrossEntropyLoss()(clf(proj(Xtr[b])), ytr[b]); loss.backward(); opt2.step()
        proj.eval(); clf.eval()
        with torch.no_grad():
            acc = float((clf(proj(E[va].to(dev))).argmax(1).cpu().numpy() == owners[va]).mean())
        if acc > best: best, bestst, pat = acc, ([p.detach().cpu().clone() for p in proj.parameters()], [p.detach().cpu().clone() for p in clf.parameters()]), 0
        else: pat += 1
        if pat >= 3: break
    for p, s in zip(proj.parameters(), bestst[0]): p.data.copy_(s.to(dev))
    for p, s in zip(clf.parameters(), bestst[1]): p.data.copy_(s.to(dev))
    proj.eval(); clf.eval()
    with torch.no_grad():
        ste = clf(proj(E[te].to(dev))).cpu().numpy(); sve = clf(proj(E[va].to(dev))).cpu().numpy()
    torch.save({"proj": proj.state_dict(), "clf": clf.state_dict()}, model_path("sclbt", dataset, seed))
    save_scores("sclbt", dataset, seed, ste, owners[te], sve, owners[va], {"tau": tau}, cost=collect_cost(t0))

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed",type=int)
    a=ap.parse_args(); main(a.dataset,a.seed)
