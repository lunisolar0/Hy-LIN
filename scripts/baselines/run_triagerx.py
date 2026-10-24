"""TriagerX 近似（按原文：Transformer 编码 bug 文本 + 分类头；身份特殊，评测仍放文本组）。
实现：sentence-transformers MiniLM 编码后冻住 + 2 层 MLP 分类头（与 Hy-LIN 同编码器，保证公平）。"""
import sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost

def main(dataset, seed, epochs=15, bs=512):
    import time as _t; t0=_t.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.manual_seed(seed); np.random.seed(seed)
    owners, texts, _, tr, va, te = load_split(dataset, seed)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    E = st.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ndev = int(owners.max()) + 1
    mlp = nn.Sequential(nn.Linear(384, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, ndev)).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    Xtr = torch.from_numpy(E[tr]); ytr = torch.from_numpy(owners[tr])
    best, bestst, pat = -1, None, 0
    for ep in range(epochs):
        mlp.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), bs):
            b = perm[s:s+bs]; opt.zero_grad()
            loss = nn.CrossEntropyLoss()(mlp(Xtr[b].to(dev)), ytr[b].to(dev)); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            acc = float((mlp(torch.from_numpy(E[va]).to(dev)).argmax(1).cpu().numpy() == owners[va]).mean())
        if acc > best: best, bestst, pat = acc, {k: v.cpu() for k, v in mlp.state_dict().items()}, 0
        else: pat += 1
        if pat >= 3: break
    mlp.load_state_dict({k: v.to(dev) for k, v in bestst.items()}); mlp.eval()
    with torch.no_grad():
        ste = mlp(torch.from_numpy(E[te]).to(dev)).cpu().numpy()
        sve = mlp(torch.from_numpy(E[va]).to(dev)).cpu().numpy()
    torch.save({"state": mlp.state_dict()}, model_path("triagerx", dataset, seed))
    save_scores("triagerx", dataset, seed, ste, owners[te], sve, owners[va], {"backbone": "minilm-frozen"}, cost=collect_cost(t0, mlp))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    a = ap.parse_args(); main(a.dataset, a.seed)
