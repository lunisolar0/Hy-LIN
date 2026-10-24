"""DeepTriage 近似（按原文：word2vec + BiLSTM + Attention + Softmax 分类）。
为可跑性：Tokenizer 用 whitespace + vocab 30k，embedding 200 维随机初始化联合训练，
BiLSTM(128x2) + additive attention + linear 全开发者分类头。"""
import sys, argparse, random
from pathlib import Path
from collections import Counter
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost

def tok(s): return s.lower().split()[:256]

class DT(nn.Module):
    def __init__(s, V, ndev, ed=200, hd=128):
        super().__init__()
        s.emb = nn.Embedding(V, ed, padding_idx=0)
        s.lstm = nn.LSTM(ed, hd, 2, bidirectional=True, batch_first=True, dropout=0.2)
        s.att = nn.Linear(2*hd, 1)
        s.fc = nn.Linear(2*hd, ndev)
    def forward(s, x):
        h, _ = s.lstm(s.emb(x))
        w = torch.softmax(s.att(h).squeeze(-1).masked_fill(x==0, -1e9), -1)
        return s.fc((h*w.unsqueeze(-1)).sum(1))

@torch.no_grad()
def predict(m, X, dev, bs=512):
    m.eval(); outs = []
    for s in range(0, len(X), bs):
        outs.append(m(X[s:s+bs].to(dev)).cpu())
    return torch.cat(outs).numpy()

def main(dataset, seed, epochs=15, bs=256):
    import time as _t; t0=_t.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    owners, texts, _, tr, va, te = load_split(dataset, seed)
    cnt = Counter(w for i in tr for w in tok(texts[i]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in cnt.most_common(30000): vocab[w] = len(vocab)
    def enc(idx):
        X = torch.zeros(len(idx), 256, dtype=torch.long)
        for r, i in enumerate(idx):
            ids = [vocab.get(w, 1) for w in tok(texts[i])]
            X[r, :len(ids)] = torch.tensor(ids)
        return X
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DT(len(vocab), int(owners.max())+1).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xtr, ytr = enc(tr), torch.from_numpy(owners[tr])
    best, bestst, pat = -1, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), bs):
            b = perm[s:s+bs]
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(m(Xtr[b].to(dev)), ytr[b].to(dev)); loss.backward(); opt.step()
        v = torch.from_numpy(predict(m, enc(va), dev))
        pred = v.argmax(1).numpy()
        acc = float((pred == owners[va]).mean())
        if acc > best: best, bestst, pat = acc, {k: v.cpu() for k, v in m.state_dict().items()}, 0
        else: pat += 1
        if pat >= 3: break
    m.load_state_dict({k: v.to(dev) for k, v in bestst.items()})
    ste = predict(m, enc(te), dev); sve = predict(m, enc(va), dev)
    torch.save({"state": m.state_dict(), "vocab": vocab}, model_path("deeptriage", dataset, seed))
    save_scores("deeptriage", dataset, seed, ste, owners[te], sve, owners[va], {"epochs": epochs}, cost=collect_cost(t0, m))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    a = ap.parse_args(); main(a.dataset, a.seed)
