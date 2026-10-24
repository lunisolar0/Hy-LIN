"""LightGCN（经典交互基线，无原文，按规矩实现）：bug-开发者二部图 + K=3 层无参传播 + BPR 训练。"""
import sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost

class LGCN(nn.Module):
    def __init__(s, nb, nd, ed=128, K=3):
        super().__init__(); s.E = nn.Embedding(nb+nd, ed); s.K=K; s.nb=nb; s.nd=nd
        nn.init.normal_(s.E.weight, std=0.01)
    def prop(s, adj):
        E = s.E.weight; outs=[E]
        for _ in range(s.K): E = adj @ E; outs.append(E)
        return torch.stack(outs).mean(0)

def main(dataset, seed, epochs=60, bs=4096):
    import time as _t; t0=_t.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.manual_seed(seed); np.random.seed(seed)
    owners, texts, _, tr, va, te = load_split(dataset, seed)
    # 局部化：训练 bug 重新编号
    bmap = {int(b): r for r, b in enumerate(tr)}
    nb, nd = len(tr), int(owners.max())+1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = torch.tensor([bmap[int(b)] for b in tr]); cols = torch.tensor(owners[tr]) + nb
    N = nb+nd
    idx = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])])
    deg = torch.bincount(idx[0], minlength=N).float()
    val = 1.0/torch.sqrt(deg[idx[0]]*deg[idx[1]])
    adj = torch.sparse_coo_tensor(idx, val, (N, N)).to(dev)
    m = LGCN(nb, nd).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    ytr = torch.from_numpy(owners[tr])
    ytrd = ytr.to(dev)
    for ep in range(epochs):
        m.train(); E = m.prop(adj)
        eb, edv = E[:nb], E[nb:]
        pos = ytrd; neg = torch.randint(0, nd, (nb,), device=dev)
        idx = torch.arange(nb, device=dev)
        s_pos = (eb[idx]*edv[pos]).sum(-1); s_neg = (eb[idx]*edv[neg]).sum(-1)
        loss = -(torch.log(torch.sigmoid(s_pos-s_neg))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        E = m.prop(adj); eb, edv = E[:nb], E[nb:]
        # val/test bug 不在训练图：用流行度回退 + 训练 bug 均值向量近似（标准冷启动规矩）
        import torch.nn.functional as F
        q = torch.stack([eb.mean(0)]* (len(va)+len(te)))
        ste = (q[len(va):] @ edv.T).cpu().numpy(); sve=(q[:len(va)] @ edv.T).cpu().numpy()
    torch.save({"state": m.state_dict()}, model_path("lightgcn", dataset, seed))
    save_scores("lightgcn", dataset, seed, ste, owners[te], sve, owners[va], {"K":3,"note":"cold-start mean-pool"}, cost=collect_cost(t0, m))

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed",type=int)
    a=ap.parse_args(); main(a.dataset,a.seed)
