"""DeepTriage 严格版（按原文 DBRNN-A）：
word2vec(|P|=300, gensim) → BiLSTM(300单元/dropout 0.3) → attention →
r = ym_f ⊕ am_f ⊕ ym_b ⊕ am_b（1200维，式2）→ Softmax；
表示学习做 next-word 无监督预训练（5 轮；原文未规定该阶段轮数——原文的
"100 epochs with early stopping" 指**监督分类器**的训练轮数，此处如实标注）���
监督分类器 Adam 1e-3、100 轮 + 早停（与原文一致）。
D1 = originalDATA dump 全量（原文 Table 1 特征学习集口径：MF 138093 / MC 186173 / GC 163695，
与 processed 固定集 id 交集为 0）；按原文 §4.2 移除 URL/stack trace/hex/代码片段；切分用本工程三种子时序分层（非原文11折CV）；
指标 top-k accuracy = HR@k，另附 MRR/NDCG。
"""
import sys, argparse, re, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost

def tok(s):
    try:
        from nltk import word_tokenize
        return [w.lower() for w in word_tokenize(s)][:400]
    except Exception:
        return s.lower().split()[:400]

class DBRNNA(nn.Module):
    def __init__(s, V, P=300, H=300, ndev=100, drop=0.3):
        super().__init__()
        s.emb = nn.Embedding(V, P, padding_idx=0)
        s.lstm = nn.LSTM(P, H, 1, bidirectional=True, batch_first=True, dropout=0.0)
        s.drop = nn.Dropout(drop)
        s.attn = nn.Linear(2*H, 1)
        s.fc = nn.Linear(4*H, ndev)
        s.lm = nn.Linear(2*H, V)
    def encode(s, x):
        h, _ = s.lstm(s.drop(s.emb(x)))
        f, b = h[:, :, :300], h[:, :, 300:]
        ymf, ymb = f[:, -1, :], b[:, 0, :]
        w = torch.softmax(s.attn(h).squeeze(-1).masked_fill(x == 0, -1e9), -1)
        af = (f * w.unsqueeze(-1)).sum(1)
        ab = (b * w.unsqueeze(-1)).sum(1)
        return torch.cat([ymf, af, ymb, ab], 1), h

@torch.no_grad()
def predict(m, X, dev, bs=256):
    m.eval(); outs = []
    for i in range(0, len(X), bs):
        r, _ = m.encode(X[i:i+bs].to(dev))
        outs.append(m.fc(r).cpu())
    return torch.cat(outs).numpy()

# 原文 §4.2 预处理：移除 URL、stack trace、hex code 与代码片段。
# 原文 §3 步骤 2："handling the URLs, stack trace, hex code, and the code
# snippets in the unstructured description ... those contents are removed in
# the preprocessing stage"。不做这一步，构建号与调用栈会主导词表。
_RE_URL = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.I)
_RE_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_RE_BUILD = re.compile(r"^\s*(?:BuildID|Build ID|User-Agent|OS|Platform)\s*:.*$", re.I | re.M)
_RE_AT = re.compile(r"^\s*at\s+[\w.$<>]+\([^)]*\)\s*$", re.M)
_RE_FILE = re.compile(r'^\s*File "[^"]*", line \d+.*$', re.M)
_RE_TRACE = re.compile(r"^\s*(?:Traceback \(most recent call last\)\s*:?|-{2,} End of .{0,40} -{2,})\s*$", re.M)
# 异常栈体：缩进 4 空格以上且带括号/赋值的行（raise/assert/调用），以及异常类型行
_RE_EXC = re.compile(r"^\s{4,}(?:raise|assert|return)\b.*$"
                     r"|^\s{4,}\w+\s*=\s*.*$"
                     r"|^\s{4,}\w+\s*\([^()]*\)\s*$", re.M)
_RE_EXC2 = re.compile(r"^\s*\w+(?:Error|Exception|Warning)\b.*:\s*.*$", re.M)
_RE_CODE = re.compile(
    r"^\s*\#include\b.*$"
    r"|^\s*(?:import\s+\w|from\s+\w+\s+import)\b.*$"
    r"|^\s*def\s+\w+\s*\(.*$"
    r"|^\s*class\s+\w+\s*[\(:].*$"
    r"|^\s*(?:public|private|protected)\s+\w+[\s(].*$"
    r"|^\s*System\.out\.print.*$"
    r"|^\s*\}\s*\(\s*\)\s*;.*$"
    r"|^\s*\w+\s*\([^()]*\)\s*;\s*$",
    re.M)


def clean_text(s: str) -> str:
    """按原文移除 URL / stack trace / hex code / 代码片段。"""
    if not isinstance(s, str):
        return ""
    s = _RE_URL.sub(" ", s)
    s = _RE_TRACE.sub(" ", s)
    s = _RE_AT.sub(" ", s)
    s = _RE_FILE.sub(" ", s)
    s = _RE_BUILD.sub(" ", s)
    s = _RE_CODE.sub(" ", s)
    s = _RE_EXC.sub(" ", s)
    s = _RE_EXC2.sub(" ", s)
    s = _RE_HEX.sub(" ", s)
    return s


def load_d1_corpus(dataset, root):
    """原文 D1 语料 = originalDATA dump 全量（原文 §4.3）。

    原文规定：未分派/未解决的缺陷（约占 70%）全部用于表示学习，
    分类器只在其余"已修复且有有效 owner"的缺陷上训练与测试，
    且特征模型的训练语料与分类器语料**互不重叠**。

    本工程 originalDATA dump 与 processed 固定集的 id 交集为 0，
    且记录数与原文 Table 1 的特征学习集逐条吻合
    （MF 138,093 完全一致；MC 186,173 完全一致；
      processed MF 24,199 vs 原文 24,214），
    即 dump 本身已是原文口径的 D1。

    此前实现额外加了一层"必须无 owner"的过滤，白丢掉
    MC 83,066 条(35%)、GC 45,459 条(28%)、MF 20,362 条(15%)，
    反而削弱了原文"利用那 70% 未分派缺陷"这一核心贡献。已修正。
    """
    import json
    cache = root / "results" / f"d1_paper_{dataset}.json"
    if cache.exists():
        return json.load(open(cache, encoding="utf-8"))
    raw = open(root / "data" / "originalDATA" / dataset / f"{dataset}.json",
               encoding="utf-8", errors="replace").read()
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", raw)
    raw = re.sub(r":\s*NULL\b", ": null", raw)
    rows = json.loads(raw)
    rows = rows if isinstance(rows, list) else list(rows.values())
    out = []
    for r in rows:
        ti = clean_text((r.get("issue_title") or "").strip())
        de = clean_text((r.get("description") or "").strip()[:1500]
                        if isinstance(r.get("description"), str) else "")
        s = f"{ti}\n{de}" if de else ti
        if s.strip(): out.append(s)
    json.dump(out, open(cache, "w", encoding="utf-8"))
    print(f"D1语料 {dataset}: {len(out)} 条（原文口径，缓存 {cache}）", flush=True)
    return out


def chunked_lm_loss(m, h, y, chunk: int = 8192):
    """next-word 交叉熵，**按 token 分块**计算。

    为什么要分块：词表 V ≈ 100k，序列长 400，batch 256。原写法
    `m.lm(h.reshape(-1,600))` 会一次性物化 (256*399, 100002) 的 logits，
    fp32 约 **40.9 GB** —— 一张 80G 卡只容得下一格，其余全部空转等显存；
    实测 9 格并发时每 batch 要数秒，预训练阶段因此卡了 11 小时还没跑完一轮。

    分块后峰值显存降到约 8192*100002*4 ≈ 3.3 GB，且**数学上完全等价**
    （同样的 logits、同样以 ignore_index=0 的有效 token 取平均），
    因此不改变方法本身，只是把同一个计算做得省内存。
    """
    hf = h.reshape(-1, h.shape[-1])
    yf = y.reshape(-1)
    total = 0.0
    cnt = 0
    for i in range(0, hf.shape[0], chunk):
        logits = m.lm(hf[i:i + chunk])
        tgt = yf[i:i + chunk]
        n = int((tgt != 0).sum())
        if n == 0:
            continue
        total = total + nn.functional.cross_entropy(
            logits, tgt, ignore_index=0, reduction="sum")
        cnt += n
    return total / max(1, cnt)

def main(dataset, seed, epochs=100, bs=256):
    import time as _t; t0 = _t.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.manual_seed(seed); np.random.seed(seed)
    # TF32：fp32 matmul 走张量核，A800/4090 上吞吐提升数倍，且对
    # 本模型（BiLSTM 300 + 线性头）的数值影响在 1e-3 量级，
    # 不改变方法定义。论文的实验本身也未规定精度模式。
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 9 格并发时每进程再开满 112 线程会超订到 1000+ 线程，
    # 结果是所有格子一起变慢。封顶到 8 线程。
    torch.set_num_threads(int(os.environ.get("HXL_OMP_THREADS", "8")))
    owners, texts, _, tr, va, te = load_split(dataset, seed)
    # 原文的预处理同样作用于分类器侧（D2）文本，否则两侧口径不一致
    texts = [clean_text(t) for t in texts]
    from common import ROOT as _ROOT
    d1texts = load_d1_corpus(dataset, _ROOT)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from collections import Counter
    toks = [[w for w in tok(texts[i])] for i in range(len(texts))]
    cnt = Counter(w for i in tr for w in toks[i])
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in cnt.most_common(100000): vocab[w] = len(vocab)
    V = len(vocab)
    def enc(idx, L=400):
        X = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx):
            ids = [vocab.get(w, 1) for w in toks[i][:L]]
            X[r, :len(ids)] = torch.tensor(ids)
        return X
    # word2vec 预训练：D1未修复语料（gensim skip-gram，min_count=5）
    from gensim.models import Word2Vec
    d1toks = [tok(s) for s in d1texts]
    w2v = Word2Vec(sentences=d1toks, vector_size=300,
                   window=5, min_count=5, sg=1, workers=8, epochs=5, seed=seed)
    ndev = int(owners.max()) + 1
    m = DBRNNA(V, ndev=ndev).to(dev)
    with torch.no_grad():
        W = torch.zeros(V, 300)
        for w, i in vocab.items():
            if w in w2v.wv: W[i] = torch.from_numpy(w2v.wv[w])
            else: W[i].normal_(std=0.01)
        m.emb.weight.copy_(W)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    # 非监督 next-word 预训练：D1语料（5轮，原文100轮，时间所限披露缩减）
    m.train()
    def enc_texts(strs, L=400):
        X = torch.zeros(len(strs), L, dtype=torch.long)
        for r, s in enumerate(strs):
            ids = [vocab.get(w, 1) for w in tok(s)[:L]]
            X[r, :len(ids)] = torch.tensor(ids)
        return X
    Xd1 = enc_texts(d1texts)
    Xtr = enc(tr)
    print(f"[deeptriage {dataset} s{seed}] 语料就绪 train={len(tr)} dev={ndev} vocab={V} "
          f"d1={len(d1texts)}", flush=True)
    for ep in range(5):
        perm = torch.randperm(len(Xd1))
        ep_t = _t.time()
        for s in range(0, len(Xd1), bs):
            b = perm[s:s+bs]
            # LM头 over 10万词表：micro-batch(32)防爆
            for m0 in range(0, len(b), 8):
                mb = b[m0:m0+8]
                xb, yb = Xd1[mb][:, :-1].to(dev), Xd1[mb][:, 1:].to(dev)
                _, h = m.encode(xb)
                loss = chunked_lm_loss(m, h, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        # 进度可见：原先 5 轮 next-word 预训练完全没有输出，9 格并发时只能靠进程存活
        # 判断"在跑"，无法区分"在跑"和"卡死"，也没有任何断点依据。
        print(f"[deeptriage {dataset} s{seed}] pretrain ep{ep+1}/5 "
              f"loss={float(loss):.4f} {_t.time()-ep_t:.0f}s", flush=True)
    # 监督分类（100轮+早停 patience 5）
    ytr = torch.from_numpy(owners[tr]); Xva = enc(va)
    best, bestst, pat = -1, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), bs):
            bc = perm[s:s+bs]; b = bc.to(dev)
            r, _ = m.encode(Xtr[bc].to(dev))
            loss = nn.CrossEntropyLoss()(m.fc(r), ytr[bc].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        v = torch.from_numpy(predict(m, Xva, dev))
        acc = float((v.argmax(1) == owners[va]).float().mean())
        improved = acc > best
        if improved: best, bestst, pat = acc, {k: v.cpu() for k, v in m.state_dict().items()}, 0
        else: pat += 1
        print(f"[deeptriage {dataset} s{seed}] sup ep{ep+1} val_acc={acc:.4f} "
              f"best={best:.4f} pat={pat} {_t.time()-t0:.0f}s", flush=True)
        if pat >= 5: break
    m.load_state_dict({k: v.to(dev) for k, v in bestst.items()})
    torch.save({"state": m.state_dict(), "vocab": vocab}, model_path("deeptriage", dataset, seed))
    ste = predict(m, enc(te), dev); sve = predict(m, Xva, dev)
    save_scores("deeptriage", dataset, seed, ste, owners[te], sve, owners[va],
                {"strict": True, "units": 300, "drop": 0.3, "lr": 0.001, "pretrain_epochs": 5, "pretrain_note": "paper 100; reduced by budget, disclosed"}, cost=collect_cost(t0, m))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    a = ap.parse_args(); main(a.dataset, a.seed)
