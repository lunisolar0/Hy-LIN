"""TriagerX 严格版（按原文 dual-transformer CBR（RQ2单组件））：
DeBERTa-base + RoBERTa-base（编码器冻结、特征 no_grad 预计算，统一预算，§4.3.2），取最后三层拼接（2304维/PLM）→
3个CNN分类器（4并行卷积fs={3,4,5,6}x256filter+BN+ReLU+maxpool+dropout，可学习CWk加权，式2）→
AdamW(wd=0.001)+linear schedule，256 tokens；
CBR单排（按原文RQ2允许单独评估ranker组件；本数据无交互/时间戳，IBR无法实例化）。
预处理按原文：URL/时间戳/特殊字符/代码块去除、hex→<hex>；<15词报告保留（保统一评测口径，披露）。
SBERT无all-mpnet-base-v2本地权重（离线），IBR未启用，无替代问题。
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost

class CNNHead(nn.Module):
    def __init__(s, ind=2304, ndev=100, drop=0.5):
        super().__init__()
        s.convs = nn.ModuleList([nn.Conv1d(ind, 256, k, padding=k//2) for k in (3, 4, 5, 6)])
        s.bns = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(4)])
        s.drop = nn.Dropout(drop)
        s.fc = nn.Linear(1024, ndev)
    def forward(s, x):  # x: [B, L, 2304]
        h = x.transpose(1, 2)
        feats = [torch.relu(bn(conv(h))).max(-1).values for conv, bn in zip(s.convs, s.bns)]
        return s.fc(s.drop(torch.cat(feats, 1)))

import re
_HEX = re.compile(r"0[xX][0-9a-fA-F]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_TS = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?|\d{2}:\d{2}(:\d{2})?")
_CODE = re.compile(r"```[\s\S]*?```")
_WS = re.compile(r"[^a-zA-Z0-9<>_\s]")
def tx_clean(s):
    s = _CODE.sub(" ", s); s = _URL.sub(" ", s); s = _TS.sub(" ", s)
    s = _HEX.sub("<hex>", s); s = _WS.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def _mem_avail_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable"):
                return int(l.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 999.0

def _pid_alive(pid):
    import os
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def _acquire_budget(need_gb, gatedir, budget_gb=None):
    """全局内存预算闸：按存活 pid 统计已预约量，原子(flock)读改写，死进程残留自动回收。
    预约持有到进程结束（特征在训练期仍驻留）。"""
    import time, os, fcntl
    # 预算上限可外部覆盖。取值必须落在"够一个最大格"与"装不下两个大格"之间：
    #   gc 单格需求 = 118000 x 64 x 4608 x 2B ~= 70G（deberta+roberta 两模型，sum(dims)=4608）
    #   mc 单格约 14G，mf 约 7G
    # 90G 会让 gc+mc(约 84G) 一起放行，实测触发 SIGBUS(135)：本机只有 125G 物理内存且与他组共享；
    # 低于 70G 又会让 gc 永远进不来，卡在闸里空转（这是踩过的坑）。
    # 80G：放行单个 gc(70G)，挡住 gc+mc(84G>80)。
    if budget_gb is None:
        budget_gb = float(os.environ.get("HXL_TXF_BUDGET_GB", "80"))
    rd = gatedir / ".txfeat_res"; rd.mkdir(exist_ok=True)
    lockf = gatedir / ".txfeat_reslock"; lockf.touch(exist_ok=True)
    myf = rd / str(os.getpid())
    while True:
        f = open(lockf, "r+"); fcntl.flock(f, fcntl.LOCK_EX)
        used = 0.0
        for p in rd.iterdir():
            try:
                pid = int(p.name)
            except Exception:
                continue
            if pid == os.getpid():
                continue
            if _pid_alive(pid):
                try:
                    used += float(p.read_text().strip() or "0")
                except Exception:
                    pass
            else:
                try:
                    p.unlink()
                except Exception:
                    pass
        ok = (used + need_gb <= budget_gb) and (_mem_avail_gb() > need_gb + 8)
        if ok:
            myf.write_text(str(need_gb))
        fcntl.flock(f, fcntl.LOCK_UN); f.close()
        if ok:
            return myf
        time.sleep(20)

def _release_budget(token):
    try:
        token.unlink()
    except Exception:
        pass

@torch.no_grad()
def build_feat_inram(texts, idx, plms_by_dev, toks, dims, devices, bs=32, keep=64):
    """内存驻留特征：建 CPU fp16 张量，全程不落盘。
    关键：每轮先向 nd 张卡各自异步发起前向(不等回读)，再统一 .cpu() 回读，
    令多卡在同一时间窗内真正并发计算（而非交替空转），实现单进程吃满双卡。"""
    n = len(idx)
    out = torch.empty((n, keep, sum(dims)), dtype=torch.float16)
    nd = len(devices)
    off = 0
    for mi, dm in enumerate(dims):
        tk = toks[mi]
        starts = list(range(0, n, bs))
        for k in range(0, len(starts), nd):
            group = starts[k:k + nd]
            pending = []
            for gi, s in enumerate(group):
                d = devices[gi]
                plm = plms_by_dev[gi][mi]
                bi = idx[s:s+bs]
                b = tk([texts[i] for i in bi], padding=True, truncation=True,
                       max_length=256, return_tensors="pt").to(d)
                hs = plm(**{x: v for x, v in b.items() if x != "token_type_ids"},
                         output_hidden_states=True).hidden_states
                pending.append((s, len(bi), hs))
            # 两卡前向已并发排队，此处统一回读
            for s, m, hs in pending:
                f = torch.cat(hs[-3:], -1)[:, :keep, :].cpu()
                if f.size(1) < keep:
                    f = torch.nn.functional.pad(f, (0, 0, 0, keep - f.size(1)))
                out[s:s + m, :, off:off + dm] = f.half()
                del hs, f
        off += dm
    return out

def main(dataset, seed, epochs=3, bs=16):
    import time as _t; t0 = _t.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.manual_seed(seed); np.random.seed(seed)
    owners, texts_raw, _, tr, va, te = load_split(dataset, seed)
    texts = [tx_clean(x) for x in texts_raw]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    ndev = int(owners.max()) + 1
    names = ("microsoft/deberta-base", "FacebookAI/roberta-base")
    toks = [AutoTokenizer.from_pretrained(n, local_files_only=True) for n in names]
    # 多卡：每张卡各持一份 PLM 副本，供 build_feat_inram 交替并行（单进程吃双卡）
    devices = [dev]
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    plms_by_dev = []
    for _d in devices:
        _ms = []
        for name in names:
            pm = AutoModel.from_pretrained(name, local_files_only=True).to(_d)
            # 编码器冻结：特征在 no_grad 下一批算完即释放（与基线统一预算，§4.3.2）。
            for p in pm.parameters():
                p.requires_grad = False
            _ms.append(pm)
        plms_by_dev.append(_ms)
    dims = tuple(3 * pm.config.hidden_size for pm in plms_by_dev[0])  # 末三层拼接维度
    import os
    # 缓存目录必须带上划分名：特征矩阵按 tr/va/te 的下标排列，划分一变内容就不同。
    # 早期版本只判断文件是否存在，换协议后会静默加载旧划分的特征。
    _split_tag = str(os.environ.get("HXL_SPLIT", "chronological"))
    featdir = model_path("triagerx", dataset, seed).parent / f"txfeat_{dataset}_{_split_tag}_s{seed}"
    featdir.mkdir(parents=True, exist_ok=True)
    mm = {}
    _uncached = [sp for sp in ("tr", "va", "te")
                 if not (featdir / f"{sp}.npy").exists()]
    if _uncached:
        import atexit
        _idxs = {"tr": tr, "va": va, "te": te}
        _need = sum(len(_idxs[sp]) * 64 * sum(dims) * 2 for sp in _uncached) / 1e9
        _gate = _acquire_budget(_need, featdir.parent)
        atexit.register(_release_budget, _gate)   # 持有到进程结束（特征训练期仍驻留）
    for split, idx in (("tr", tr), ("va", va), ("te", te)):
        fp = featdir / f"{split}.npy"
        if fp.exists():
            a = np.load(fp, mmap_mode="r")
            mm[split] = ("disk", fp, tuple(a.shape))
        else:
            t = build_feat_inram(texts, idx, plms_by_dev, toks, dims, devices)
            mm[split] = ("ram", t, tuple(t.shape))
    def get_batch(split, bi):
        kind, ref, _ = mm[split]
        if kind == "disk":
            a = np.load(ref, mmap_mode="r")
            return torch.from_numpy(np.array(a[bi])).float()
        return ref[bi].float()
    n_tr, n_va, n_te = mm["tr"][2][0], mm["va"][2][0], mm["te"][2][0]
    heads = [CNNHead(4608, ndev).to(dev) for _ in range(3)]
    cw = nn.Parameter(torch.zeros(3, device=dev))
    ytr_cpu = torch.from_numpy(owners[tr])
    opt = torch.optim.AdamW([pr for h in heads for pr in h.parameters()] + [cw],
                            lr=2e-5, weight_decay=0.001)
    del plms_by_dev
    import gc as _gc; _gc.collect()
    Xt_len, Xv_len = n_tr, n_va
    tot = (n_tr // bs) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(tot*0.1), tot)
    wfn = lambda: torch.softmax(cw, 0)
    for ep in range(epochs):
        for h in heads: h.train()
        perm = torch.randperm(n_tr)
        for s in range(0, n_tr, bs):
            b = perm[s:s+bs]
            xb = torch.stack([get_batch("tr", int(i)) for i in b]).to(dev)
            w = wfn()
            logits = sum(wi * h(xb) for wi, h in zip(w, heads))
            loss = nn.CrossEntropyLoss()(logits, ytr_cpu[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        w = wfn()
        sve = torch.cat([sum(wi * h(torch.stack([get_batch("va", i) for i in range(s, min(s+bs, n_va))]).to(dev)) for wi, h in zip(w, heads)).cpu() for s in range(0, n_va, bs)]).numpy()
        ste_cbr = torch.cat([sum(wi * h(torch.stack([get_batch("te", i) for i in range(s, min(s+bs, n_te))]).to(dev)) for wi, h in zip(w, heads)).cpu() for s in range(0, n_te, bs)]).numpy()
    ste = ste_cbr
    torch.save({"heads": [h.state_dict() for h in heads], "cw": cw.detach().cpu()},
               model_path("triagerx", dataset, seed))
    save_scores("triagerx", dataset, seed, ste, owners[te], sve, owners[va],
                {"strict": True, "plms": ["deberta-base", "roberta-base"], "kpft": True,
                 "ranker": "CBR-only-per-RQ2", "preproc": "paper-clean-keep-short",
                 "ibr": "not-applicable-no-interaction-data"},
                cost=collect_cost(t0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    a = ap.parse_args(); main(a.dataset, a.seed)
