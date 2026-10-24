"""汇总 results/baselines -> RQ1 大表 CSV（mean±sd，三种子）。列：MRR | HR@1/3/5/10 | NDCG@3/5/10。

Hy-LIN 行的来源有两处（等价，都是同一锁定配置）：
  1. results/baselines/<ds>/seed<s>/hylin.json —— run_hylin.py 产出的正式格；
  2. results/runs/ref_hp_locked_<ds>_s<s>_*/metrics.json —— 锁点时跑的参考复跑格。
二者指标同源同口径。为避免重复烧 9 格算力，第 2 类在缺第 1 类时作为别名回退，
并在输出里标注来源，保证 RQ1 大表的 hylin 行不会因命名差异而整行 n=0。
"""
import json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "baselines"
RUNS = ROOT / "results" / "runs"
METHODS = ["tfidf_svm", "deeptriage", "triagerx", "lightgcn", "cbtmf", "sclbt", "hylin"]
DS = ["gc", "mc", "mf"]
COLS = ["MRR", "HR@1", "HR@3", "HR@5", "HR@10", "NDCG@3", "NDCG@5", "NDCG@10"]


def _locked_alias(ds: str, seed: int):
    """锁点参考格别名回退。

    优先 ref_hp_locked_<ds>_s<seed>_*；MF 侧锁点是 hp_kb20_mf_s<seed>_*
    （k_bug=20 即锁点值）。通用做法：扫描 results/runs 下**配置与锁点逐项一致**
    的格，按名称稳定性排序取第一个，不写死 run_id 前缀——写死会在换锁点后
    悄悄匹配到旧配置的格子。
    """
    lock_keys = ("k_bug", "k_dev", "lr", "dropout", "freq_exponent",
                 "exp_layers", "imp_layers", "gate_mode", "valve", "fusion")
    try:
        lock_hp = json.loads((ROOT / "results" / "locked_hp.json").read_text())
    except Exception:
        lock_hp = {}
    lock = json.loads((ROOT / "configs" / "locked_model.json").read_text()) \
        if ("k_bug" in (json.loads((ROOT / "configs" / "locked_model.json").read_text()) if (ROOT / "configs" / "locked_model.json").exists() else {})) else lock_hp
    targets = {k: lock.get(k, lock_hp.get(k)) for k in lock_keys}

    # 1) 明确命名的参考格优先
    cands = sorted(RUNS.glob(f"ref_hp_locked_{ds}_s{seed}_*/metrics.json"))
    cands += sorted(RUNS.glob(f"*{ds}_s{seed}_*/metrics.json"))
    seen = set()
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        try:
            d = json.loads(p.read_text())
            c = d.get("config") or {}
            if any(abs(float(c.get(k, -999)) - float(v)) > 1e-9
                   for k, v in targets.items() if isinstance(v, (int, float))):
                continue
            if any(c.get(k) != v for k, v in targets.items() if not isinstance(v, (int, float))):
                continue
            t = d["test"]
            return t.get("all", t), p.parent.name
        except Exception:
            continue
    return None, None


def collect(ds: str, m: str):
    vals = {c: [] for c in COLS}
    n = 0
    source = "baselines"
    for s in (42, 43, 44):
        p = RES / ds / f"seed{s}" / f"{m}.json"
        if p.exists():
            t = json.loads(p.read_text())["test"]
            n += 1
        elif m == "hylin":
            t, alias = _locked_alias(ds, s)
            if t is None:
                continue
            n += 1
            source = f"runs:{alias}"
        else:
            continue
        for c in COLS:
            vals[c].append(t[c])
    return vals, n, source


rows = []
for ds in DS:
    for m in METHODS:
        vals, n, source = collect(ds, m)
        if n == 0:
            rows.append({"dataset": ds, "method": m, "n": 0, "source": ""})
        else:
            r = {"dataset": ds, "method": m, "n": n, "source": source}
            for c in COLS:
                r[c] = f"{statistics.mean(vals[c]):.4f}±{statistics.pstdev(vals[c]):.4f}" if n > 1 else f"{vals[c][0]:.4f}"
            rows.append(r)
import csv
out = ROOT / "tables" / "rq1_big.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "method", "n", "source"] + COLS)
    w.writeheader(); w.writerows(rows)
print(f"wrote {out} ({len(rows)} rows)")
for r in rows: print(r)
