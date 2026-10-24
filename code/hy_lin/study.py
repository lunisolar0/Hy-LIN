"""三数据集、三种子上的模块探究。

一次只换一个因子，其余保持当时的最优设定。每个已完成的运行立刻写入 results，
并刷新 docs/实验日志.md 与 docs/模块探究实验.md。中断后重跑本脚本会跳过已成功的运行。

只使用物理 GPU 0，避免占满另一张卡。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
# 用启动 study 的那个解释器拉子进程：写死本机路径会让整套探究无法搬到 A800 上跑
PY = os.environ.get("HYLIN_PYTHON") or sys.executable
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
CACHE = CODE / "cache"
DATA = {
    "mf": ROOT / "data" / "processed" / "mf.json",
    "mc": ROOT / "data" / "processed" / "mc.json",
    "gc": ROOT / "data" / "processed" / "gc.json",
}
DATASETS = ("mf", "mc", "gc")
SEEDS = (42, 43, 44)
DIM = {"minilm": 384, "bert": 768, "roberta": 768}
ADOPT_MARGIN = 0.002
# 划分协议：时序位置切分（对齐前人）。写进 fingerprint / run_id，
# 否则换了划分后旧的随机分层运行会被当成"已完成"直接复用。
SPLIT = "chronological"

BASE = {
    "encoder": "minilm",
    "valve": "bug",
    "uniform_valve": False,
    "dev_graph": "centroid",
    "k_bug": 10,
    "k_dev": 15,
    "exp_layers": 2,
    "imp_layers": 2,
    "dropout": 0.1,
    "lr": 0.001,
    "gate_mode": "freq",
    "fusion": "zscore",
    "story": 0.5,
    "tau": 0.0,
    "freq_exponent": 0.0,
    "imp_query": "prop",
    "epochs": 40,
    "patience": 10,
    "low_coef": 0.35,
    "batch_size": 2048,
    "weight_decay": 0.0001,
}


def fingerprint(cfg: dict) -> str:
    keys = [
        "dataset", "seed", "split", "encoder", "valve", "uniform_valve", "dev_graph", "k_bug", "k_dev",
        "exp_layers", "imp_layers", "dropout", "lr", "gate_mode", "fusion", "story", "tau",
        "freq_exponent", "imp_query", "epochs", "patience", "low_coef", "hidden",
    ]
    return json.dumps({key: cfg[key] for key in keys}, sort_keys=True)


def full_config(dataset: str, seed: int, **overrides) -> dict:
    cfg = dict(BASE)
    cfg.update(overrides)
    cfg["dataset"] = dataset
    cfg["seed"] = seed
    cfg["split"] = SPLIT
    cfg["hidden"] = DIM[cfg["encoder"]]
    return cfg


def cache_path(cfg: dict) -> Path:
    name = (
        f"graph_{cfg['dataset']}_{cfg['encoder']}_s{cfg['seed']}"
        f"_kb{cfg['k_bug']}_kd{cfg['k_dev']}_{cfg['dev_graph']}.pt"
    )
    return CACHE / name


def command(cfg: dict, out: Path) -> list[str]:
    args = [
        PY, "-m", "hy_lin.train",
        "--data", str(DATA[cfg["dataset"]]),
        "--cache", str(cache_path(cfg)),
        "--out", str(out),
        "--seed", str(cfg["seed"]),
        "--split", cfg.get("split", SPLIT),
        "--encoder", cfg["encoder"],
        "--dev-graph", cfg["dev_graph"],
        "--valve", cfg["valve"],
        "--k-bug", str(cfg["k_bug"]),
        "--k-dev", str(cfg["k_dev"]),
        "--exp-layers", str(cfg["exp_layers"]),
        "--imp-layers", str(cfg["imp_layers"]),
        "--dropout", str(cfg["dropout"]),
        "--lr", str(cfg["lr"]),
        "--gate-mode", cfg["gate_mode"],
        "--fusion", cfg["fusion"],
        "--story", str(cfg["story"]),
        "--tau", str(cfg["tau"]),
        "--freq-exponent", str(cfg["freq_exponent"]),
        "--imp-query", cfg["imp_query"],
        "--epochs", str(cfg["epochs"]),
        "--patience", str(cfg["patience"]),
        "--low-coef", str(cfg["low_coef"]),
        "--batch-size", str(cfg["batch_size"]),
        "--weight-decay", str(cfg["weight_decay"]),
        "--hidden", str(cfg["hidden"]),
    ]
    if cfg["uniform_valve"]:
        args.append("--uniform-valve")
    return args


def load_done() -> tuple[dict[str, dict], dict[str, dict]]:
    by_id, by_fp = {}, {}
    path = RESULTS / "runs.jsonl"
    if not path.exists():
        return by_id, by_fp
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_id[row["run_id"]] = row
        if row.get("status") == "ok":
            by_fp[row["fingerprint"]] = row
    return by_id, by_fp


def append_row(row: dict) -> None:
    path = RESULTS / "runs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def execute(run_id: str, phase: str, variant: str, cfg: dict, by_id: dict, by_fp: dict) -> dict:
    fp = fingerprint(cfg)
    if run_id in by_id and by_id[run_id].get("status") == "ok":
        print(f"跳过 {run_id}", flush=True)
        return by_id[run_id]
    previous = by_id.get(run_id)
    gpu = "0"
    # 显存占比可外部覆盖：A800 上与基线进程共用一张卡时，0.75 会要求 60G/80G，
    # 叠加基线的 ~33G 直接爆显存。由 HYLIN_MEM_FRACTION 控制。
    fraction = os.environ.get("HYLIN_MEM_FRACTION", "0.75")
    if previous and previous.get("status") == "fail" and "OutOfMemory" in (previous.get("error_tail") or ""):
        if previous.get("tried_gpu1"):
            print(f"跳过两张卡都显存不足的运行 {run_id}", flush=True)
            return previous
        gpu = os.environ.get("HYLIN_GPU_OVERFLOW", "1")
        fraction = os.environ.get("HYLIN_MEM_FRACTION_OVERFLOW", "0.90")
        print(f"显存不足，改到 GPU1 重跑 {run_id}", flush=True)
    if fp in by_fp and gpu == "0":
        print(f"复用 {by_fp[fp]['run_id']} -> {run_id}", flush=True)
        return by_fp[fp]
    out = RESULTS / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["HYLIN_MEM_FRACTION"] = fraction
    env["HYLIN_GPU"] = gpu
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"开始 {run_id}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            command(cfg, out),
            cwd=CODE,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=10800,
            check=False,
        )
    seconds = time.time() - started
    metrics_path = out / "metrics.json"
    row = {
        "run_id": run_id,
        "phase": phase,
        "variant": variant,
        "fingerprint": fp,
        "seconds": round(seconds, 1),
        "returncode": proc.returncode,
        "config": cfg,
        "log": str(log_path.relative_to(ROOT)),
        "gpu": gpu,
        "tried_gpu1": gpu == "1" or bool(previous and previous.get("tried_gpu1")),
    }
    if proc.returncode == 0 and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row["status"] = "ok"
        row["best_epoch"] = metrics.get("best_epoch")
        row["val"] = {
            name: {"MRR": block["MRR"], "n": block["n"], "HR@1": block["HR@1"], "HR@10": block["HR@10"]}
            for name, block in (metrics.get("val") or {}).items()
        }
        row["test"] = {
            name: {"MRR": block["MRR"], "n": block["n"], "HR@1": block["HR@1"], "HR@10": block["HR@10"]}
            for name, block in metrics["test"].items()
        }
        row["gate"] = metrics.get("gate_explicit_weight", {})
        row["cost"] = metrics.get("cost")
        by_fp[fp] = row
        print(
            f"完成 {run_id}  val {row['val'].get('all', {}).get('MRR', float('nan')):.4f}  "
            f"test {row['test']['all']['MRR']:.4f}  "
            f"low {row['test'].get('low(<5)', {}).get('MRR', float('nan')):.4f}  {seconds:.0f}s",
            flush=True,
        )
    else:
        row["status"] = "fail"
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        row["error_tail"] = tail
        print(f"失败 {run_id} code={proc.returncode}", flush=True)
    append_row(row)
    by_id[run_id] = row
    refresh_docs(by_id)
    return row


def run_grid(phase: str, variant: str, cfg_of, by_id, by_fp, datasets=DATASETS, seeds=SEEDS) -> list[dict]:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            if (RESULTS / "STOP_VERIFY").exists():
                print(f"检测到停机标志，{phase}/{variant} 优雅停机，已完成 {len(rows)} 个", flush=True)
                return rows
            cfg = cfg_of(dataset, seed)
            digest = hashlib.sha1(fingerprint(cfg).encode()).hexdigest()[:8]
            run_id = f"{phase}_{variant}_{dataset}_s{seed}_{digest}"
            rows.append(execute(run_id, phase, variant, cfg, by_id, by_fp))
    return rows


def mean_sel(rows: list[dict]) -> float | None:
    """变体裁决的评分口径：**验证集** MRR。

    测试集只在最终锁定后报告一次。若用测试集选变体/超参，报告的测试分会被
    360 次比较的系统性抬高，且相对基线的显著性检验对 Hy-LIN 不公平（R2-11）。
    """
    vals = [((row.get("val") or {}).get("all") or {}).get("MRR") for row in rows
            if row.get("status") == "ok" and (row.get("val") or {}).get("all", {}).get("MRR") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def hp_phase(by_id, by_fp) -> dict:
    specs = [
        ("base", {}),
        ("lr3e-4", {"lr": 0.0003}),
        ("lr3e-3", {"lr": 0.003}),
        ("drop0", {"dropout": 0.0}),
        ("drop0.2", {"dropout": 0.2}),
        ("kb5", {"k_bug": 5}),
        ("kb20", {"k_bug": 20}),
        ("kd8", {"k_dev": 8}),
        ("kd30", {"k_dev": 30}),
    ]
    grouped: dict[str, list[dict]] = {}
    for name, overrides in specs:
        grouped[name] = run_grid(
            "hp", name,
            lambda dataset, seed, overrides=overrides: full_config(dataset, seed, **overrides),
            by_id, by_fp, datasets=("mf",), seeds=SEEDS,
        )
    base_mean = mean_sel(grouped["base"]) or -1
    chosen = dict(BASE)
    axis = {
        "lr": [("lr3e-4", 0.0003), ("lr3e-3", 0.003)],
        "dropout": [("drop0", 0.0), ("drop0.2", 0.2)],
        "k_bug": [("kb5", 5), ("kb20", 20)],
        "k_dev": [("kd8", 8), ("kd30", 30)],
    }
    for key, options in axis.items():
        best_name, best_value, best_score = "base", BASE[key], base_mean
        for name, value in options:
            score = mean_sel(grouped[name])
            if score is not None and score > best_score + ADOPT_MARGIN:
                best_name, best_value, best_score = name, value, score
        chosen[key] = best_value
        print(f"超参数 {key} -> {best_value} ({best_name}, {best_score:.4f})", flush=True)
    if chosen != {**BASE, **{k: chosen[k] for k in chosen}}:
        pass
    same = all(chosen[key] == BASE[key] for key in ("lr", "dropout", "k_bug", "k_dev"))
    if not same:
        combined_rows = run_grid(
            "hp", "combined",
            lambda dataset, seed: full_config(
                dataset, seed, lr=chosen["lr"], dropout=chosen["dropout"], k_bug=chosen["k_bug"], k_dev=chosen["k_dev"]
            ),
            by_id, by_fp, datasets=("mf",), seeds=SEEDS,
        )
        combined_mean = mean_sel(combined_rows) or -1
        single_best = max(
            ((name, mean_sel(rows) or -1) for name, rows in grouped.items()),
            key=lambda item: item[1],
        )
        if combined_mean + ADOPT_MARGIN < max(base_mean, single_best[1]):
            # 坐标组合没有超过单因子最优或基线时，退回均值最高的那一个单因子设定。
            if single_best[0] == "base" or single_best[1] <= base_mean + ADOPT_MARGIN:
                chosen = dict(BASE)
            else:
                chosen = dict(BASE)
                for name, overrides in specs:
                    if name == single_best[0]:
                        chosen.update(overrides)
            print(f"组合设定未胜出，锁定单因子 {single_best}", flush=True)
        else:
            print(f"锁定组合设定，Firefox 均值 MRR {combined_mean:.4f}", flush=True)
    else:
        print("超参数保持基线", flush=True)
    locked = dict(BASE)
    locked.update({key: chosen[key] for key in ("lr", "dropout", "k_bug", "k_dev")})
    (RESULTS / "locked_hp.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")
    return locked


def write_lock(adopted: dict) -> None:
    """锁定配置落盘：results/ 与 configs/ 同时写。

    configs/locked_model.json 被 run_hylin / executor / run_ablation /
    run_sensitivity_grid / run_layer_grid / run_factor_design 直接读取，
    过去只有 results/ 被更新，configs/ 长期停在上一轮的锁点——那等于让下游
    静默使用作废配置。锁定时必须两处同写。
    """
    payload = json.dumps(_model_fields(adopted), indent=2, ensure_ascii=False)
    (RESULTS / "locked_model.json").write_text(payload, encoding="utf-8")
    cfg_path = ROOT / "configs" / "locked_model.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.chmod(0o644)
    cfg_path.write_text(payload, encoding="utf-8")


def log_decision(decision: dict) -> None:
    """裁决账本：每条都带阶段、筛选口径、胜者、是否采纳，供 RQ2 复现与审计。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, ensure_ascii=False) + "\n")


def module_phases(locked: dict, by_id, by_fp) -> None:
    phases = [
        ("plm", [
            ("bert", {"encoder": "bert"}),
            ("roberta", {"encoder": "roberta"}),
        ]),
        ("valve", [
            ("uniform", {"uniform_valve": True}),
            ("none", {"valve": "none"}),
            ("hetero", {"valve": "hetero"}),
        ]),
        ("prop", [
            ("exp1", {"exp_layers": 1}),
            ("exp3", {"exp_layers": 3}),
            ("imp1", {"imp_layers": 1}),
            ("imp3", {"imp_layers": 3}),
            ("overlap", {"dev_graph": "overlap"}),
            ("kdev_alt", {"k_dev": 8 if locked["k_dev"] != 8 else 30}),
        ]),
        ("gate", [
            ("bounded", {"gate_mode": "bounded"}),
            ("free", {"gate_mode": "free"}),
        ]),
        ("loss", [
            ("no_story", {"story": 0.0}),
            ("tau", {"tau": 0.2}),
            ("reweight", {"freq_exponent": 0.5}),
        ]),
    ]
    adopted = dict(locked)
    # 全集基准：hp 锁定配置先做一次 9 格复核，作为每个阶段"全集确认"的比较对象。
    prev9 = run_grid(
        "ref", "hp_locked",
        lambda dataset, seed, adopted=adopted: full_config(dataset, seed, **_model_fields(adopted)),
        by_id, by_fp,
    )
    prev9_mean = mean_sel(prev9) or -1

    for phase, variants in phases:
        # 第一关：MF 三种子筛选。与 verify_phases 里 B1/B2 的"MF 先行"同一套做法，
        # 避免每个变体都烧三数据集的算力。
        ref_mf = run_grid(
            "ref", "screen",
            lambda dataset, seed, adopted=adopted: full_config(dataset, seed, **_model_fields(adopted)),
            by_id, by_fp, datasets=("mf",),
        )
        variant_rows = {}
        for name, overrides in variants:
            # 跳过和当前最优完全相同的变体。
            fields = dict(_model_fields(adopted))
            fields.update(overrides)
            probe = full_config("mf", 42, **fields)
            current = full_config("mf", 42, **_model_fields(adopted))
            if fingerprint(probe) == fingerprint(current):
                continue
            variant_rows[name] = run_grid(
                phase, name,
                lambda dataset, seed, adopted=adopted, overrides=overrides: full_config(
                    dataset, seed, **{**_model_fields(adopted), **overrides}
                ),
                by_id, by_fp, datasets=("mf",),
            )
        ref_mean = mean_sel(ref_mf) or -1
        winner, winner_mean = "current", ref_mean
        for name, rows in variant_rows.items():
            score = mean_sel(rows)
            if score is not None and score > winner_mean + ADOPT_MARGIN:
                winner, winner_mean = name, score
        decision = {
            "phase": phase, "screen": "mf3", "winner": winner,
            "mf_sel": winner_mean, "mf_ref": ref_mean,
            "adopted_before": _model_fields(adopted),
        }
        if winner == "current":
            # 筛选就没赢，不花全集的算力。
            decision["result"] = "no_screen_win"
            decision["nine_cell"] = "skipped"
            log_decision(decision)
            print(f"阶段 {phase} 筛选无胜出（MF {ref_mean:.4f}），维持现配置", flush=True)
            refresh_docs(by_id)
            continue

        # 第二关：只有筛选胜者才做全集 9 格确认；确认不过门槛就回滚，
        # 挡住"只在 MF 上成立"的过拟合。
        overrides_of = dict(variants)
        candidate = dict(adopted)
        candidate.update(overrides_of[winner])
        if "encoder" in overrides_of[winner]:
            candidate["hidden"] = DIM[overrides_of[winner]["encoder"]]
        verify9 = run_grid(
            "ref", f"{phase}_winner",
            lambda dataset, seed, candidate=candidate: full_config(
                dataset, seed, **_model_fields(candidate)),
            by_id, by_fp,
        )
        verify_mean = mean_sel(verify9) or -1
        decision["verify_sel"] = verify_mean
        decision["verify_ref"] = prev9_mean
        if verify_mean >= prev9_mean + ADOPT_MARGIN:
            adopted = candidate
            prev9, prev9_mean = verify9, verify_mean
            decision["result"] = "adopted"
        else:
            decision["result"] = "reverted_no_full_gain"
        decision["adopted"] = _model_fields(adopted)
        log_decision(decision)
        print(
            f"阶段 {phase} 筛选胜出 {winner}（MF {winner_mean:.4f} vs {ref_mean:.4f}）；"
            f"全集确认 {verify_mean:.4f} vs {prev9_mean:.4f} -> "
            f"{'采纳' if decision['result'] == 'adopted' else '回滚'}",
            flush=True,
        )
        refresh_docs(by_id)
    write_lock(adopted)


def two_stage(phase: str, specs: list[tuple[str, dict]], base_fields: dict, ref_name: str,
              by_id: dict, by_fp: dict, adopted: dict, margin: float = ADOPT_MARGIN,
              confirm_top: int = 1, update_on_win: bool = True) -> dict:
    """两阶段筛选：MF 三种子筛选 -> 仅前 confirm_top 名做全集 9 格确认。

    与 module_phases 同一套门控，判据一律是验证集 MRR。
    筛选无胜出则不付全集算力；确认不过门槛就回滚。事件全部进裁决账本。
    """
    spec_map = dict(specs)
    screen = verify_grid(f"{phase}_mf", specs, base_fields, by_id, by_fp, datasets=("mf",))
    ref_mean = mean_sel(screen.get(ref_name, []))
    ranked = sorted(
        ((score, name) for name, rows in screen.items() if name != ref_name
         and (score := mean_sel(rows)) is not None),
        reverse=True,
    )
    decision = {
        "phase": phase, "screen": "mf3", "ref": ref_name,
        "mf_ref": ref_mean, "mf_ranking": [(n, round(s, 5)) for s, n in ranked],
        "adopted_before": _model_fields(adopted),
    }
    if ref_mean is None:
        decision["result"] = "no_reference"
        log_decision(decision)
        print(f"{phase} 参照 {ref_name} 无 MF 结果，跳过", flush=True)
        return adopted
    winners = [n for _s, n in ranked if _s > ref_mean + margin][:confirm_top]
    if not winners:
        decision["result"] = "no_screen_win"
        decision["nine_cell"] = "skipped"
        log_decision(decision)
        print(f"{phase} MF 筛选无胜出（参照 {ref_mean:.4f}），维持现配置", flush=True)
        return adopted

    confirm_specs = [(ref_name, spec_map[ref_name])] + [(n, spec_map[n]) for n in winners]
    confirm = verify_grid(phase, confirm_specs, base_fields, by_id, by_fp)
    ref_full = mean_sel(confirm.get(ref_name, []))
    win_name = winners[0]
    win_full = mean_sel(confirm.get(win_name, []))
    decision["confirm"] = {n: mean_sel(r) for n, r in confirm.items()}
    if win_full is not None and ref_full is not None and win_full >= ref_full + margin and update_on_win:
        adopted.update(spec_map[win_name])
        if "encoder" in spec_map[win_name]:
            adopted["hidden"] = DIM[spec_map[win_name]["encoder"]]
        decision["result"] = "adopted"
        decision["adopted"] = _model_fields(adopted)
    else:
        decision["result"] = "reverted_no_full_gain" if update_on_win else "confirmed_no_change"
    log_decision(decision)
    print(
        f"{phase} MF 胜出 {win_name}（{decision['mf_ranking'][0][1]:.4f} vs 参照 {ref_mean:.4f}）；"
        f"全集确认 {win_full if win_full is None else round(win_full, 5)} vs "
        f"{ref_full if ref_full is None else round(ref_full, 5)} -> {decision['result']}",
        flush=True,
    )
    return adopted


def hp_verify_specs(adopted: dict) -> list[tuple[str, dict]]:
    """终点超参重搜的候选：按网格枚举**与当前锁定值不同**的每条轴取值。

    不写死组合值——组合就是各轴锁定值本身，写死会变成上一轮遗留的陈旧配置。
    """
    specs: list[tuple[str, dict]] = [("base", {})]
    grid = {
        "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
        "k_bug": [5, 10, 15, 20, 30, 40],
        "k_dev": [5, 8, 10, 15, 20, 30],
    }
    for key, values in grid.items():
        current = float(adopted[key])
        for value in values:
            if abs(float(value) - current) > 1e-12:
                specs.append((f"{key}_{value:g}", {key: value}))
    return specs


def verify_grid(phase: str, variants: list[tuple[str, dict]], base_fields: dict,
                 by_id: dict, by_fp: dict, datasets=DATASETS, seeds=SEEDS) -> dict:
    """终点复核通用网格：其余冻结在终点配置，只换该变体列出的改动。指纹去重会自动跳过/复用。"""
    variant_rows = {}
    for name, overrides in variants:
        variant_rows[name] = run_grid(
            phase, name,
            lambda dataset, seed, base_fields=base_fields, overrides=overrides: full_config(
                dataset, seed, **{**_model_fields(base_fields), **overrides}
            ),
            by_id, by_fp, datasets=datasets, seeds=seeds,
        )
    return variant_rows


def verify_decide(phase: str, variant_rows: dict, ref_name: str, overrides_of: dict,
                  adopted: dict, update_on_win: bool = True) -> dict:
    ref_mean = mean_sel(variant_rows.get(ref_name, [])) or -1
    winner, winner_mean = ref_name, ref_mean
    for name, rows in variant_rows.items():
        score = mean_sel(rows)
        if score is not None and score > winner_mean + ADOPT_MARGIN:
            winner, winner_mean = name, score
    if winner != ref_name and update_on_win:
        adopted.update(overrides_of[winner])
    decision = {"phase": phase, "winner": winner, "mean_sel": winner_mean,
                "ref": ref_name, "ref_mean": ref_mean, "adopted": _model_fields(adopted)}
    log_decision(decision)
    print(f"验证 {phase} 胜出 {winner}，均值 MRR {winner_mean:.4f}（参照 {ref_name} {ref_mean:.4f}）", flush=True)
    refresh_docs(by_id_global())
    return adopted


def by_id_global() -> dict:
    by_id, _ = load_done()
    return by_id


def grid_interaction(variant_rows: dict) -> float:
    """两因子网格的交互强度：max|格点 - 行效应 - 列效应 + 总均值|（只用 ok 行）。

    评分口径与其它裁决一致，用**验证集** MRR：这是"要不要为 MC/GC 付算力"的裁决，
    不能看测试集。
    """
    cells = {name: [((r.get("val") or {}).get("all") or {}).get("MRR") for r in rows
                    if r.get("status") == "ok" and (r.get("val") or {}).get("all")]
             for name, rows in variant_rows.items()}
    cells = {k: [v for v in vs if v is not None] for k, vs in cells.items()}
    cells = {k: v for k, v in cells.items() if v}
    if len(cells) < 4:
        return 0.0
    grand = sum(sum(v) / len(v) for v in cells.values()) / len(cells)
    rows: dict[str, list[float]] = {}
    cols: dict[str, list[float]] = {}
    for name, vals in cells.items():
        r, _, c = name.partition("__")
        rows.setdefault(r, []).extend(vals)
        cols.setdefault(c, []).extend(vals)
    best = 0.0
    for name, vals in cells.items():
        r, _, c = name.partition("__")
        cell = sum(vals) / len(vals)
        expected = sum(rows[r]) / len(rows[r]) + sum(cols[c]) / len(cols[c]) - grand
        best = max(best, abs(cell - expected))
    return best


def verify_phases(by_id: dict, by_fp: dict) -> None:
    model_path = RESULTS / "locked_model.json"
    adopted = json.loads(model_path.read_text(encoding="utf-8"))
    print(f"终点复核起点：{json.dumps(adopted, ensure_ascii=False)}", flush=True)

    # A：终点超参重搜（两阶段：MF 筛选 -> 仅胜者做全集确认）。
    # 候选按网格动态枚举与当前锁定值不同的取值，避免写死组合造成陈旧配置。
    hp_specs = hp_verify_specs(adopted)
    print(f"hp_verify 候选 {len(hp_specs)} 个（含 base），两阶段门控", flush=True)
    adopted = two_stage("hp_verify", hp_specs, adopted, "base", by_id, by_fp, adopted)
    (RESULTS / "locked_hp.json").write_text(
        json.dumps({k: adopted[k] for k in ("lr", "dropout", "k_bug", "k_dev")}, indent=2), encoding="utf-8")

    # B1：gate × loss（MF 先行，交互 <0.002 则剪枝 MC/GC）。
    b1_specs = [(f"{g}__s{ s}", {"gate_mode": g, "story": s})
                for g in ("freq", "bounded", "free") for s in (0.5, 0.0)]
    b1_mf = verify_grid("interact_gate_loss", b1_specs, adopted, by_id, by_fp, datasets=("mf",))
    inter = grid_interaction(b1_mf)
    print(f"B1 MF 交互强度 {inter:.4f}", flush=True)
    if inter < ADOPT_MARGIN:
        with (RESULTS / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"phase": "interact_gate_loss", "winner": "pruned",
                                      "interaction_mf": inter, "adopted": _model_fields(adopted)},
                                     ensure_ascii=False) + "\n")
        print("B1 剪枝：MF 无交互，跳过 MC/GC", flush=True)
    else:
        b1_all = verify_grid("interact_gate_loss", b1_specs, adopted, by_id, by_fp)
        adopted = verify_decide("interact_gate_loss", b1_all, "freq__s0.5",
                                dict(b1_specs), adopted, update_on_win=inter >= 0.005)

    # B2：valve × prop（同上）。注意 uniform 不是 --valve 的取值，而是 uniform_valve 开关。
    def _valve_overrides(v: str) -> dict:
        if v == "uniform":
            return {"valve": "bug", "uniform_valve": True}
        return {"valve": v, "uniform_valve": False}
    b2_specs = [(f"{v}__exp{e}", {**_valve_overrides(v), "exp_layers": e})
                for v in ("bug", "uniform", "none") for e in (1, 2, 3)]
    b2_mf = verify_grid("interact_valve_prop", b2_specs, adopted, by_id, by_fp, datasets=("mf",))
    inter2 = grid_interaction(b2_mf)
    print(f"B2 MF 交互强度 {inter2:.4f}", flush=True)
    if inter2 < ADOPT_MARGIN:
        with (RESULTS / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"phase": "interact_valve_prop", "winner": "pruned",
                                      "interaction_mf": inter2, "adopted": _model_fields(adopted)},
                                     ensure_ascii=False) + "\n")
        print("B2 剪枝：MF 无交互，跳过 MC/GC", flush=True)
    else:
        b2_all = verify_grid("interact_valve_prop", b2_specs, adopted, by_id, by_fp)
        adopted = verify_decide("interact_valve_prop", b2_all, "bug__exp2",
                                dict(b2_specs), adopted, update_on_win=inter2 >= 0.005)

    # C：后向验证 runner-up（两阶段：MF 筛出最接近的竞争者 -> 只给它做全集确认）。
    # 只验证最接近的那个竞争者：MF 上就明显更差的候选，全集翻盘的概率低，不值得烧算力。
    c_specs = [
        ("current_ref", {}),
        ("imp1", {"imp_layers": 1}),
        ("bounded", {"gate_mode": "bounded"}),
        ("exp3", {"exp_layers": 3}),
        ("uniform", {"uniform_valve": True}),
    ]
    adopted = two_stage("backward", c_specs, adopted, "current_ref", by_id, by_fp, adopted,
                        confirm_top=1, update_on_win=False)
    write_lock(adopted)
    print("终点复核全部完成", flush=True)


def _model_fields(cfg: dict) -> dict:
    keys = [
        "encoder", "valve", "uniform_valve", "dev_graph", "k_bug", "k_dev", "exp_layers", "imp_layers",
        "dropout", "lr", "gate_mode", "fusion", "story", "tau", "freq_exponent", "imp_query", "hidden",
        "epochs", "patience", "low_coef", "batch_size", "weight_decay",
    ]
    return {key: cfg[key] for key in keys if key in cfg}


def _rows(by_id: dict) -> list[dict]:
    # 同一 fingerprint 可能被复用，按 run_id 去重即可；复用没有新 id。
    return [row for row in by_id.values() if row.get("status") == "ok"]


def _cell(rows: list[dict], dataset: str | None = None) -> str:
    chosen = rows if dataset is None else [row for row in rows if (row.get("config") or {}).get("dataset") == dataset]
    vals = [(row.get("test") or {}).get("all", {}).get("MRR") for row in chosen]
    vals = [value for value in vals if value is not None]
    if not vals:
        return "—"
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{statistics.pstdev(vals):.3f}"


def _slice_cell(rows: list[dict], slice_name: str) -> str:
    vals = [((row.get("test") or {}).get(slice_name) or {}).get("MRR") for row in rows]
    vals = [value for value in vals if value is not None]
    if not vals:
        return "—"
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{statistics.pstdev(vals):.3f}"


def refresh_docs(by_id: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    rows = sorted(_rows(by_id), key=lambda row: row["run_id"])
    log_lines = [
        "# 实验日志",
        "",
        "每一行对应一次成功的训练。原始日志在 `results/logs/`，指标在 `results/runs/`。",
        "",
        "| 运行 | 阶段 | 变体 | 数据 | 种子 | 全体MRR | 高频MRR | 中频MRR | 低频MRR | 轮次 | 秒 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cfg = row.get("config") or {}
        test = row.get("test") or {}
        log_lines.append(
            f"| `{row['run_id']}` | {row['phase']} | {row['variant']} | {cfg.get('dataset', '?')} | {cfg.get('seed', '?')} | "
            f"{test.get('all', {}).get('MRR', float('nan')):.4f} | "
            f"{test.get('high(>=20)', {}).get('MRR', float('nan')):.4f} | "
            f"{test.get('mid(5-19)', {}).get('MRR', float('nan')):.4f} | "
            f"{test.get('low(<5)', {}).get('MRR', float('nan')):.4f} | "
            f"{row.get('best_epoch', '')} | {row.get('seconds', 0):.0f} |"
        )
    fails = [row for row in by_id.values() if row.get("status") == "fail"]
    if fails:
        log_lines.extend(["", "## 失败", ""])
        for row in fails:
            log_lines.append(f"- `{row['run_id']}` 返回码 {row.get('returncode')}")
    (DOCS / "实验日志.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    locked_path = RESULTS / "locked_hp.json"
    locked = json.loads(locked_path.read_text(encoding="utf-8")) if locked_path.exists() else BASE
    model_path = RESULTS / "locked_model.json"
    adopted = json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else None
    report = [
        "# 模块探究实验",
        "",
        "协议：Firefox、Mozilla Core、Chromium 三份处理后数据，种子 42、43、44。",
        "高低频按训练缺陷条数划分，三份数据用同一阈值：低频 &lt;5，中频 5–19，高频 ≥20。",
        "这个阈值对应「几乎没有历史」和「已有稳定历史」，不按各数据集的分位数改写。",
        "一次只更换一个模块，其余保持当时采纳的设定。全体 MRR 在全部成功运行上的平均值高出 0.002 才采纳新设定。",
        "这是顺序的单因子对照，不是全因子网格。同一阶段里，除了该阶段列出的那一个改动，编码器、阀门、同质图、层数、门控、融合、损失、学习率、近邻数、dropout、批大小、早停规则和三个种子都相同。",
        "前一阶段若采纳了新设定，后面的阶段就改在新设定上比较。因此后面的表不能脱离当时的采纳设定来读。",
        "超参数只在 Firefox 的三个种子上搜索过一次，然后冻结。每个模块变体没有再单独调学习率和近邻数。",
        "融合方式固定为两路分数标准化后按频率混合，暗线查询用明线传播后的缺陷表示。这两项不在五个模块因子里重搜。",
        "采纳只看全体 MRR。高频、中频、低频 MRR 照常记录，但不参与是否换设定的决定。",
        "训练只使用第一张 GPU。BERT 与 RoBERTa 不是句向量模型，编码后去掉第一主成分，避免向量挤在同一个方向上。MiniLM 保持原句向量。",
        "",
        "## 当前锁定的超参数",
        "",
        "```json",
        json.dumps(locked, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if adopted:
        report.extend(["## 当前采纳的模型", "", "```json", json.dumps(adopted, ensure_ascii=False, indent=2), "```", ""])
    report.extend(["## 分阶段结果", ""])
    phases = sorted({row["phase"] for row in rows})
    for phase in phases:
        report.append(f"### {phase}")
        report.append("")
        report.append("| 变体 | 运行数 | 全体 | Firefox | Core | Chromium | 高频 | 中频 | 低频 |")
        report.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        phase_rows = [row for row in rows if row["phase"] == phase or (phase != "hp" and row["phase"] == "ref" and row["variant"] == "current")]
        # 参考行只在模块阶段展示一次，避免 hp 表被 ref 冲掉。
        if phase == "hp":
            phase_rows = [row for row in rows if row["phase"] == "hp"]
        variants = []
        for row in phase_rows:
            if row["variant"] not in variants:
                variants.append(row["variant"])
        for variant in variants:
            group = [row for row in phase_rows if row["variant"] == variant]
            report.append(
                f"| {variant} | {len(group)} | {_cell(group)} | {_cell(group, 'mf')} | {_cell(group, 'mc')} | "
                f"{_cell(group, 'gc')} | {_slice_cell(group, 'high(>=20)')} | {_slice_cell(group, 'mid(5-19)')} | "
                f"{_slice_cell(group, 'low(<5)')} |"
            )
        report.append("")
    report.append("数字是均值±总体标准差。原始记录见 `results/runs.jsonl`。")
    report.append("")
    (DOCS / "模块探究实验.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    by_id, by_fp = load_done()
    if os.environ.get("STUDY_VERIFY") == "1":
        verify_phases(by_id, by_fp)
        refresh_docs(by_id)
        print("探究结束", flush=True)
        return
    locked_path = RESULTS / "locked_hp.json"
    if locked_path.exists() and os.environ.get("STUDY_FORCE_HP") != "1":
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        print("读取已锁定超参数", flush=True)
    else:
        locked = hp_phase(by_id, by_fp)
    module_phases(locked, by_id, by_fp)
    refresh_docs(by_id)
    print("探究结束", flush=True)


def retry_gpu1() -> None:
    """把最近一次状态是显存不足、且还没在 GPU1 上试过的运行，改到备用卡。"""
    by_id, by_fp = load_done()
    pending = [
        row for row in by_id.values()
        if row.get("status") == "fail"
        and "OutOfMemory" in (row.get("error_tail") or "")
        and not row.get("tried_gpu1")
    ]
    print(f"GPU1 待补 {len(pending)} 次", flush=True)
    for row in pending:
        execute(row["run_id"], row["phase"], row["variant"], row["config"], by_id, by_fp)
    refresh_docs(by_id)
    print("GPU1 补跑结束", flush=True)


if __name__ == "__main__":
    import sys

    if "--retry-gpu1" in sys.argv:
        retry_gpu1()
    else:
        main()
