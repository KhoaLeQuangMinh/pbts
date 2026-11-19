"""Collect a model's out-of-fold results: its 5 inner-fold runs together cover all 234 development cases once."""
import json
from pathlib import Path

import pandas as pd

from pbts import config as C


def run_dirs(model: str, tag: str = "default"):
    return {k: C.RUNS_DIR / model / tag / f"fold_{k}" for k in range(C.N_INNER)}


def collect_model(model: str, tag: str = "default", ckpt: int = 100, require_all=False):
    """Returns (per_case_region, per_lesion, confusion_sum, info). Rows carry model/fold columns."""
    pcr, les, conf, info = [], [], None, {"model": model, "tag": tag, "folds_done": [], "folds_missing": []}
    for k, d in run_dirs(model, tag).items():
        e = d / "eval" / f"ckpt{ckpt:03d}"
        if not (e / "EVAL_DONE.json").exists():
            info["folds_missing"].append(k)
            continue
        info["folds_done"].append(k)
        a = pd.read_csv(e / "per_case_region.csv"); a["fold"] = k; pcr.append(a)
        b = pd.read_csv(e / "per_lesion.csv"); b["fold"] = k; les.append(b)
        c = pd.read_csv(e / "confusion.csv", index_col=0)
        conf = c if conf is None else conf + c
        info.setdefault("pred_dirs", {})[k] = str(e / "predictions")
        info.setdefault("prob_dirs", {})[k] = str(e / "probabilities")
        cost = _cost(d)
        for kk, v in cost.items():
            info.setdefault("cost", {}).setdefault(kk, []).append(v)
    if require_all and info["folds_missing"]:
        raise RuntimeError(f"{model}/{tag}: folds missing {info['folds_missing']}")
    if not pcr:
        return None, None, None, info
    pcr, les = pd.concat(pcr, ignore_index=True), pd.concat(les, ignore_index=True)
    pcr["model"] = les["model"] = model
    dup = pcr.groupby(["case", "region"]).size()
    if (dup > 1).any():
        raise RuntimeError(f"{model}: a case appears in more than one fold's validation set")
    return pcr, les, conf, info


def _cost(run_dir: Path):
    out = {}
    ep = run_dir / "epochs.jsonl"
    if ep.exists():
        L = [json.loads(l) for l in open(ep) if l.strip()]
        out["train_gpu_h"] = sum(r.get("epoch_s", 0) for r in L) / 3600
        out["peak_GB"] = max((r.get("gpu_peak_GB", 0) for r in L), default=0)
    ms = run_dir / "provenance" / "model_summary.json"
    if ms.exists():
        out["params_M"] = json.load(open(ms))["params"] / 1e6
    t = run_dir / "eval" / "ckpt100" / "prediction_timing.csv"
    if t.exists():
        out["predict_s_per_case"] = float(pd.read_csv(t).seconds.mean())
    return out
