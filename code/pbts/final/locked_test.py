"""THE FINAL, ONE-TIME evaluation on the 60 locked test scans (outer fold 0).

    python -m pbts.final.locked_test --models nnunet_resenc_l primus_v2b segmamba [--tag m=t ...] \
        [--cross_model_ensemble] --confirm "I UNDERSTAND THIS IS THE FINAL LOCKED-TEST EVALUATION"

For every model: each of its 5 fold models predicts the 60 locked scans (8-flip TTA), the 5 probability maps are
averaged (the model's final ensemble), and the result is scored ONCE with the shared evaluator. Every access is
appended to data/locked_test/ACCESS_LOG.txt by the guard. A second run is refused unless --rerun_reason is given
(it is then logged, and should be reported as such).
Outputs: runs/final_locked/<model>/{fold_k,ensemble5}/..., runs/final_locked/FINAL_DONE.json
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from pbts import config as C

CONFIRM = "I UNDERSTAND THIS IS THE FINAL LOCKED-TEST EVALUATION"
OUT = C.RUNS_DIR / "final_locked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tag", nargs="*", default=[])
    ap.add_argument("--cross_model_ensemble", action="store_true")
    ap.add_argument("--confirm", required=True)
    ap.add_argument("--rerun_reason", default=None)
    a = ap.parse_args()
    if a.confirm != CONFIRM:
        raise SystemExit(f'--confirm must be exactly: "{CONFIRM}"')
    done = OUT / "FINAL_DONE.json"
    if done.exists() and not a.rerun_reason:
        raise SystemExit(f"{done} exists: the locked test was already evaluated. Use --rerun_reason to override (logged).")

    from pbts.analysis.collect import run_dirs
    from pbts.eval.evaluate import evaluate
    from pbts.final.ensemble import average_probs
    from pbts.final.netload import load_network
    from pbts.infer.predict import predict_cases, write_outputs
    tags = dict(t.split("=") for t in a.tag)
    locked = json.load(open(C.LOCKED_DIR / "cases.json"))["locked"]
    purpose = f"FINAL locked-test evaluation of {a.models}" + (f" (RERUN: {a.rerun_reason})" if a.rerun_reason else "")

    for m in a.models:                                     # refuse before touching anything if a run is unfinished
        for k, d in run_dirs(m, tags.get(m, "default")).items():
            if not (d / "COMPLETE.json").exists():
                raise SystemExit(f"{m} fold {k} is not complete ({d}); the locked test needs all 5 folds of every model")

    summary = {"models": a.models, "tags": tags, "cases": len(locked), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "rerun_reason": a.rerun_reason,
               "code_commit": json.load(open(C.ROOT / "code/VERSION.json")).get("commit") if (C.ROOT / "code/VERSION.json").exists() else None}
    for m in a.models:
        for k, d in run_dirs(m, tags.get(m, "default")).items():
            fold_out = OUT / m / f"fold_{k}"
            if (fold_out / "prediction_timing.csv").exists():
                continue
            net, frame, _ = load_network(d)
            predict_cases(net, locked, fold_out, tta=True, save_probs=True, allow_locked=True, purpose=purpose,
                          frame_xyz=frame)
            del net
        ens = OUT / m / "ensemble5"
        for case in locked:
            probs, _ = average_probs([OUT / m / f"fold_{k}" / "probabilities" / f"{case}.npz" for k in range(C.N_INNER)])
            write_outputs(case, probs, json.load(open(C.HEAD_DIR / "records" / f"{case}.json")), ens, save_probs=True)
        df = evaluate(locked, ens / "predictions", ens, allow_locked=True, purpose=purpose)
        summary[m] = df.groupby("region")[["off_global_dsc", "off_all_instance_f1", "lw2023_dice"]].mean().round(4).to_dict()

    if a.cross_model_ensemble and len(a.models) > 1:
        ens = OUT / ("ensemble__" + "__".join(a.models))
        for case in locked:
            probs, _ = average_probs([OUT / m / "ensemble5" / "probabilities" / f"{case}.npz" for m in a.models])
            write_outputs(case, probs, json.load(open(C.HEAD_DIR / "records" / f"{case}.json")), ens, save_probs=True)
        df = evaluate(locked, ens / "predictions", ens, allow_locked=True, purpose=purpose + " (cross-model ensemble)")
        summary["cross_model_ensemble"] = df.groupby("region")[["off_global_dsc", "off_all_instance_f1", "lw2023_dice"]].mean().round(4).to_dict()

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(done, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
