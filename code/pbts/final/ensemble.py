"""Probability-averaging ensembles (analysis only; never changes any model's own results).

Cross-model ensemble on the development set (out-of-fold):
    python -m pbts.final.ensemble --name resenc_primus_segmamba --models nnunet_resenc_l primus_v2b segmamba
For each case, the probability maps of every model's run whose validation fold holds that case are averaged,
then argmax. Results are written like a model's runs (runs/ensemble_<name>/default/fold_k/eval/ckpt100/), so the
comparison dashboard can include the ensemble as one more column.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pbts import config as C


def average_probs(files):
    probs, box = None, None
    for f in files:
        z = np.load(f)
        b = (tuple(z["box_lo"]), tuple(z["box_hi"]))
        if box is not None and b != box:
            raise ValueError(f"{f}: box {b} != {box}")
        box = b
        p = z["probs"].astype(np.float32)
        probs = p if probs is None else probs + p
    return probs / len(files), box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tag", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from pbts.analysis.collect import collect_model
    from pbts.eval.evaluate import evaluate
    from pbts.infer.predict import write_outputs
    tags = dict(t.split("=") for t in a.tag)
    infos = {m: collect_model(m, tags.get(m, "default"), require_all=True)[3] for m in a.models}
    split = json.load(open(C.HEAD_DIR / "study_split.json"))
    for k, fold in enumerate(split["inner_folds"]):
        out = C.RUNS_DIR / f"ensemble_{a.name}" / "default" / f"fold_{k}" / "eval" / "ckpt100"
        for case in fold["val"]:
            files = [Path(infos[m]["prob_dirs"][k]) / f"{case}.npz" for m in a.models]
            probs, _ = average_probs(files)
            rec = json.load(open(C.HEAD_DIR / "records" / f"{case}.json"))
            write_outputs(case, probs, rec, out, save_probs=True)
        evaluate(fold["val"], out / "predictions", out, workers=a.workers)
        json.dump({"members": a.models, "tags": tags, "rule": "mean of softmax probabilities, then argmax"},
                  open(out.parents[1] / "ensemble.json", "w"), indent=1)
        print("fold", k, "done")


if __name__ == "__main__":
    main()
