"""Runs the ORIGINAL BraTS-2023-Metrics functions (in env_ref2023: its own pinned requirements, numpy 1.24.3) on binary region masks, for comparison.
Usage: env_ref2023/bin/python -I tests/ref2023_runner.py <jobs.json> <out.json>
jobs: list of {"pred": path, "gt": path, "key": str}. Each file holds ONE region encoded as label 3."""
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

from pathlib import Path  # noqa: E402

# project root = folder holding code/ (this file is ROOT/code/tests/ref2023_runner.py); runs with -I, no pbts import
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repos/BraTS-2023-Metrics"))
import metrics as M  # noqa: E402  (original code, unmodified)


def run(job):
    try:
        return _run(job)
    except Exception as e:  # report, don't hide: the test decides what to do with it
        import traceback
        return {"key": job["key"], "error": repr(e), "trace": traceback.format_exc()[-600:]}


def _run(job):
    tp, fn, fp, gt_tp, pairs, full_dice, full_hd95, *_ = M.get_LesionWiseScores(job["pred"], job["gt"], "ET", 3)
    thresh = 50
    rows = [(len(p), v, d, (374.0 if np.isinf(h) else h)) for p, _, v, d, h in pairs]
    counted = [r for r in rows if r[1] > thresh]
    fn_sub = sum(1 for r in rows if r[0] == 0 and r[1] <= thresh)
    tp_sub = sum(1 for r in rows if r[0] != 0 and r[1] <= thresh)
    denom = len(counted) + len(fp)
    lw_dice = sum(r[2] for r in counted) / denom if denom else float("nan")
    lw_hd = (sum(r[3] for r in counted) + 374 * len(fp)) / denom if denom else float("nan")
    lw_dice = 1.0 if math.isnan(lw_dice) else lw_dice
    lw_hd = 0.0 if math.isnan(lw_hd) else lw_hd
    return {"key": job["key"], "lw2023_dice": float(lw_dice), "lw2023_hd95": float(lw_hd),
            "lw2023_tp": len(gt_tp) - tp_sub, "lw2023_fn": len(fn) - fn_sub, "lw2023_fp": int(len(fp)),
            "legacy_dice": float(full_dice), "legacy_hd95": float(full_hd95)}


if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1]))
    with Pool(16) as pool:
        out = pool.map(run, jobs)
    json.dump(out, open(sys.argv[2], "w"))
