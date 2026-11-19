"""Official BraTS 2026 PED scorer, run as-is (never re-implemented, so our numbers cannot drift from the
challenge's). It lives in its own environment (env_eval: BraTS_evaluation 0.0.10, panoptica 2.1.7) and is
called as a subprocess, in parallel chunks of cases.

Known behaviour of the official tool (verified on synthetic cases, analysis/official_edge10):
  - region absent in GT and prediction -> NaN (the case drops out of that region's mean)
  - region absent in GT, anything predicted -> DSC 0, NSD 0, HD95 373, one FP instance
  - whole-volume NSD uses a 0.5 mm tolerance, per-instance NSD uses 1 mm (voxel spacing)
  - 'lesionwise_*' columns average matched instances only (false alarms/misses appear only in F1/prec/rec)
"""
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pbts import config as C

EVAL_BIN = C.ROOT / "env_eval/bin"
OFFICIAL_VERSIONS = {"BraTS_evaluation": "0.0.10", "panoptica": "2.1.7", "config": "ped"}


def _run_chunk(cases, pred_dir, ref_dir, work):
    ref_d, pred_d = work / "ref", work / "pred"
    ref_d.mkdir(parents=True); pred_d.mkdir(parents=True)
    for c in cases:
        os.symlink(Path(ref_dir) / f"{c}.nii.gz", ref_d / f"{c}.nii.gz")
        os.symlink(Path(pred_dir) / f"{c}.nii.gz", pred_d / f"{c}.nii.gz")
    pan = work / "panoptica.json"
    log = open(work / "log.txt", "w")
    subprocess.check_call([str(EVAL_BIN / "brats-evaluate"), "--config", "ped", "--ref_path", str(ref_d),
                           "--pred_path", str(pred_d), "--summary_json", str(pan)], stdout=log, stderr=log)
    return json.load(open(pan))


def score_official(cases, pred_dir, ref_dir=None, out_dir=None, workers: int = 8):
    """Score predictions <pred_dir>/<case>.nii.gz against GT in the original space.
    Returns (per-case DataFrame from the official parser, raw panoptica JSON)."""
    ref_dir = Path(ref_dir or C.RAW_DIR / "labelsTr")
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="official_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="official_work_", dir=out_dir))
    chunks = [cases[i::workers] for i in range(workers) if cases[i::workers]]
    with ThreadPoolExecutor(len(chunks)) as ex:
        parts = list(ex.map(lambda a: _run_chunk(a[1], pred_dir, ref_dir, tmp / f"chunk{a[0]}"), enumerate(chunks)))
    merged = {"metrics": [m for p in parts for m in p["metrics"]],
              "missings": [m for p in parts for m in p.get("missings", [])]}
    pan = out_dir / "official_panoptica.json"
    json.dump(merged, open(pan, "w"))
    csv = out_dir / "official_scores.csv"
    subprocess.check_call([str(EVAL_BIN / "brats-parse-metrics"), "seg", "--json_path", str(pan),
                           "--output_csv_path", str(csv)], stdout=subprocess.DEVNULL)
    shutil.rmtree(tmp, ignore_errors=True)
    df = pd.read_csv(csv)
    df = df[~df.subject_id.isin(["mean", "std", "median"])].copy()
    df["case"] = df.subject_id.str.replace(".nii.gz", "", regex=False)
    json.dump(OFFICIAL_VERSIONS, open(out_dir / "official_versions.json", "w"))
    if merged["missings"]:
        raise RuntimeError(f"official scorer could not find predictions for {merged['missings']}")
    return df.drop(columns=["subject_id"]).set_index("case").loc[cases].reset_index(), merged
