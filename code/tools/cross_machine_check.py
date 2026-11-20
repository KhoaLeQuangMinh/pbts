"""Same trained weights, another machine: predict the run's validation scans here and compare with the predictions
the original machine saved. Minutes of GPU, no training. Checks data preparation + environment + prediction code
end to end (2026-11-21: ResEnc fold 0 trained on gpu-a240, re-predicted on blackwell).

    python tools/cross_machine_check.py <dir with run_config.json + checkpoints/weights_100.pth> <reference predictions dir> <out dir>

Writes <out>/predictions/*.nii.gz and <out>/cross_machine.csv (per case: voxels that differ, Dice between the two
machines' predictions per region).
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbts import config as C  # noqa: E402
from pbts.final.netload import load_network  # noqa: E402
from pbts.infer.predict import predict_cases  # noqa: E402


def main(run, ref, out):
    net, frame, cfg = load_network(run)
    cases = cfg["val_cases"]
    predict_cases(net, cases, out, tta=True, save_probs=False, frame_xyz=frame,
                  log=lambda m: print(m, flush=True))
    rows = []
    for c in cases:
        a = np.asanyarray(nib.load(str(out / "predictions" / f"{c}.nii.gz")).dataobj)
        b = np.asanyarray(nib.load(str(ref / f"{c}.nii.gz")).dataobj)
        row = {"case": c, "voxels_differing": int((a != b).sum()), "tumour_voxels": int(((a > 0) | (b > 0)).sum())}
        for r, labs in C.REGIONS.items():
            pa, pb = np.isin(a, labs), np.isin(b, labs)
            s = int(pa.sum() + pb.sum())
            row[f"{r}_dice_between_machines"] = 1.0 if s == 0 else 2 * int((pa & pb).sum()) / s
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "cross_machine.csv", index=False)
    print(f"{len(df)} cases | identical predictions: {(df.voxels_differing == 0).sum()} | voxels differing: median "
          f"{df.voxels_differing.median():.0f}, max {df.voxels_differing.max()} "
          f"(of median {df.tumour_voxels.median():.0f} tumour voxels)")
    print(df.filter(like="dice_between").agg(["mean", "min"]).T.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
