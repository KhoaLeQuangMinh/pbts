"""Monitor: fast scores on 10 fixed validation cases every few epochs, plus pictures of 3 of them.
Logging only: never used to choose anything (no checkpoint selection, no early stopping).

Scores are computed in the cropped box (cheap): per region whole-volume Dice, presence, number of
predicted pieces and of false-positive pieces (pieces touching no GT voxel of that region). The full official
evaluation runs only on the saved checkpoints after training.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from pbts import config as C
from pbts.infer.predict import Predictor, load_case_inputs

CC26 = np.ones((3, 3, 3), bool)
COLORS = {1: (0.2, 0.4, 1.0), 2: (1.0, 0.45, 0.2), 3: (0.1, 0.75, 0.5), 4: (0.95, 0.75, 0.1)}   # ET NET CC ED


def select_monitor_cases(val_cases, n=10):
    """Deterministic, spans the hard situations: CC+ED, CC only, ED only, neither, tiny/huge tumor, follow-up,
    no ET; filled up with evenly spaced remaining cases. The first 3 also get pictures."""
    m = pd.read_csv(C.MANIFEST).set_index("subject_id").loc[sorted(val_cases)]
    cc, ed, et = m.vox_CC > 0, m.vox_ED > 0, m.vox_ET > 0
    picks = []
    def take(mask, order=None):
        cand = m[mask & ~m.index.isin(picks)]
        if order is not None:
            cand = cand.sort_values(order)
        if len(cand):
            picks.append(cand.index[0])
    take(cc & ed); take(~cc & ~ed); take(m.timepoint >= 100)          # picture cases
    take(cc & ~ed); take(ed & ~cc); take(~et)
    take(pd.Series(True, m.index), "vox_WT")                           # smallest tumor
    big = m[~m.index.isin(picks)].sort_values("vox_WT", ascending=False)  # largest tumor
    if len(big):
        picks.append(big.index[0])
    rest = [c for c in m.index if c not in picks]
    while len(picks) < min(n, len(m)) and rest:
        picks.append(rest.pop(len(rest) // 2))
    return picks[:n]


_CACHE = {}


def _cached_case(case):
    """Monitor cases are re-used every few epochs: keep inputs and GT box labels in memory (~70 MB/case)."""
    if case not in _CACHE:
        x, rec = load_case_inputs(case)
        p = C.NNUNET_RAW / C.DATASET_NAME / "labelsTr" / f"{case}.nii.gz"
        _CACHE[case] = (x, rec, np.asanyarray(nib.load(str(p)).dataobj).astype(np.uint8))
    return _CACHE[case]


def _region_stats(pred, gt, region):
    labs = C.REGIONS[region]
    p, g = np.isin(pred, labs), np.isin(gt, labs)
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    lab, n = ndi.label(p, structure=CC26)
    n_fp = 0
    if n:
        touching = np.unique(lab[g & p])
        n_fp = n - int((touching > 0).sum())
    return {"region": region, "dice": float(2 * inter / denom) if denom else float("nan"),
            "gt_present": bool(g.any()), "pred_present": bool(p.any()),
            "gt_vox": int(g.sum()), "pred_vox": int(p.sum()), "n_pred_pieces": int(n), "n_fp_pieces": int(n_fp)}


def _picture(case, x, gt, pred, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tumor = gt > 0
    c = np.round(np.argwhere(tumor).mean(0)).astype(int) if tumor.any() else np.array(gt.shape) // 2
    flair = x[0]
    lo, hi = np.percentile(flair, [1, 99.5])
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    for row, (lab, name) in enumerate([(gt, "ground truth"), (pred, "prediction")]):
        for col, (sl, plane) in enumerate([((slice(None), slice(None), c[2]), "axial"),
                                           ((slice(None), c[1], slice(None)), "coronal"),
                                           ((c[0], slice(None), slice(None)), "sagittal")]):
            img = np.clip((flair[sl] - lo) / (hi - lo + 1e-6), 0, 1)
            rgb = np.stack([img] * 3, -1)
            l2 = lab[sl]
            for k, col_rgb in COLORS.items():
                m = l2 == k
                rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(col_rgb)
            ax[row, col].imshow(np.rot90(rgb))
            ax[row, col].set_axis_off()
            ax[row, col].set_title(f"{name} · {plane}", fontsize=8)
    fig.suptitle(title + "    ET blue · NET orange · CC green · ED yellow", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=70)
    plt.close(fig)


@torch.no_grad()
def run_monitor(network, cases, epoch, out_dir, n_pictures=3, device="cuda", frame_xyz=C.FRAME_XYZ):
    pred = Predictor(network, device=device, frame_xyz=frame_xyz)
    rows, images = [], []
    for i, case in enumerate(cases):
        x, rec, gt = _cached_case(case)
        logits = pred.predict_logits(x, rec["pad_value"], tta=False)
        lab = logits.argmax(0).astype(np.uint8)
        for region in C.REGIONS:
            rows.append({"epoch": epoch, "case": case, **_region_stats(lab, gt, region)})
        if i < n_pictures:
            path = Path(out_dir) / "monitor_png" / f"ep{epoch:04d}_{case}.png"
            _picture(case, x, gt, lab, path, f"{case} · epoch {epoch}")
            images.append(path)
    df = pd.DataFrame(rows)
    summary = {}
    for region, d in df.groupby("region"):
        present = d[d.gt_present]
        absent = d[~d.gt_present]
        summary[f"{region}_dice_when_present"] = float(present.dice.mean()) if len(present) else float("nan")
        summary[f"{region}_false_alarm_rate_when_absent"] = float(absent.pred_present.mean()) if len(absent) else float("nan")
        summary[f"{region}_fp_pieces"] = int(d.n_fp_pieces.sum())
    return rows, summary, images
