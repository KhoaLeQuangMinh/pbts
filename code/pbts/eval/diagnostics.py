"""Per-case diagnostics computed on the final label maps (original 240x240x155 space).

For each region: presence outcome (is the region in the GT / in the prediction), volumes, the 2023 lesion-wise
scores, and per-lesion rows for every GT lesion and every false-positive predicted component (size, centroid,
distance to the nearest GT voxel of that region). Plus a 5x5 voxel confusion matrix of labels inside the
union of GT tumor and predicted tumor.
"""
import numpy as np
from scipy import ndimage as ndi

from pbts import config as C
from pbts.eval.lesionwise2023 import region_scores

REGION_NAMES = list(C.REGIONS)          # ET NET CC ED TC WT


def region_mask(seg: np.ndarray, region: str) -> np.ndarray:
    return np.isin(seg, C.REGIONS[region])


def presence_outcome(gt_present: bool, pred_present: bool) -> str:
    return {(True, True): "TP", (True, False): "FN", (False, True): "FP", (False, False): "TN"}[(gt_present, pred_present)]


def evaluate_case(case: str, pred: np.ndarray, gt: np.ndarray, spacing=C.SPACING):
    """Returns (region_rows, lesion_rows, confusion[5,5])."""
    assert pred.shape == gt.shape == C.IMAGE_SHAPE, (pred.shape, gt.shape)
    region_rows, lesion_rows = [], []
    for region in REGION_NAMES:
        p, g = region_mask(pred, region), region_mask(gt, region)
        summ, lesions, internals = region_scores(p, g, spacing)
        row = {"case": case, "region": region, **summ,
               "gt_present": bool(g.any()), "pred_present": bool(p.any()),
               "presence": presence_outcome(bool(g.any()), bool(p.any()))}
        region_rows.append(row)

        les_map = internals["lesions"]
        cents = ndi.center_of_mass(g, les_map, [l["gt_lesion"] for l in lesions]) if lesions else []
        for l, c in zip(lesions, cents):
            lesion_rows.append({"case": case, "region": region, "kind": "gt_lesion", "id": l["gt_lesion"],
                                "vol": l["gt_vol"], "cx": c[0], "cy": c[1], "cz": c[2],
                                "detected": l["detected"], "counted_2023": l["counted"],
                                "dice": l["dice"], "hd95": l["hd95"], "n_pred_matched": l["n_pred_matched"],
                                "dist_to_gt_mm": 0.0})
        fp_ids = internals["fp_ids"]
        if fp_ids:
            pcc = internals["pred_cc"]
            sizes = ndi.sum(p, pcc, fp_ids)
            cents = ndi.center_of_mass(p, pcc, fp_ids)
            dist = ndi.distance_transform_edt(~g, sampling=spacing) if g.any() else None
            for i, sz, c in zip(fp_ids, sizes, cents):
                d = float(dist[pcc == i].min()) if dist is not None else float("nan")
                lesion_rows.append({"case": case, "region": region, "kind": "pred_fp", "id": int(i),
                                    "vol": float(sz) * float(np.prod(spacing)), "cx": c[0], "cy": c[1], "cz": c[2],
                                    "detected": False, "counted_2023": True, "dice": 0.0, "hd95": 374.0,
                                    "n_pred_matched": 0, "dist_to_gt_mm": d})

    tumor = (gt > 0) | (pred > 0)
    conf = np.zeros((5, 5), np.int64)
    np.add.at(conf, (gt[tumor].astype(int), pred[tumor].astype(int)), 1)
    return region_rows, lesion_rows, conf
