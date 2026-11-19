"""BraTS 2023-2025 lesion-wise metric (Saluja et al., github.com/rachitsaluja/BraTS-2023-Metrics @ 43c9052),
re-implemented for arbitrary binary region masks so it covers all 6 pediatric regions (the original only
knows the adult WT/TC/ET label mapping). Behaviour is meant to be IDENTICAL to the original and is tested
against it (tests/test_eval.py); only speed differs (dilations are done on cropped boxes).

Per region, for the PED setting (dilation 3, small-lesion threshold 50 voxels):
  - GT lesions = 26-connected components of GT, grouped together when their 3x-dilations touch
  - a predicted component is matched to a GT lesion if it overlaps that lesion's 3x-dilation
  - lesion Dice/HD95 = Dice/HD95 between the GT lesion and the union of its matched predicted components
  - FP = predicted components matched to no GT lesion
  - lesion-wise Dice = sum(Dice of GT lesions > 50 vox) / (n GT lesions > 50 vox + n FP);  1 if 0/0
  - lesion-wise HD95 = (sum HD95 + 374 * n FP) / (same denominator); inf -> 374;  0 if 0/0
"""
import numpy as np
import surface_distance
from scipy import ndimage as ndi

DIL_STRUCT = ndi.generate_binary_structure(3, 2)
CC26 = np.ones((3, 3, 3), bool)
MAX_HD = 374.0


def _cc26(m):
    lab, n = ndi.label(m, structure=CC26)
    return lab, n


def _dilate_local(mask_local, iters):
    """binary_dilation on a crop; exact w.r.t. the full image when the crop has >= iters voxels of margin
    around the mask (or touches the image border, where scipy's zero border matches the full image)."""
    return ndi.binary_dilation(mask_local, structure=DIL_STRUCT, iterations=iters)


def _grow(sl, shape, k):
    return tuple(slice(max(s.start - k, 0), min(s.stop + k, n)) for s, n in zip(sl, shape))


def _union(a, b):
    return tuple(slice(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(a, b))


def _hd95(gt, pred, spacing):
    sd = surface_distance.compute_surface_distances(gt.astype(bool), pred.astype(bool), spacing)
    return float(surface_distance.compute_robust_hausdorff(sd, 95))


def region_scores(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), dil: int = 3, thresh: float = 50):
    """pred, gt: boolean masks of ONE region.
    Returns (summary dict, list of per-GT-lesion dicts, internals {lesions, pred_cc, fp_ids} for diagnostics)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    vox = float(np.prod(spacing))
    shape = gt.shape

    # full-image ("legacy") Dice / HD95
    if not gt.any() and not pred.any():
        full_dice, full_hd95 = 1.0, 0.0
    else:
        full_dice = 2.0 * (gt & pred).sum() / (gt.sum() + pred.sum())
        full_hd95 = _hd95(gt, pred, spacing)

    gt_cc, n_gt_cc = _cc26(gt)
    pred_cc, n_pred = _cc26(pred)
    pred_sizes = np.bincount(pred_cc.ravel(), minlength=n_pred + 1)
    pred_boxes = ndi.find_objects(pred_cc)

    # group GT components by their 3x dilation (original: components of the dilated GT mask, intersected with GT)
    gt_dil = ndi.binary_dilation(gt, structure=DIL_STRUCT, iterations=dil) if gt.any() else gt
    gt_dil_cc, n_lesions = _cc26(gt_dil)
    lesions = np.where(gt, gt_dil_cc, 0)

    matched_pred, per_lesion = set(), []
    for k, box in enumerate(ndi.find_objects(lesions), start=1):
        if box is None:
            continue
        gbox = _grow(box, shape, dil + 1)
        g_local = lesions[gbox] == k
        g_dil_local = _dilate_local(g_local, dil)
        hits = np.unique(pred_cc[gbox][g_dil_local])
        hits = [int(h) for h in hits if h != 0]
        matched_pred.update(hits)
        gt_vol = float(g_local.sum()) * vox

        if hits:
            ubox = gbox
            for h in hits:
                ubox = _union(ubox, pred_boxes[h - 1])
            g_u = lesions[ubox] == k
            p_u = np.isin(pred_cc[ubox], hits)
            inter = float((g_u & p_u).sum())
            dsc = 2.0 * inter / (g_u.sum() + sum(pred_sizes[h] for h in hits))
            hd = _hd95(g_u, p_u, spacing)
        else:
            dsc, hd = 0.0, float("inf")   # original: dice(empty, gt) = 0, HD95 of empty prediction = inf
        per_lesion.append({"gt_lesion": k, "gt_vol": gt_vol, "n_pred_matched": len(hits),
                           "dice": dsc, "hd95": min(hd, MAX_HD) if np.isfinite(hd) else MAX_HD,
                           "detected": bool(hits), "counted": gt_vol > thresh})

    fp = [c for c in range(1, n_pred + 1) if c not in matched_pred]
    counted = [l for l in per_lesion if l["counted"]]
    denom = len(counted) + len(fp)
    if denom == 0:
        lw_dice, lw_hd95 = 1.0, 0.0
    else:
        lw_dice = sum(l["dice"] for l in counted) / denom
        lw_hd95 = (sum(l["hd95"] for l in counted) + MAX_HD * len(fp)) / denom

    summary = {
        "lw2023_dice": float(lw_dice), "lw2023_hd95": float(lw_hd95),
        "lw2023_tp": sum(1 for l in per_lesion if l["detected"] and l["counted"]),
        "lw2023_fn": sum(1 for l in per_lesion if not l["detected"] and l["counted"]),
        "lw2023_fp": len(fp),
        "legacy_dice": float(full_dice), "legacy_hd95": float(full_hd95),
        "gt_vol": float(gt.sum()) * vox, "pred_vol": float(pred.sum()) * vox,
    }
    return summary, per_lesion, {"lesions": lesions, "pred_cc": pred_cc, "fp_ids": fp}
