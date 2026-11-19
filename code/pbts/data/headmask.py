"""Head mask from the images only (never the labels), so the identical step runs on validation/test scans.

The background of these scans is NOT exactly zero (registration leaves faint noise), so "non-zero"
cropping keeps ~75% of the volume. Instead:
  1. scale each channel by its own 99th percentile (of voxels > 0), clip at 1.5, and sum the 4
  2. Otsu threshold on that sum (voxels > 0)        -> head vs background noise
  3. keep the largest 3D connected component
  4. binary closing (array padded first, otherwise closing erodes the outer slices)
  5. fill holes slice by slice (axial)
  6. already skull-stripped scans (exact-zero background, ~2 of 294) make Otsu split the brain itself,
     so if the non-zero region is < STRIPPED_RATIO x the Otsu head, use the filled non-zero region.
The mask is used only to place the crop box and to compute normalisation statistics. It never edits voxels.
Verified visually on all 294 training scans (analysis/head_box/montage_*.png).
"""
import numpy as np
from scipy import ndimage as ndi

from pbts import config as C


def _otsu(x: np.ndarray, bins: int = 256) -> float:
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _fill_axial(m: np.ndarray) -> np.ndarray:
    return np.stack([ndi.binary_fill_holes(m[:, :, z]) for z in range(m.shape[2])], axis=2)


def head_mask(imgs: np.ndarray):
    """imgs: [C, X, Y, Z] float. Returns (mask[X,Y,Z] bool, info dict)."""
    s = np.zeros(imgs.shape[1:], np.float32)
    for im in imgs:
        v = im[im > 0]
        if v.size:
            s += np.clip(im / np.percentile(v, 99), 0, 1.5)
    t = _otsu(s[s > 0])
    m = _largest_component(s > t)
    k = C.HEADMASK_CLOSING_ITERS + 1
    m = ndi.binary_closing(np.pad(m, k, mode="edge"), iterations=C.HEADMASK_CLOSING_ITERS)[k:-k, k:-k, k:-k]
    m = _fill_axial(m)

    nonzero = np.any(imgs != 0, axis=0)
    stripped = bool(nonzero.sum() < C.STRIPPED_RATIO * m.sum())
    if stripped:
        m = _fill_axial(_largest_component(nonzero))

    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0)
    info = {"otsu_threshold": t, "skull_stripped": stripped, "head_vox": int(m.sum()),
            "nonzero_vox": int(nonzero.sum()), "box_lo": lo.tolist(), "box_hi": hi.tolist()}
    return m, info
