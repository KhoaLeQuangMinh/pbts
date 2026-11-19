"""Shared prediction routine: identical for every model.

  1. load the case's cropped, normalised 4-channel box (written by pbts.data.preprocess) and its record
  2. place it in the 192x224x160 frame, padding with the case's background value per channel; boxes larger
     than the frame (2 training scans) are covered by nnU-Net-style sliding windows (step 0.5, Gaussian weights)
  3. 8-way mirroring test-time augmentation; logits are averaged over flips (and windows), as in nnU-Net
  4. softmax -> probabilities (kept, float16, in box coordinates) -> argmax -> label map pasted into the
     original 240x240x155 space with the original header. No post-processing of any kind.
Axis order: our files are x,y,z (nibabel); networks were trained on nnU-Net arrays, which are z,y,x. The
transposition happens only here and in nnU-Net's own loader; tests/test_train.py checks they agree.
"""
import itertools
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from pbts import config as C
from pbts.data import io
from pbts.data.guard import check_not_locked


def load_case_inputs(case: str):
    rec = json.load(open(C.HEAD_DIR / "records" / f"{case}.json"))
    base = C.LOCKED_DIR / "images" if rec["locked"] else C.NNUNET_RAW / C.DATASET_NAME / "imagesTr"
    x = np.stack([nib.load(str(base / f"{case}_{c:04d}.nii.gz")).get_fdata(dtype=np.float32)
                  for c in range(len(C.CHANNELS))])
    return x, rec


def to_network_layout(x_xyz: np.ndarray) -> torch.Tensor:
    """[C, X, Y, Z] (our files) -> [C, Z, Y, X] (nnU-Net arrays, what every network is trained on)."""
    return torch.from_numpy(np.ascontiguousarray(x_xyz)).permute(0, 3, 2, 1).contiguous()


class Predictor:
    def __init__(self, network: torch.nn.Module, device="cuda", frame_xyz=C.FRAME_XYZ, n_classes=len(C.LABELS),
                 amp_dtype=torch.float16):
        self.net = network
        self.device = torch.device(device)
        self.frame_xyz = tuple(frame_xyz)
        self.n_classes = n_classes
        self.amp_dtype = amp_dtype

    def _forward(self, x):
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            out = self.net(x)
        if isinstance(out, (list, tuple)):          # deep supervision left on: full resolution is the first
            out = out[0]
        return out.float()

    def _tta(self, patch, tta: bool):
        out = self._forward(patch)
        if not tta:
            return out
        n = 1
        for k in (1, 2, 3):
            for axes in itertools.combinations((2, 3, 4), k):
                out += torch.flip(self._forward(torch.flip(patch, axes)), axes)
                n += 1
        return out / n

    @torch.no_grad()
    def predict_logits(self, x_xyz: np.ndarray, pad_value, tta: bool = True) -> np.ndarray:
        """x_xyz: [C, X, Y, Z] cropped box. Returns logits [K, X, Y, Z] (float32)."""
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
        cdim, *box = x_xyz.shape
        F = self.frame_xyz
        before = [max(0, (f - s) // 2) for s, f in zip(box, F)]
        padded_shape = [max(s, f) for s, f in zip(box, F)]
        xp = np.empty((cdim, *padded_shape), np.float32)
        for c in range(cdim):
            xp[c] = pad_value[c]
        xp[(slice(None),) + tuple(slice(b, b + s) for b, s in zip(before, box))] = x_xyz

        t = to_network_layout(xp)                                          # -> C, Z, Y, X (nnU-Net order)
        frame_zyx = F[::-1]
        steps = compute_steps_for_sliding_window(tuple(t.shape[1:]), frame_zyx, 0.5)
        windows = list(itertools.product(*steps))
        acc = torch.zeros((self.n_classes, *t.shape[1:]), dtype=torch.float32, device=self.device)
        wsum = torch.zeros(t.shape[1:], dtype=torch.float32, device=self.device)
        g = compute_gaussian(frame_zyx, sigma_scale=1. / 8, value_scaling_factor=10, device=self.device) \
            if len(windows) > 1 else torch.ones(frame_zyx, device=self.device)
        for sz, sy, sx in windows:
            sl = (slice(sz, sz + frame_zyx[0]), slice(sy, sy + frame_zyx[1]), slice(sx, sx + frame_zyx[2]))
            patch = t[(slice(None),) + sl][None].to(self.device, non_blocking=True)
            acc[(slice(None),) + sl] += self._tta(patch, tta)[0] * g
            wsum[sl] += g
        logits = (acc / wsum).permute(0, 3, 2, 1)                          # -> K, X, Y, Z (back to our x,y,z order)
        crop = tuple(slice(b, b + s) for b, s in zip(before, box))
        return logits[(slice(None),) + crop].cpu().numpy()

    def predict_case(self, case: str, tta: bool = True):
        x, rec = load_case_inputs(case)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        logits = self.predict_logits(x, rec["pad_value"], tta)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        e = np.exp(logits - logits.max(0, keepdims=True))
        probs = e / e.sum(0, keepdims=True)
        mem = torch.cuda.max_memory_allocated() / 2**30 if self.device.type == "cuda" else 0.0
        return probs.astype(np.float32), rec, {"case": case, "seconds": dt, "peak_GB": mem, "tta": tta}


def write_outputs(case: str, probs: np.ndarray, rec: dict, out_dir, save_probs: bool = True):
    out_dir = Path(out_dir)
    labels_box = probs.argmax(0).astype(np.uint8)
    full = np.zeros(rec["original_shape"], np.uint8)
    lo, hi = rec["box_lo"], rec["box_hi"]
    full[tuple(slice(a, b + 1) for a, b in zip(lo, hi))] = labels_box
    ref = nib.load(str(io.raw_image_path(case, 0)))                       # original geometry/header
    io.save_like(full, ref, out_dir / "predictions" / f"{case}.nii.gz", np.uint8)
    if save_probs:
        (out_dir / "probabilities").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "probabilities" / f"{case}.npz", probs=probs.astype(np.float16),
                            box_lo=np.array(lo), box_hi=np.array(hi), labels=np.array(list(C.LABELS)))


def predict_cases(network, cases, out_dir, tta=True, save_probs=True, allow_locked=False, purpose="",
                  device="cuda", log=None):
    check_not_locked(cases, allow_locked=allow_locked, purpose=purpose)
    pred = Predictor(network, device=device)
    rows = []
    for case in cases:
        probs, rec, timing = pred.predict_case(case, tta=tta)
        write_outputs(case, probs, rec, out_dir, save_probs=save_probs)
        rows.append(timing)
        if log:
            log(f"predicted {case} in {timing['seconds']:.1f}s, peak {timing['peak_GB']:.1f} GB")
    import pandas as pd
    pd.DataFrame(rows).to_csv(Path(out_dir) / "prediction_timing.csv", index=False)
    return rows
