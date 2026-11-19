"""Data-pipeline tests. Run on the server:  ../env/bin/python -m pytest -q tests/test_data.py
Set PBTS_TEST_ALL=1 to run the per-case checks on all 294 cases instead of a sample."""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from pbts import config as C
from pbts.data import io
from pbts.data.guard import check_not_locked, locked_cases
from pbts.data.splits import patient

SPECIAL = ["BraTS-PED-00024-000", "BraTS-PED-00025-103",          # skull-stripped
           "BraTS-PED-00118-100", "BraTS-PED-00118-101",          # oversized heads
           "BraTS-PED-00010-000", "BraTS-PED-00156-000"]          # stray labels / labels outside mask
ALL = io.list_kaggle_cases()
SAMPLE = ALL if os.environ.get("PBTS_TEST_ALL") else sorted(set(SPECIAL) | set(ALL[::30]))


def record(case):
    return json.load(open(C.HEAD_DIR / "records" / f"{case}.json"))


def cropped_paths(case, rec):
    if rec["locked"]:
        return [C.LOCKED_DIR / "images" / f"{case}_{c:04d}.nii.gz" for c in range(4)], None
    base = C.NNUNET_RAW / C.DATASET_NAME
    return [base / "imagesTr" / f"{case}_{c:04d}.nii.gz" for c in range(4)], base / "labelsTr" / f"{case}.nii.gz"


@pytest.mark.parametrize("case", SAMPLE)
def test_crop_roundtrip_and_normalisation(case):
    rec = record(case)
    imgs, seg, ref = io.load_raw_case(case)
    img_paths, lab_path = cropped_paths(case, rec)
    lo, hi = np.array(rec["box_lo"]), np.array(rec["box_hi"])
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
    mask = np.asanyarray(nib.load(str(C.HEAD_DIR / "masks" / f"{case}.nii.gz")).dataobj).astype(bool)

    for c, p in enumerate(img_paths):
        im = nib.load(str(p))
        # geometry: cropped voxel (0,0,0) is original voxel lo
        assert np.allclose(im.affine[:3, 3], (ref.affine @ np.r_[lo, 1])[:3], atol=1e-4)
        x = im.get_fdata(dtype=np.float32)
        assert x.shape == tuple(rec["box_shape"])
        # de-normalising reproduces the original intensities inside the box
        back = x.astype(np.float64) * rec["norm_std"][c] + rec["norm_mean"][c]
        assert np.allclose(back, imgs[c][sl], rtol=1e-4, atol=1e-3 * rec["norm_std"][c])
        # head voxels are mean 0 / std 1; raw 0 maps to the recorded pad value
        assert abs(x[mask].mean()) < 1e-3 and abs(x[mask].std() - 1) < 1e-3
        assert np.isclose((0 - rec["norm_mean"][c]) / rec["norm_std"][c], rec["pad_value"][c])

    if lab_path is not None:
        lab = np.asanyarray(nib.load(str(lab_path)).dataobj).astype(np.uint8)
        full = np.zeros(C.IMAGE_SHAPE, np.uint8)
        full[sl] = lab                                     # map back into 240x240x155
        diff = full != seg
        assert diff.sum() == rec["label_vox_outside_box"]  # only the recorded stray voxels differ
        assert (seg[diff] > 0).all() and (full[diff] == 0).all()


def test_split_integrity():
    split = json.load(open(C.HEAD_DIR / "study_split.json"))
    locked, dev = locked_cases(), set(split["development"])
    assert len(locked) == 60 and len(dev) == 234 and not (locked & dev)
    assert not ({patient(c) for c in locked} & {patient(c) for c in dev})
    vals = [c for f in split["inner_folds"] for c in f["val"]]
    assert sorted(vals) == sorted(dev)
    for f in split["inner_folds"]:
        assert not ({patient(c) for c in f["train"]} & {patient(c) for c in f["val"]})


def test_locked_cases_not_in_nnunet_dataset():
    base = C.NNUNET_RAW / C.DATASET_NAME
    present = {f.split("_")[0] for f in os.listdir(base / "imagesTr")} | \
              {f.replace(".nii.gz", "") for f in os.listdir(base / "labelsTr")}
    assert len(present) == 234
    assert not (present & locked_cases())


def test_guard_blocks_locked():
    some_locked = sorted(locked_cases())[:1]
    with pytest.raises(PermissionError):
        check_not_locked(some_locked)
    check_not_locked(["BraTS-PED-00001-000"] if "BraTS-PED-00001-000" not in locked_cases() else [])


def test_records_complete():
    import pandas as pd
    df = pd.read_csv(C.HEAD_DIR / "records.csv")
    assert len(df) == 294 and df.locked.sum() == 60
    assert df.skull_stripped.sum() == 2 and df.oversized_vs_frame.sum() == 2
    assert df.label_vox_outside_box.sum() == 116
