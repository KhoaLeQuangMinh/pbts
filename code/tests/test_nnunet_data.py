"""nnU-Net must store our preprocessed data untouched (only transposed x,y,z -> z,y,x).
Run on the server with nnUNet_* env vars set:  ../env/bin/python -m pytest -q tests/test_nnunet_data.py"""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from pbts import config as C

PRE = C.NNUNET_PREPROCESSED / C.DATASET_NAME
DATA = PRE / "pbtsWholeHead_3d"
SPLIT = json.load(open(C.HEAD_DIR / "study_split.json"))
CASES = SPLIT["development"] if os.environ.get("PBTS_TEST_ALL") else SPLIT["development"][::20] + \
    ["BraTS-PED-00118-100", "BraTS-PED-00024-000"]


def test_plans_and_splits():
    plans = json.load(open(PRE / "pbtsWholeHeadPlans.json"))
    cfg = plans["configurations"]["3d_wholehead"]
    assert cfg["patch_size"] == list(reversed(C.FRAME_XYZ))
    assert cfg["normalization_schemes"] == ["NoNormalization"] * 4
    assert json.load(open(PRE / "splits_final.json")) == SPLIT["inner_folds"]
    assert len([f for f in os.listdir(DATA) if f.endswith("_seg.b2nd")]) == 234


@pytest.mark.parametrize("case", CASES)
def test_preprocessed_equals_ours(case):
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    ds = nnUNetDatasetBlosc2(str(DATA), [case])
    data, seg, _, props = ds.load_case(case)
    data, seg = np.asarray(data), np.asarray(seg)
    base = C.NNUNET_RAW / C.DATASET_NAME
    ours = np.stack([nib.load(str(base / "imagesTr" / f"{case}_{c:04d}.nii.gz")).get_fdata(dtype=np.float32)
                     for c in range(4)])
    lab = np.asanyarray(nib.load(str(base / "labelsTr" / f"{case}.nii.gz")).dataobj)
    assert data.shape == (4,) + ours.shape[1:][::-1], "nnU-Net cropped or resampled the data"
    assert np.allclose(data, ours.transpose(0, 3, 2, 1), atol=1e-5), "nnU-Net changed intensities"
    assert np.array_equal(seg[0], lab.transpose(2, 1, 0).astype(seg.dtype)), "nnU-Net changed labels"
    assert [list(b) for b in props["bbox_used_for_cropping"]] == [[0, s] for s in data.shape[1:]]
