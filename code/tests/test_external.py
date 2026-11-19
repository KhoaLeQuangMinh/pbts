"""The unlabeled-scan pipeline must preprocess exactly like the development data (CPU).
    ../env/bin/python -m pytest -q -p no:cacheprovider tests/test_external.py"""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from pbts import config as C
from pbts.data import io
from pbts.final.external import SUFFIX, crop_normalize, load_external_case

CASES = ["BraTS-PED-00001-000", "BraTS-PED-00024-000", "BraTS-PED-00118-100"]   # normal, stripped, oversized


@pytest.mark.parametrize("case", CASES)
def test_external_preprocessing_is_bitwise_identical(case, tmp_path):
    d = tmp_path / case
    d.mkdir()
    for c, ch in enumerate(C.CHANNELS):                 # challenge naming, pointing at our raw float32 files
        os.symlink(io.raw_image_path(case, c), d / f"{case}-{SUFFIX[ch]}.nii.gz")
    name, imgs, ref = load_external_case(d)
    x, rec = crop_normalize(imgs)
    ours = json.load(open(C.HEAD_DIR / "records" / f"{case}.json"))
    assert name == case and rec["box_lo"] == ours["box_lo"] and rec["box_hi"] == ours["box_hi"]
    assert np.allclose(rec["pad_value"], ours["pad_value"], rtol=0, atol=0)
    base = C.NNUNET_RAW / C.DATASET_NAME / "imagesTr"
    dev = np.stack([nib.load(str(base / f"{case}_{c:04d}.nii.gz")).get_fdata(dtype=np.float32) for c in range(4)])
    assert np.array_equal(x, dev), "external preprocessing differs from the development preprocessing"
