"""Reading/writing NIfTI files, including the odd layout of the Kaggle download."""
import glob
import gzip
import hashlib
import os

import nibabel as nib
import numpy as np

from pbts import config as C


def load_nifti(path) -> nib.Nifti1Image:
    """Load a NIfTI image.

    Handles the Kaggle layout, where each image became a folder '<name>.nii/' holding a single
    '<name>.nii.gz.part' file that may or may not still be gzipped.
    """
    path = str(path)
    if os.path.isdir(path):
        inner = sorted(glob.glob(os.path.join(path, "*")))
        if len(inner) != 1:
            raise ValueError(f"{path}: expected exactly 1 file inside, found {len(inner)}")
        path = inner[0]
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return nib.Nifti1Image.from_bytes(raw)


def kaggle_image_path(case: str, ch: int):
    return C.KAGGLE_DIR / "imagesTr" / f"{case}_{ch:04d}.nii"


def kaggle_label_path(case: str):
    return C.KAGGLE_DIR / "labelsTr" / f"{case}.nii"


def raw_image_path(case: str, ch: int):
    return C.RAW_DIR / "imagesTr" / f"{case}_{ch:04d}.nii.gz"


def raw_label_path(case: str):
    return C.RAW_DIR / "labelsTr" / f"{case}.nii.gz"


def save_like(array: np.ndarray, ref: nib.Nifti1Image, path, dtype) -> None:
    """Save an array with the reference image's affine and header (qform/sform codes kept)."""
    hdr = ref.header.copy()
    hdr.set_data_dtype(dtype)
    hdr["scl_slope"], hdr["scl_inter"] = 1.0, 0.0
    img = nib.Nifti1Image(np.asarray(array, dtype=dtype), ref.affine, header=hdr)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    nib.save(img, str(path))


def load_raw_case(case: str, with_label: bool = True):
    """Return (images[4,X,Y,Z] float32, label[X,Y,Z] uint8 or None, reference image)."""
    refs = [nib.load(str(raw_image_path(case, c))) for c in range(len(C.CHANNELS))]
    imgs = np.stack([r.get_fdata(dtype=np.float32) for r in refs])
    seg = None
    if with_label:
        seg = np.asanyarray(nib.load(str(raw_label_path(case))).dataobj).astype(np.uint8)
    return imgs, seg, refs[0]


def fingerprint(array: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def list_kaggle_cases():
    files = glob.glob(str(C.KAGGLE_DIR / "imagesTr" / "*_0000.nii*"))
    return sorted({os.path.basename(p).split("_")[0] for p in files})
