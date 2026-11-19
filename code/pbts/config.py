"""Fixed study constants. Everything in this file is identical for every model."""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths (server)
ROOT = Path(os.environ.get("PBTS_ROOT", "/data/khoalq/pbts"))
KAGGLE_DIR = ROOT / "data/kagglehub_cache/datasets/ckhangnguyn/pbts-preprocessed/versions/1"
MANIFEST = KAGGLE_DIR / "cases_manifest.csv"
METADATA = KAGGLE_DIR / "BraTS-PEDs_metadata.tsv"
NESTED_SPLITS = KAGGLE_DIR / "splits_nested_5x5.json"

RAW_DIR = ROOT / "data/raw"            # float32 .nii.gz in the original 240x240x155 space, all 294 cases
HEAD_DIR = ROOT / "data/head"          # cropped + normalised arrays and per-case records, all 294 cases
LOCKED_DIR = ROOT / "data/locked_test" # the 60 locked cases. Never registered with nnU-Net.

NNUNET_RAW = ROOT / "nnunet/raw"
NNUNET_PREPROCESSED = ROOT / "nnunet/preprocessed"
NNUNET_RESULTS = ROOT / "nnunet/results"
DATASET_ID = 502
DATASET_NAME = "Dataset502_PBTSHead"   # only the 234 development cases live here

RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"

# W&B mirror (local files are the master copy). Project visibility is TEAM; every team member signed the
# BraTS-PEDs data agreement, so monitor pictures may be mirrored. Never set the project to Public.
# The entity (account) name is personal, so it lives in secrets/, not in git.
WANDB_PROJECT = "Ped-BraTS"
WANDB_ENTITY_FILE = ROOT / "secrets/wandb_entity"
WANDB_KEY_FILE = ROOT / "secrets/wandb_key"
WANDB_LOG_IMAGES = True   # only ever True for a private project (BraTS data-use terms)

# ---------------------------------------------------------------- data definition
CHANNELS = ("FLAIR", "T1", "T1CE", "T2")          # file suffixes _0000 .. _0003
LABELS = {"background": 0, "ET": 1, "NET": 2, "CC": 3, "ED": 4}
REGIONS = {"ET": (1,), "NET": (2,), "CC": (3,), "ED": (4,), "TC": (1, 2, 3), "WT": (1, 2, 3, 4)}
IMAGE_SHAPE = (240, 240, 155)                      # x, y, z (nibabel order), 1 mm isotropic, LPS
SPACING = (1.0, 1.0, 1.0)

# Whole-head training/prediction frame, x, y, z. Covers 292/294 head boxes; the 2 larger ones
# (BraTS-PED-00118-100/-101, 232-238 mm in y) lose a few mm of scalp per sample ("option A").
FRAME_XYZ = (192, 224, 160)

# ---------------------------------------------------------------- split
OUTER_FOLD = 0      # outer fold 0 of the nested file = the 60 locked test scans
N_INNER = 5         # its 5 inner folds = cross-validation on the 234 development scans
TUNING_FOLD = 0     # inner fold used for (pre-declared, <=3 per model) tuning runs

# ---------------------------------------------------------------- head mask
HEADMASK_CLOSING_ITERS = 3
STRIPPED_RATIO = 1.5   # non-zero voxels < 1.5 x Otsu head => scan was already skull-stripped
