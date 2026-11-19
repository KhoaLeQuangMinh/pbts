# pbts

Comparison of adult brain-tumor segmentation models on pediatric data (BraTS-PEDs 2026, 294 scans),
to learn where each model succeeds and fails before designing a child-specific method.

Models (all trained from scratch, same data/augmentation/prediction/evaluation, each with its own
optimiser recipe): nnU-Net ResEnc L, Primus, SegMamba, Swin UNETR.

## Layout
- `code/pbts/` — shared package every model imports
  - `data/` — conversion (`convert.py`), study split + locked-test guard (`splits.py`, `guard.py`),
    head mask and whole-head preprocessing (`headmask.py`, `preprocess.py`), nnU-Net plan (`nnunet_setup.py`)
- `code/tests/` — checks that must pass before any training run

## Data pipeline (run on the GPU server, from `code/`)
```
python -m pbts.data.convert        # Kaggle download -> verified float32 NIfTI
python -m pbts.data.splits         # 60 locked test / 234 development, 5 inner folds
python -m pbts.data.preprocess     # head crop + per-channel z-score over head voxels
nnUNetv2_extract_fingerprint -d 502 && nnUNetv2_plan_experiment -d 502 -pl nnUNetPlannerResEncL
python -m pbts.data.nnunet_setup   # whole-head plan (192x224x160 frame), no renormalisation
nnUNetv2_preprocess -d 502 -plans_name pbtsWholeHeadPlans -c 3d_wholehead
python -m pytest tests/
```

No patient data, images or metadata are stored in this repository.
