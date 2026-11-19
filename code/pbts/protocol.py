"""Pre-declared study protocol. Committed BEFORE any training run; changing it after results exist
invalidates the comparison, so any change must be a new, dated, explained commit.

Decision (2026-11-19): each model is trained with its authors' ORIGINAL recipe (no tuning), plus one
pre-declared learning-rate rescue. Shared parts (data, frame, augmentation, prediction, scoring)
live elsewhere in pbts and are identical for every model.
"""

# Training length is set once, for every model, by the nnU-Net pilot (Phase B), in training samples
# (1 epoch = 1 pass over the ~188-scan training fold). Recipes below are rescaled to that length:
# schedules and warmups keep the same FRACTION of training as in the original code.
TRAINING_SAMPLES = None   # set after the pilot, in a separate commit

RECIPES = {
    "nnunet_resenc_l": {
        "source": "nnunetv2 nnUNetTrainer + nnUNetResEncUNetLPlans (3d_fullres architecture)",
        "optimizer": "SGD", "lr": 1e-2, "momentum": 0.99, "nesterov": True, "weight_decay": 3e-5,
        "schedule": "poly(exponent=0.9)", "warmup_fraction": 0.0,
        "loss": "Dice + CE", "deep_supervision": True, "batch_size": 2,
    },
    "primus_v2b": {
        "source": "nnunetv2 nnUNet_PrimusV2B_Trainer (AbstractPrimus)",
        "optimizer": "AdamW", "lr": 3e-4, "betas": (0.9, 0.98), "weight_decay": 5e-2,
        "schedule": "linear warmup then poly", "warmup_fraction": 50 / 1000,
        "loss": "Dice + CE", "deep_supervision": False, "drop_path": 0.2, "batch_size": 1,
    },
    "segmamba": {
        "source": "ge-xing/SegMamba 3_train.py @ cff3597",
        "optimizer": "SGD", "lr": 1e-2, "momentum": 0.99, "nesterov": True, "weight_decay": 3e-5,
        "schedule": "poly", "warmup_fraction": 0.0,
        "loss": "CE only (as in the original code)", "deep_supervision": False, "batch_size": None,  # set by memory test
        "deviation": "out_chans 4 -> 5 (our label set); whole-head frame instead of 128^3",
    },
    "swinunetr": {
        "source": "MONAI SwinUNETR, BraTS21 research-contributions recipe (TO VERIFY against that code before use)",
        "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-5,
        "schedule": "linear warmup then cosine", "warmup_fraction": 50 / 300,
        "loss": "Dice (TO VERIFY)", "deep_supervision": False, "feature_size": 48,
        "gradient_checkpointing": True, "batch_size": 1,
        "deviation": "5-label softmax output instead of 3 sigmoid regions; trained from scratch",
    },
}

# One rescue, checked on each model's FIRST run (inner fold 0). Same rule for every model.
RESCUE = {
    "triggers": {
        "non_finite_loss": "training stopped by the NaN/Inf guard",
        "not_learning": "mean training loss over the 5 epochs ending at 10% of training is not below "
                        "0.9 x the mean over the first 5 epochs (covers divergence and stagnation)",
        "floor": "mean whole-volume WT Dice (official scorer) on the fold-0 validation scans < 0.5",
    },
    "action": "rerun inner fold 0 ONCE with lr / 3; nothing else changes",
    "if_rescue_also_fails": "report the model as not trainable under this protocol; no further changes",
    "later_folds": "folds 1-4 use exactly the recipe that passed fold 0. A crash in a later fold resumes "
                   "once from its last checkpoint; if it recurs it is reported as a failure, never re-tuned.",
}

# The 3 pilot runs that fix shared choices (not per-model tuning): all nnU-Net ResEnc L, inner fold 0.
PILOT = {
    "wholehead": "3d_wholehead, TRAINING_SAMPLES candidate = 100k; doubles as ResEnc's fold-0 run",
    "patch128": "3d_patch128, same number of samples; whole-head vs patch evidence",
    "length_decision": "score the 25/50/75/100% checkpoints of the whole-head run; the shortest length "
                       "within noise of 100% becomes TRAINING_SAMPLES for every model",
}
