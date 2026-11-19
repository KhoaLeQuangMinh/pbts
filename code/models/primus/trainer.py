"""PrimusV2-B: nnU-Net's own Primus trainer (AdamW lr 3e-4, wd 0.05, linear warmup then poly, grad clip 1,
no deep supervision, drop-path 0.2), unchanged except what PBTSTrainerMixin documents."""
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import nnUNet_PrimusV2B_Trainer

from pbts.train.base import PBTSTrainerMixin


class PBTS_PrimusV2B(PBTSTrainerMixin, nnUNet_PrimusV2B_Trainer):
    recipe_name = "primus_v2b"
