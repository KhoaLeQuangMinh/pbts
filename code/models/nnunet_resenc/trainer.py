"""nnU-Net ResEnc L: nnU-Net's own trainer and the ResEnc-L architecture from our plan (unchanged).
Recipe (pbts.protocol): SGD lr 1e-2, momentum 0.99 Nesterov, wd 3e-5, poly, grad clip 12, deep supervision."""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from pbts.train.base import PBTSTrainerMixin


class PBTS_ResEncL(PBTSTrainerMixin, nnUNetTrainer):
    recipe_name = "nnunet_resenc_l"
