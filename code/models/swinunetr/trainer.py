"""Swin UNETR (Hatamizadeh et al., 2022), MONAI implementation, trained from scratch.
Recipe verified from Project-MONAI/research-contributions SwinUNETR/BRATS21: AdamW lr 1e-4, wd 1e-5, linear warmup
(50 of 300 epochs) then cosine to 0, batch 1, AMP, no gradient clipping. feature_size 48; gradient checkpointing on
(needed to fit the whole-head frame; it changes memory/speed, not the maths). Single output."""
import math

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from pbts.train.base import PBTSTrainerMixin


class WarmupCosine:
    """Per-epoch schedule with nnU-Net's interface (step(current_epoch)); matches MONAI's
    LinearWarmupCosineAnnealingLR(warmup_start_lr=0, eta_min=0) evaluated at integer epochs."""
    def __init__(self, optimizer, base_lr, warmup_epochs, max_epochs):
        self.opt, self.base, self.warm, self.max = optimizer, base_lr, max(1, warmup_epochs), max_epochs

    def lr_at(self, e):
        if e < self.warm:
            return self.base * e / self.warm
        return 0.5 * self.base * (1 + math.cos(math.pi * (e - self.warm) / max(1, self.max - self.warm)))

    def step(self, current_epoch=None):
        for g in self.opt.param_groups:
            g["lr"] = self.lr_at(current_epoch or 0)

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass


class PBTS_SwinUNETR(PBTSTrainerMixin, nnUNetTrainer):
    recipe_name = "swinunetr"

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device or torch.device("cuda"))
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(img_size=tuple(configuration_manager.patch_size), in_channels=num_input_channels,
                         out_channels=num_output_channels, feature_size=48, use_checkpoint=True, spatial_dims=3)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        warm = max(1, round(self.recipe["warmup_fraction"] * self.num_epochs))
        return opt, WarmupCosine(opt, self.initial_lr, warm, self.num_epochs)

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
