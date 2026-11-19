"""SegMamba (Xing et al., MICCAI 2024). Network from ge-xing/SegMamba @ cff3597 (repos/SegMamba), with its own
bundled mamba_ssm 1.0.1 / causal_conv1d 1.0.0 CUDA extensions built for this environment.
Recipe (pbts.protocol): SGD lr 1e-2, momentum 0.99 Nesterov, wd 3e-5, poly 0.9, grad clip 12, AMP; single output.
Only change to the network: each Mamba layer's `nslices` = the real depth of its feature map (the original
hard-codes 64/32/16/8, which is only correct for 128-slice inputs)."""
import sys

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from pbts import config as C
from pbts.train.base import PBTSTrainerMixin

SEGMAMBA_REPO = C.ROOT / "repos/SegMamba"


def build_segmamba(in_ch, out_ch, patch_size_zyx):
    if str(SEGMAMBA_REPO) not in sys.path:
        sys.path.insert(0, str(SEGMAMBA_REPO))
    from model_segmamba.segmamba import SegMamba
    net = SegMamba(in_chans=in_ch, out_chans=out_ch, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384])
    depth = patch_size_zyx[0]        # MambaLayer flattens (D, H, W) with D slowest: chunks along D are slices
    for i, stage in enumerate(net.vit.stages):
        n = depth // (2 ** (i + 1))  # stem stride 2, then one stride-2 downsampling per stage
        assert depth % (2 ** (i + 1)) == 0, (depth, i)
        for layer in stage:
            layer.mamba.nslices = n
    return net


class PBTS_SegMamba(PBTSTrainerMixin, nnUNetTrainer):
    recipe_name = "segmamba"

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        import torch
        super().__init__(plans, configuration, fold, dataset_json, device or torch.device("cuda"))
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return build_segmamba(num_input_channels, num_output_channels, configuration_manager.patch_size)

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
