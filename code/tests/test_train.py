"""Training-infrastructure tests (CPU only). Run on the server with nnUNet_* env vars set:
    ../env/bin/python -m pytest -q -p no:cacheprovider tests/test_train.py"""
import json

import numpy as np
import pytest
import torch

from pbts import config as C
from pbts.infer.predict import load_case_inputs, to_network_layout

PRE = C.NNUNET_PREPROCESSED / C.DATASET_NAME
CASES = ["BraTS-PED-00001-000", "BraTS-PED-00118-100", "BraTS-PED-00024-000"]   # normal, oversized, skull-stripped


def _dataset(cases):
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    return nnUNetDatasetBlosc2(str(PRE / "pbtsWholeHead_3d"), cases)


@pytest.mark.parametrize("case", CASES)
def test_predictor_layout_matches_nnunet_training_arrays(case):
    data, _, _, _ = _dataset([case]).load_case(case)
    x, _ = load_case_inputs(case)
    assert torch.equal(to_network_layout(x), torch.from_numpy(np.asarray(data, dtype=np.float32)))


@pytest.mark.parametrize("case", CASES)
def test_loader_pads_with_case_background(case):
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from pbts.train.loader import PBTSDataLoader, pad_values
    plans = load_json(str(PRE / "pbtsWholeHeadPlans.json"))
    lm = PlansManager(plans).get_label_manager(load_json(str(PRE / "dataset.json")))
    frame = list(reversed(C.FRAME_XYZ))
    dl = PBTSDataLoader(_dataset([case]), 1, [f + 40 for f in frame], frame, lm, oversample_foreground_percent=0.0,
                        sampling_probabilities=None, pad_sides=None, transforms=None)
    for _ in range(3):
        b = dl.generate_train_batch()
        data, seg = b["data"][0].numpy(), b["target"][0, 0].numpy()
        outside = seg == -1
        assert outside.any(), "frame+40 must reach outside every case"
        for c, v in enumerate(pad_values(case)):
            assert np.allclose(data[c][outside], v), f"channel {c} padded with {np.unique(data[c][outside])[:3]}, expected {v}"
        assert not np.isnan(data).any()
