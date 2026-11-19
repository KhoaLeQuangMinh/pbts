"""Rebuild a trained network from its run directory (same trainer class, same plan, same weights)."""
import json
import os
from pathlib import Path

import torch

from pbts import config as C


def load_network(run_dir, weights="weights_100.pth", device="cuda"):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "run_config.json"))
    os.environ.update({"nnUNet_raw": str(C.NNUNET_RAW), "nnUNet_preprocessed": str(C.NNUNET_PREPROCESSED),
                       "nnUNet_results": str(run_dir / "nnunet"), "PBTS_RUN_CONFIG": str(run_dir / "run_config.json"),
                       "nnUNet_compile": "false"})
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from pbts.train.run import _load_trainer_class
    pre = C.NNUNET_PREPROCESSED / C.DATASET_NAME
    plans = load_json(str(pre / "pbtsWholeHeadPlans.json"))
    plans["continue_training"] = True
    trainer = _load_trainer_class(cfg["model"])(plans=plans, configuration=cfg["configuration"], fold=cfg["fold"],
                                               dataset_json=load_json(str(pre / "dataset.json")), device=torch.device(device))
    trainer.initialize()
    state = torch.load(run_dir / "checkpoints" / weights, map_location=device, weights_only=False)["network_weights"]
    net = trainer.network._orig_mod if hasattr(trainer.network, "_orig_mod") else trainer.network
    net.load_state_dict(state)
    trainer.set_deep_supervision_enabled(False)
    net.eval()
    return net, tuple(reversed(trainer.configuration_manager.patch_size)), cfg
