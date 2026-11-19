"""Step 03: register the 234 development cases with nnU-Net using OUR fixed plan.

    python -m pbts.data.nnunet_setup          (after nnUNetv2_extract_fingerprint + plan_experiment -pl nnUNetPlannerResEncL)

Builds pbtsWholeHeadPlans.json from nnU-Net's own ResEnc-L plan, changing only:
  - patch = the whole-head frame (nnU-Net arrays are z,y,x; our FRAME_XYZ reversed)
  - normalisation = NoNormalization (pbts.data.preprocess already z-scored over head voxels)
  - batch size 2 (measured: 18.4 GB at this frame on the A100)
  - an extra '3d_patch128' configuration for the patch-vs-whole-head pilot (same data, same network)
Then writes splits_final.json = the 5 inner folds of the study split.
Run nnUNetv2_preprocess -d 502 -plans_name pbtsWholeHeadPlans -c 3d_wholehead afterwards.
"""
import copy
import json

from pbts import config as C

PLANS_NAME = "pbtsWholeHeadPlans"
CONFIG = "3d_wholehead"
CONFIG_PILOT = "3d_patch128"
FRAME_ZYX = list(reversed(C.FRAME_XYZ))


def main():
    pre = C.NNUNET_PREPROCESSED / C.DATASET_NAME
    src = json.load(open(pre / "nnUNetResEncUNetLPlans.json"))
    base = src["configurations"]["3d_fullres"]

    cfg = copy.deepcopy(base)
    cfg.update({
        "data_identifier": "pbtsWholeHead_3d",
        "patch_size": FRAME_ZYX,
        "batch_size": 2,
        "normalization_schemes": ["NoNormalization"] * len(C.CHANNELS),
        "use_mask_for_norm": [False] * len(C.CHANNELS),
    })
    strides = cfg["architecture"]["arch_kwargs"]["strides"]
    total = [1, 1, 1]
    for s in strides:
        total = [t * k for t, k in zip(total, s)]
    assert all(p % t == 0 for p, t in zip(FRAME_ZYX, total)), (FRAME_ZYX, total)

    pilot = {"inherits_from": CONFIG, "patch_size": [128, 128, 128], "batch_size": 2}

    plans = copy.deepcopy(src)
    plans["plans_name"] = PLANS_NAME
    plans["configurations"] = {CONFIG: cfg, CONFIG_PILOT: pilot}
    plans["pbts_note"] = ("Derived from nnUNetResEncUNetLPlans 3d_fullres: only patch size (whole-head frame), "
                          "batch size and normalisation (NoNormalization, data pre-normalised) were changed.")
    json.dump(plans, open(pre / f"{PLANS_NAME}.json", "w"), indent=1)

    split = json.load(open(C.HEAD_DIR / "study_split.json"))
    json.dump(split["inner_folds"], open(pre / "splits_final.json", "w"), indent=1)
    print(f"wrote {PLANS_NAME}.json ({CONFIG}: patch {FRAME_ZYX}, bs 2; {CONFIG_PILOT}: 128^3) and splits_final.json "
          f"({[len(f['val']) for f in split['inner_folds']]} val cases)")


if __name__ == "__main__":
    main()
