# Study plan

What this study does, every decision it has made, why, and where in the code each decision lives. Use it to check
that the code does what the plan says. When the two disagree, that is a bug in one of them; please raise it.

Last updated 2026-11-21. The binding version of the training settings is [`code/pbts/protocol.py`](../code/pbts/protocol.py):
it was committed before any training run, and every later change is a dated amendment inside it.

---

## 1. Goal

Take strong adult brain-tumour segmentation models and train them, under identical conditions, on paediatric scans
(BraTS-PEDs 2026, 294 scans). Learn where each one succeeds and where it fails in children. Use that to design a
child-specific method afterwards.

Two requirements follow from that, and most decisions below serve one of them:

- **Comparable.** Everything except the model is shared: data, split, input, augmentation, loss, training length,
  prediction and scoring. A difference in results should come from the architecture and its own recipe, not from
  anything we did differently.
- **Each model at its best, without tuning.** Each model keeps its authors' original optimiser recipe. There is no
  per-model tuning, only one pre-declared rescue for a model that fails to train.

Keep the raw predictions, not just the scores, so any analysis can be redone later without retraining.

## 2. Data

| | |
|---|---|
| Source | BraTS-PEDs 2026 training data, from the Kaggle dataset named in [`config.py`](../code/pbts/config.py) |
| Scans | 294, each 240×240×155 at 1 mm, 4 channels: FLAIR, T1, T1CE, T2 |
| Labels | 0 background, 1 ET (enhancing tumour), 2 NET (non-enhancing tumour), 3 CC (cystic component), 4 ED (oedema) |
| Scored regions | ET, NET, CC, ED, TC = ET+NET+CC, WT = all four |

Labels are often absent: ED is absent in 78% of scans, CC in 65% and ET in 33%. A scan with no ED where the
model predicts some ED counts as a false alarm. This matters for the loss (section 5) and for scoring (section 8).

**Conversion** ([`data/convert.py`](../code/pbts/data/convert.py)). Each case is copied to float32 NIfTI and checked
against the dataset manifest: shape, identical affine across channels and label, spacing, orientation, finite
values, label values and per-label voxel counts. Any failure stops the step.

## 3. Split: a locked test set plus 5-fold cross-validation

| | Scans | Use |
|---|---|---|
| Locked test | 60 | Outer fold 0 of `splits_nested_5x5.json`. Scored **once**, at the very end (section 10). |
| Development | 234 | Cross-validated over that outer fold's 5 inner folds (about 187 training / 47 validation scans each). |

- [`data/splits.py`](../code/pbts/data/splits.py) checks that no child has scans on both sides: none in both the
  locked test and development, and none in both training and validation of any inner fold. Follow-up scans of the
  same child always move together.
- [`data/guard.py`](../code/pbts/data/guard.py): any code that reads a case refuses a locked case unless it is the
  final evaluation, which writes an audit line to `data/locked_test/ACCESS_LOG.txt`.
- The locked cases are never registered with nnU-Net.

Every model's 5 inner-fold runs together predict each of the 234 development scans exactly once ("out-of-fold"),
and model comparisons are made on those predictions.

## 4. Input: the whole head, no skull-stripping

Code: [`data/headmask.py`](../code/pbts/data/headmask.py), [`data/preprocess.py`](../code/pbts/data/preprocess.py),
[`data/nnunet_setup.py`](../code/pbts/data/nnunet_setup.py), [`train/loader.py`](../code/pbts/train/loader.py).

1. **Head mask from the images only, never the labels**, so the identical step runs on test scans. Each channel is
   scaled by its 99th percentile and the four are summed. Then: an Otsu threshold, the largest connected component,
   closing and hole filling. The background is not exactly zero, so plain "non-zero" cropping would keep about 75%
   of the volume. The roughly 2 already skull-stripped scans are detected and handled.
2. **Crop** all channels and the label to the mask's bounding box.
3. **Normalise:** z-score each channel with the mean and standard deviation of the voxels inside the head. Nothing
   else: no skull-stripping, no clipping, no blanking. nnU-Net's own normalisation is switched off
   (`NoNormalization`), so the data is not normalised twice.
4. **Fixed frame of 192×224×160 voxels** (x, y, z). Every model sees the whole head in one piece; there are no
   patches. The frame fits 292 of 294 heads. The 2 larger ones (BraTS-PED-00118-100/-101) lose a few mm of scalp
   per training sample ("option A"). At prediction time, sliding windows cover them completely.
5. **Padding.** Where the frame is larger than the head, it is filled with that case's own background value (raw
   intensity 0 after normalisation), not with 0. After z-scoring, 0 means "average head brightness", so padding
   with 0 would draw a bright frame around every sample.

Each case keeps a record (`data/head/records/<case>.json`): crop box, normalisation statistics and padding value.
Predictions are mapped back with it and scored in the original 240×240×155 space.

**Why the whole head, not 128³ patches:** decided by the pilot (section 7).

## 5. Models and training

All four models are trained from scratch: no pretrained weights. Each runs as an nnU-Net trainer with our mixin in
front ([`train/base.py`](../code/pbts/train/base.py)), so data loading, augmentation, loss and logging are literally
the same code for every model.

### Shared by every model

| What | Setting | Why |
|---|---|---|
| Loss | nnU-Net Dice + cross-entropy, 1:1 | Regions are often absent. SegMamba's original cross-entropy alone under-segments tiny regions; Swin UNETR's original Dice alone gives an all-or-nothing signal on absent regions. Either would mix up the effect of the loss with the effect of the architecture. |
| Augmentation | nnU-Net default training transforms | Same for all |
| Training length | 100,000 training samples (about 540 epochs; 1 epoch = 1 pass over the training fold) | Pilot, section 7. Schedules and warm-ups keep the same *fraction* of training as in the original code. |
| Saved weights | At 25 / 50 / 75 / 100% of training, plus a "latest" checkpoint every 30 min for resuming | No "best" checkpoint and no early stopping: choosing a checkpoint on validation scores would leak validation into training. |
| Monitoring | Fast scores on 10 fixed validation scans every 5 epochs | For logging only; never used to choose anything |
| Precision | Mixed precision (AMP) | |

Deep supervision stays per model, because it depends on whether the architecture has decoder levels.

### Each model's own recipe

| Model | Implementation | Optimiser | Schedule | Grad clip | Deep sup. | Batch |
|---|---|---|---|---|---|---|
| nnU-Net ResEnc L | nnU-Net trainer, ResEnc-L plan | SGD lr 1e-2, momentum 0.99 Nesterov, wd 3e-5 | poly 0.9 | 12 | yes | 2 |
| PrimusV2-B | nnU-Net `nnUNet_PrimusV2B_Trainer` | AdamW lr 3e-4, betas (0.9, 0.98), wd 0.05; drop-path 0.2 | linear warm-up (5%) then poly | 1 | no | 1 |
| SegMamba | ge-xing/SegMamba @ cff3597 | SGD lr 1e-2, momentum 0.99 Nesterov, wd 3e-5 | poly 0.9 | 12 | no | 1 |
| Swin UNETR | MONAI 1.4.0, feature size 48 | AdamW lr 1e-4, wd 1e-5 | linear warm-up (50/300) then cosine | none | no | 1 |

Recipes were taken from each model's own code; the source of each is in `protocol.RECIPES`. Necessary deviations,
each recorded in the recipe's `deviation` field:

- **SegMamba**
  - Output channels 4 → 5 (our label set).
  - Whole-head frame instead of 128³.
  - Batch 1 instead of 2: batch 2 does not fit a 40 GB A100 at this frame. Same number of samples, so twice the steps.
  - Shared loss instead of cross-entropy only.
  - Each Mamba layer's `nslices` is set to the real depth of its feature map (80/40/20/10 for our 160-slice frame).
    The original hard-codes 64/32/16/8, which is correct only for 128-slice inputs.
  - Poly lr updated per epoch instead of per step.
- **Swin UNETR**
  - A 5-label softmax output instead of 3 sigmoid regions.
  - Shared loss instead of Dice only.
  - Gradient checkpointing on, needed to fit the frame in memory. It changes memory use and speed, not the maths.

HFF-Net was considered and dropped.

## 6. Rescue rule: the only allowed change to a recipe

Defined in `protocol.RESCUE` and applied automatically by [`final/queue.py`](../code/pbts/final/queue.py). It is
checked on each model's **first** run (inner fold 0) only.

- **Triggers:**
  - the loss becomes NaN or infinite;
  - the model is not learning: at 10% of training, the mean training loss of the last 5 epochs is not below 0.9 ×
    the mean of the first 5;
  - after training, the mean WT Dice on the fold-0 validation scans is below 0.5.
- **Action:** rerun fold 0 once with lr / 3. Nothing else changes.
- **If that also fails,** the model is reported as not trainable under this protocol. No further changes.
- **Folds 1–4** use exactly the recipe that passed fold 0 and start only after fold 0 has passed. A crash in a later
  fold is resumed once from its last checkpoint; if it crashes again, it is reported as a failure, never re-tuned.

## 7. The pilot (Decision 1, 2026-11-20)

Three nnU-Net ResEnc L runs on inner fold 0 fixed two shared choices for every model. The evidence is from
[`tools/pilot_decision.py`](../code/tools/pilot_decision.py); the full numbers are in the 2026-11-20 amendment in
`protocol.py`.

- **Training length = 100,000 samples.** The 25/50/75% checkpoints of the 100k run were each compared with the 100%
  checkpoint, on the same 48 validation scans, with patient-bootstrap 95% CIs. "Within noise" was read strictly:
  every region and every metric must have a CI containing 0. The 75% checkpoint passes for every Dice score but not
  for boundary accuracy (NSD) on NET and ED, so 100% it is. That run also serves as ResEnc's fold-0 run.
- **Whole head, not 128³ patches.** The 128³ run has slightly sharper boundaries (NSD WT +0.027), but it predicts
  absent regions much more often: false alarms on ET/CC/ED of 0.56/0.44/0.39 against 0.19/0.17/0.08.

## 8. Prediction and scoring

**Prediction** ([`infer/predict.py`](../code/pbts/infer/predict.py)), identical for every model:

1. Place the cropped, normalised case in the frame. The 2 oversized heads use sliding windows (step 0.5, Gaussian
   weights).
2. Run 8-way flip test-time augmentation and average the logits over flips (and windows).
3. Softmax. The probabilities are kept as float16 in crop coordinates.
4. Argmax, then paste back into the original 240×240×155 space with the original header.
5. **No post-processing of any kind**: no removal of small components and no label replacement. The study compares
   what each model actually predicts; a clean-up step would be one more tuned choice.

After training, **every saved checkpoint (25/50/75/100%)** predicts all validation scans of its fold, and each
prediction is scored.

**Scoring** ([`eval/evaluate.py`](../code/pbts/eval/evaluate.py)):

- **The official BraTS 2026 scorer, run as-is** (BraTS_evaluation 0.0.10 + panoptica 2.1.7, in its own environment,
  [`eval/official.py`](../code/pbts/eval/official.py)). It is called, never re-implemented, so our numbers cannot
  drift from the challenge's. Its edge cases are documented in that file. For example, a region absent in both truth
  and prediction gives NaN, and that case drops out of that region's mean.
- **The BraTS 2023–25 lesion-wise Dice/HD95** ([`eval/lesionwise2023.py`](../code/pbts/eval/lesionwise2023.py)). It
  is re-implemented so it covers all 6 paediatric regions, and tested to match the original code (@ 43c9052) exactly.
- **Diagnostics** ([`eval/diagnostics.py`](../code/pbts/eval/diagnostics.py)):
  - presence (region present or absent, in truth and in prediction) and volumes;
  - one row per true lesion and per false-alarm piece, with size, position and distance to the tumour;
  - a 5×5 voxel label confusion matrix.

## 9. Analysis

- **Per run:** [`report/run_report.py`](../code/pbts/report/run_report.py) builds `report.html`. It contains MRI
  images, so it is never published.
- **Across models** ([`analysis/compare.py`](../code/pbts/analysis/compare.py)): out-of-fold results on all 234
  development scans. Only models with all 5 folds complete enter the statistics, so every comparison is on
  identical scans.
- **Statistics** ([`analysis/stats.py`](../code/pbts/analysis/stats.py)):
  - 95% CIs from a bootstrap over **children**, not scans, so a child's repeat scans move together;
  - paired per-scan differences with a Wilcoxon signed-rank test and Holm correction;
  - NaNs are dropped pairwise, never filled in.
- **Subgroups** ([`analysis/strata.py`](../code/pbts/analysis/strata.py)): collection (a rough proxy for tumour
  type), follow-up scan, tumour size and position, which labels are present, image quality, skull-stripped,
  oversized head.
  - **Metadata is limited.** Collection is known for 275 of 294 scans. Age and sex exist only for the CBTN
    collection (134 scans), so those two subgroups cannot be separated from the collection effect.
- **Ensembles** ([`final/ensemble.py`](../code/pbts/final/ensemble.py)) average probabilities. They are for analysis
  only and never change a model's own results.

## 10. The locked test, once at the end

[`final/locked_test.py`](../code/pbts/final/locked_test.py):

- Each model's 5 fold models predict the 60 locked scans; their probabilities are averaged, and the result is scored
  once.
- It requires a typed confirmation phrase and logs every access.
- A second run is refused unless a reason is given, which is then logged and must be reported.

Nothing is tuned on the locked test, and it is not looked at before the development comparison is finished.

For scans without labels (for example the 91 official validation cases), [`final/external.py`](../code/pbts/final/external.py)
writes challenge-ready predictions with exactly the same preprocessing.

## 11. What every run keeps

`runs/<model>/<tag>/fold_<k>/`. Tags: `default` is the original recipe and `rescue_lr3` the rescue.

| Path | Contents |
|---|---|
| `run_config.json`, `provenance/` | Exact settings, code version (commit + uncommitted diff) and environment, written at start and on every resume |
| `epochs.jsonl`, `steps.jsonl`, `case_losses.jsonl`, `monitor.jsonl`, `system.jsonl`, `events.jsonl` | Full training logs (append-only) |
| `checkpoints/weights_{025,050,075,100}.pth` | Weights at each fraction of training |
| `eval/ckptNNN/predictions/` | Label maps for every validation scan, for every checkpoint |
| `eval/ckpt100/probabilities/` | Probability maps (float16) of the final checkpoint |
| `eval/ckptNNN/per_case_region.csv`, `per_lesion.csv`, `confusion*.csv`, `official_*` | Scores and diagnostics |
| `COMPLETE.json` | Written only when every expected file exists |
| `report.html` | Per-run report (contains MRI images; stays private) |

A run is about 2.5 GB. Local files are the master copy. W&B (team project "Ped-BraTS", never public) is a live
mirror of the logs.

## 12. Where it runs

Two servers that cannot reach each other. Each model's folds are spread over both
([`code/plans/main_study.json`](../code/plans/main_study.json)):

| | gpu-a240 | blackwell |
|---|---|---|
| GPU | A100 40 GB (+ 10/20 GB MIG slices) | 1× RTX PRO 5000 Blackwell 48 GB |
| Scheduling | SLURM, shared with other users | No scheduler, shared; **one of our jobs at a time** |
| ResEnc L | folds 0, 2, 4 | folds 1, 3 |
| PrimusV2-B | folds 0–3 | fold 4 |
| SegMamba | folds 0, 1, 3 | folds 2, 4 |
| Swin UNETR | folds 2, 4 | folds 0, 1, 3 |

- **Queue:** one per machine ([`final/queue.py`](../code/pbts/final/queue.py)). It runs fold 0 of every model
  first and applies the rescue rule. On gpu-a240 it picks the smallest GPU that fits the model's measured memory
  peak.
- **Handover between machines:**
  - When a fold 0 finishes, [`scripts/fold0_sync.py`](../scripts/fold0_sync.py) copies its verdict into the plan
    as `fold0_external`, so the other machine can start that model's later folds.
  - [`scripts/gather_runs.py`](../scripts/gather_runs.py) copies finished blackwell runs to gpu-a240 and verifies
    them file by file. All runs end up in one place for the analysis and the locked test.
- **The two machines give the same results** (checked 2026-11-21 by
  [`tools/data_fingerprint.py`](../code/tools/data_fingerprint.py) and
  [`tools/cross_machine_check.py`](../code/tools/cross_machine_check.py)):
  - the prepared data is identical (4,415 of 4,415 files);
  - the same ResEnc weights predict the same on both: a median of 2 voxels differ per scan, and Dice between
    machines is ≥ 0.9997.
- **Environments:** exact package lists are in [`code/envs/`](../code/envs/). blackwell runs PyTorch 2.7.1 instead of
  2.5.1 because its GPU needs it; every difference is listed in `blackwell_env_deviations.txt`.

## 13. Changing the plan

- **Shared code** (data, split, input, augmentation, loss, prediction, scoring) must not change while the study
  runs. Runs trained before and after a change would no longer be comparable.
- **A change to `protocol.py`** must be a new, dated amendment in that file, stating the reason and the evidence.
  Each run records the exact code version it ran with.
- **Fixes that don't touch results** (logging, reports, analysis, dashboard) are fine at any time; reports and
  analysis can be regenerated from the saved predictions.
- **Workflow:** work on a branch and open a pull request. The servers run what is on `main`.

## 14. Known loose ends

- `config.TUNING_FOLD` and its comment ("≤3 tuning runs per model") come from an earlier plan. Nothing uses them:
  the plan is the original recipe plus one rescue (section 6).
- The third-party repositories used on the servers live in `repos/` (not in git):
  - `ge-xing/SegMamba` @ cff3597, with its own CUDA extensions built by
    [`tools/build_segmamba.sh`](../code/tools/build_segmamba.sh);
  - `rachitsaluja/BraTS-2023-Metrics` @ 43c9052, the reference for the lesion-wise tests;
  - MONAI's BRATS21 Swin UNETR code, the source of its recipe.

## 15. Suggested review checklist

1. Locked test: is it really never read during development? See `data/guard.py` and every `check_not_locked` call.
2. Leakage: is there any path from validation scans into training or checkpoint choice? See `train/base.py` and
   `train/monitor.py`.
3. Preprocessing: is it identical for training, validation, locked test and external scans? See
   `tests/test_external.py` and `tests/test_nnunet_data.py`.
4. Recipes: do `protocol.RECIPES` match each model's original code, and are the deviations complete?
5. Scoring: does `eval/official.py` call the official scorer correctly, and does `lesionwise2023.py` match the
   original? See `tests/test_eval.py`.
6. Statistics: are bootstrap and paired tests correct, with children (not scans) as the unit? See
   `analysis/stats.py` and `tests/test_analysis.py`.
