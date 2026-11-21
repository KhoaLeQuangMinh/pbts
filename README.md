# pbts

Adult brain-tumour segmentation models, retrained under identical conditions on paediatric scans (BraTS-PEDs 2026,
294 scans), to learn where each succeeds and fails in children before designing a child-specific method.

Models, all trained from scratch with their authors' own optimiser recipes: **nnU-Net ResEnc L, PrimusV2-B, SegMamba,
Swin UNETR**. Everything else is shared by every model: data, split, whole-head input, augmentation, loss, training
length, prediction and scoring.

**Live progress of every run:** https://khoalequangminh.github.io/pbts/ (updated about every 10 minutes).

**Start with [docs/PLAN.md](docs/PLAN.md).** It lists every decision, why it was made, and where it lives in the code.
The binding training settings, with dated amendments, are in [code/pbts/protocol.py](code/pbts/protocol.py).

No patient data, images or metadata are stored in this repository (BraTS data-use terms). See *Data rules* below.

## Layout

```
code/
  pbts/                shared package: every model goes through the same code
    config.py          fixed constants: paths, labels, regions, frame size, split
    protocol.py        pre-declared training protocol: recipes, rescue rule, pilot, amendments
    data/              Kaggle download -> verified NIfTI (convert), split + locked-test guard (splits, guard),
                       head mask + crop + z-score (headmask, preprocess), nnU-Net plan (nnunet_setup)
    train/             trainer mixin (base), data loader padding (loader), monitor, one run = run.py
    infer/predict.py   shared prediction: 8-flip TTA, probabilities kept, no post-processing
    eval/              official BraTS 2026 scorer (official), 2023 lesion-wise metric, diagnostics, evaluate.py
    analysis/          out-of-fold collection, statistics, subgroups, cross-model comparison
    report/            per-run HTML report
    final/             run queue, status, ensembles, locked-test evaluation, unlabeled-scan prediction
    logging/           run logger (local JSON lines + W&B mirror)
  models/<model>/      one trainer per model (nnU-Net trainer + our mixin + that model's network)
  plans/main_study.json  which model and fold runs on which machine, plus fold-0 verdicts handed between machines
  tests/               checks that must pass before training
  tools/               one-off tools: pilot decision, data fingerprint, cross-machine check, SegMamba build
  envs/                exact package lists per server, environment setup
  slurm/, local/       job scripts for gpu-a240 (SLURM) and blackwell (no scheduler)
scripts/               run from a laptop: sync code to the servers, status dashboard, fold-0 handover, copying runs
docs/PLAN.md           the study plan
```

## Setting up a server

The code expects one project folder per server: `<root>/code` (this repo's `code/`), with `env/`, `data/`, `runs/`,
`repos/` and `secrets/` beside it. `config.py` finds `<root>` from its own location, or from `PBTS_ROOT`.

1. **Environments:** [code/envs/setup_env.sh](code/envs/setup_env.sh) builds `env/` (training) and `env_eval/` (the
   official scorer) from the exact package lists in `code/envs/`.
2. **Third-party code** in `<root>/repos/`:
   - `git clone https://github.com/ge-xing/SegMamba` and check out `cff3597`, then run
     [code/tools/build_segmamba.sh](code/tools/build_segmamba.sh);
   - `git clone https://github.com/rachitsaluja/BraTS-2023-Metrics` and check out `43c9052`, needed only for
     `tests/test_eval.py`.
3. **Secrets** (never in git) in `<root>/secrets/`: your own Kaggle API token, and optionally a W&B key and entity.
4. **Data:** accept the BraTS-PEDs data terms on Kaggle first. Then, from `code/`, run
   `../env/bin/python -m pbts.data.download`, followed by
   `MACHINE=<name> bash local/prepare_data.sh` ([prepare_data.sh](code/local/prepare_data.sh)). It runs convert,
   split, preprocess, the nnU-Net fingerprint, plan and preprocessing, then a fingerprint of the result.
5. **Tests:** from `code/`, run `../env/bin/python -m pytest -q tests/`. `test_train.py` and `test_nnunet_data.py`
   need `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` set; copy the `export` line from `prepare_data.sh`.
6. **Check against a reference machine:** `tools/data_fingerprint.py --compare` shows the prepared data is
   identical, and [code/local/validate_machine.sh](code/local/validate_machine.sh) runs the full validation.

## Running

From `code/` on a server:

```
python -m pbts.train.run --model nnunet_resenc_l --fold 0 --tag smoke --smoke     # 20-step end-to-end check
python -m pbts.train.run --model nnunet_resenc_l --fold 0 --tag default           # one run (resumes automatically)
python -m pbts.final.queue plans/main_study.json --machine gpu-a240 --dry_run     # what the queue would start
python -m pbts.report.run_report ../runs/nnunet_resenc_l/default/fold_0           # per-run report
python -m pbts.analysis.compare --models nnunet_resenc_l primus_v2b segmamba swinunetr   # once all folds are done
```

The main study runs through the queue, one per machine. `scripts/dashboard.py`, run on a laptop with ssh access to
both servers, shows the progress of every run.

## Data rules

- Never commit scans, labels, predictions, probability maps, reports, or anything derived from patient data;
  `.gitignore` blocks the usual file types.
- Per-run reports (`report.html`) contain MRI images: share them only privately, with people who have signed the
  BraTS-PEDs data agreement.
- The W&B project stays team-only, never public.
- The 60 locked test scans are scored once, at the end of the study, through `pbts.final.locked_test` only.

## Contributing

Work on a branch and open a pull request; the servers run what is on `main`. While the study is running:

- **Don't change shared code that affects results:** data, split, input, augmentation, loss, prediction, scoring.
  Runs before and after the change would not be comparable.
- **A change to `protocol.py`** needs a new dated amendment in that file, with the reason.
- **Logging, reports, analysis, the dashboard and tests** can change at any time; reports and analysis regenerate
  from the saved predictions.
