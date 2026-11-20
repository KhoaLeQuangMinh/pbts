"""One run = one model x one inner fold x one tag. Resumes automatically.

    python -m pbts.train.run --model nnunet_resenc_l --fold 0 --tag default --samples 100000
    python -m pbts.train.run --model nnunet_resenc_l --fold 0 --tag smoke --smoke          # 20 steps end-to-end

After training, every saved fraction checkpoint (25/50/75/100%) predicts ALL validation cases of the fold with
the shared predictor (8-flip TTA) and is scored by the shared evaluator. Label maps are kept for every
checkpoint; probability maps for the final one (and every checkpoint with --keep_all_probs). The run is marked
COMPLETE only if every expected file exists.

Run directory: runs/<model>/<tag>/fold_<k>/
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

from pbts import config as C
from pbts import protocol as P

MODELS = {
    "nnunet_resenc_l": "models.nnunet_resenc.trainer:PBTS_ResEncL",
    "primus_v2b": "models.primus.trainer:PBTS_PrimusV2B",
    "segmamba": "models.segmamba.trainer:PBTS_SegMamba",
    "swinunetr": "models.swinunetr.trainer:PBTS_SwinUNETR",
}
DEFAULT_CONFIG = "3d_wholehead"


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_run_config(a):
    split = json.load(open(C.HEAD_DIR / "study_split.json"))
    fold = split["inner_folds"][a.fold]
    run_dir = C.RUNS_DIR / a.model / a.tag / f"fold_{a.fold}"
    from pbts.train.monitor import select_monitor_cases
    cfg = {
        "model": a.model, "tag": a.tag, "fold": a.fold, "configuration": a.config,
        "run_name": f"{a.model}|{a.tag}|fold{a.fold}", "run_dir": str(run_dir),
        "samples": a.samples, "lr_factor": a.lr_factor, "batch_size": a.batch_size,
        "train_cases": fold["train"], "val_cases": fold["val"],
        "split_fingerprint": split["fingerprints"], "monitor_cases": select_monitor_cases(fold["val"]),
        "rescue_checks": bool(a.fold == 0 and a.lr_factor == 1.0 and not a.smoke),
        "recipe": P.RECIPES[a.model], "shared_training": P.SHARED_TRAINING, "protocol_sha": _sha(
            {"recipes": P.RECIPES, "shared": P.SHARED_TRAINING, "rescue": P.RESCUE}),
        "smoke": {"iters": a.smoke_iters, "epochs": a.smoke_epochs} if a.smoke else None,
        "no_wandb": a.no_wandb, "seed": 12345 + a.fold, "torch_compile": False,
    }
    return run_dir, cfg


def _load_trainer_class(name):
    mod, cls = MODELS[name].split(":")
    return getattr(importlib.import_module(mod), cls)


def train(run_dir: Path, cfg: dict):
    os.environ["nnUNet_raw"] = str(C.NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(C.NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(run_dir / "nnunet")
    os.environ["PBTS_RUN_CONFIG"] = str(run_dir / "run_config.json")
    # torch.compile off for every model (identical setting; avoids compiler failures on custom kernels)
    os.environ["nnUNet_compile"] = "false"
    import numpy as np
    import random
    import torch
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    from batchgenerators.utilities.file_and_folder_operations import load_json
    pre = C.NNUNET_PREPROCESSED / C.DATASET_NAME
    plans = load_json(str(pre / "pbtsWholeHeadPlans.json"))
    trainer_cls = _load_trainer_class(cfg["model"])
    probe = trainer_cls.__mro__  # noqa: F841  (fails early if import is broken)
    out = run_dir / "nnunet" / C.DATASET_NAME
    latest = list(out.glob(f"*/fold_{cfg['fold']}/checkpoint_latest.pth"))
    final = list(out.glob(f"*/fold_{cfg['fold']}/checkpoint_final.pth"))
    plans["continue_training"] = bool(latest or final)
    trainer = trainer_cls(plans=plans, configuration=cfg["configuration"], fold=cfg["fold"],
                          dataset_json=load_json(str(pre / "dataset.json")), device=torch.device("cuda"))
    if final:
        trainer.initialize()
        trainer.frame_xyz = tuple(reversed(trainer.configuration_manager.patch_size))
        return trainer
    if latest:
        trainer.load_checkpoint(str(latest[0]))
    trainer.run_training()
    return trainer


def score_checkpoints(trainer, run_dir: Path, cfg: dict, keep_all_probs=False):
    import torch
    from pbts.eval.evaluate import evaluate
    from pbts.infer.predict import predict_cases
    cases = cfg["val_cases"] if not cfg["smoke"] else cfg["val_cases"][:2]
    results = {}
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    for f in (25, 50, 75, 100):
        w = run_dir / "checkpoints" / f"weights_{f:03d}.pth"
        out = run_dir / "eval" / f"ckpt{f:03d}"
        if not w.exists() or (out / "EVAL_DONE.json").exists():
            continue
        state = torch.load(w, map_location="cuda", weights_only=False)["network_weights"]
        (trainer.network._orig_mod if hasattr(trainer.network, "_orig_mod") else trainer.network).load_state_dict(state)
        t0 = time.time()
        predict_cases(trainer.network, cases, out, tta=True, save_probs=(f == 100 or keep_all_probs),
                      log=trainer.print_to_log_file, frame_xyz=trainer.frame_xyz)
        df = evaluate(cases, out / "predictions", out, workers=12)
        results[f] = {"predict_eval_s": time.time() - t0,
                      "WT_off_global_dsc_mean": float(df[df.region == "WT"].off_global_dsc.mean())}
        if trainer.pbts_log:
            trainer.pbts_log.event("checkpoint_scored", fraction=f, **results[f])
    return results


def rescue_floor(run_dir: Path, cfg: dict):
    import pandas as pd
    if not cfg["rescue_checks"]:
        return None
    df = pd.read_csv(run_dir / "eval" / "ckpt100" / "per_case_region.csv")
    wt = float(df[df.region == "WT"].off_global_dsc.mean())
    passed = wt >= 0.5
    if not passed:
        json.dump({"trigger": "floor", "detail": f"mean WT whole-volume Dice {wt:.4f} < 0.5",
                   "rule": P.RESCUE["triggers"]["floor"]}, open(run_dir / "RESCUE_TRIGGER.json", "w"), indent=1)
    return {"WT_dice": wt, "passed": passed}


def completeness(run_dir: Path, cfg: dict):
    n_val = len(cfg["val_cases"]) if not cfg["smoke"] else 2
    need = ["run_config.json", "provenance/run_config.json", "provenance/host.json", "provenance/model_summary.json",
            "steps.jsonl" if not cfg["smoke"] else "epochs.jsonl", "epochs.jsonl", "case_losses.jsonl",
            "monitor.jsonl", "events.jsonl", "checkpoints/weights_100.pth"]
    missing = [p for p in need if not (run_dir / p).exists()]
    for f in (25, 50, 75, 100):
        if not (run_dir / "checkpoints" / f"weights_{f:03d}.pth").exists():
            continue
        e = run_dir / "eval" / f"ckpt{f:03d}"
        if not (e / "EVAL_DONE.json").exists():
            missing.append(f"eval/ckpt{f:03d}/EVAL_DONE.json")
        n_pred = len(list((e / "predictions").glob("*.nii.gz")))
        if n_pred != n_val:
            missing.append(f"eval/ckpt{f:03d}/predictions ({n_pred}/{n_val})")
    n_prob = len(list((run_dir / "eval/ckpt100/probabilities").glob("*.npz")))
    if n_prob != n_val:
        missing.append(f"eval/ckpt100/probabilities ({n_prob}/{n_val})")
    return missing


def _process_alive(info):
    """Is the process that wrote a lock still alive? (same host: check its pid; SLURM job: ask squeue)."""
    job = info.get("slurm_job_id")
    if job:
        import subprocess
        out = subprocess.run(["squeue", "-h", "-j", str(job)], capture_output=True, text=True).stdout
        return bool(out.strip())
    if info.get("host") == os.uname().nodename and info.get("pid"):
        try:
            os.kill(int(info["pid"]), 0)
            return True
        except OSError:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", type=int, default=P.TRAINING_SAMPLES)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--lr_factor", type=float, default=1.0, help="1.0, or 1/3 for the protocol rescue run")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_iters", type=int, default=20)
    ap.add_argument("--smoke_epochs", type=int, default=1)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--keep_all_probs", action="store_true")
    a = ap.parse_args()
    if a.lr_factor not in (1.0, 1 / 3):
        sys.exit("lr_factor must be 1.0 or 1/3 (protocol: one rescue, lr / 3)")
    if a.samples is None:
        if not a.smoke:
            sys.exit("--samples is required until protocol.TRAINING_SAMPLES is fixed by the pilot")
        a.samples = 40
    run_dir, cfg = build_run_config(a)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        old = json.load(open(cfg_path))
        # This run's own recipe + the shared settings must match; protocol_sha (ALL recipes) is recorded but not
        # enforced, so a dated amendment for ANOTHER model does not block resuming this run (2026-11-20).
        keys = ["model", "tag", "fold", "configuration", "samples", "lr_factor", "batch_size", "train_cases",
                "val_cases", "recipe", "shared_training"]
        norm = lambda v: json.loads(json.dumps(v, default=str))   # tuples in memory vs lists in the JSON file
        diff = [k for k in keys if norm(old.get(k)) != norm(cfg.get(k))]
        if diff:
            sys.exit(f"run_config.json exists with different {diff}: refusing to mix two runs in {run_dir}")
        cfg["monitor_cases"] = old["monitor_cases"]
    json.dump(cfg, open(cfg_path, "w"), indent=1, default=str)

    lock = run_dir / "RUNNING.lock"                            # one process per run directory
    if lock.exists():
        old = json.load(open(lock))
        same_job = old.get("slurm_job_id") and old.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")
        if not same_job and _process_alive(old):
            sys.exit(f"{run_dir} is locked by {old}: another process is running this run")
    json.dump({"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(), "host": os.uname().nodename,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(lock, "w"))
    try:
        trainer = train(run_dir, cfg)
        scores = score_checkpoints(trainer, run_dir, cfg, keep_all_probs=a.keep_all_probs)
        floor = rescue_floor(run_dir, cfg)
        try:
            from pbts.report.run_report import build
            build(run_dir)
        except Exception as e:                                   # a report bug must not lose a finished run
            if trainer.pbts_log:
                trainer.pbts_log.event("report_failed", error=repr(e)[:500])
        missing = completeness(run_dir, cfg)
        if not (run_dir / "report.html").exists():
            missing.append("report.html")
        status = {"complete": not missing, "missing": missing, "checkpoint_scores": scores, "rescue_floor": floor,
                  "rescue_triggered": (run_dir / "RESCUE_TRIGGER.json").exists(),
                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(status, open(run_dir / ("COMPLETE.json" if not missing else "INCOMPLETE.json"), "w"), indent=1)
        if trainer.pbts_log:
            trainer.pbts_log.event("run_finished", **{k: v for k, v in status.items() if k != "checkpoint_scores"})
            trainer.pbts_log.close(exit_code=0 if not missing else 2)
    finally:
        if lock.exists() and json.load(open(lock)).get("pid") == os.getpid():
            lock.unlink()
    print(json.dumps(status, indent=1))
    sys.exit(0 if not missing else 2)


if __name__ == "__main__":
    main()
