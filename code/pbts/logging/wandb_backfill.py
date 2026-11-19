"""Replay a finished (or running) run's LOCAL logs into the private W&B project, for runs started with
--no_wandb. Local files are never modified. Usage:  python -m pbts.logging.wandb_backfill <run_dir> [...]"""
import json
import sys
from pathlib import Path

from pbts import config as C
from pbts.logging.runlog import RunLogger, _jsonl_safe


def backfill(run_dir):
    run = Path(run_dir)
    cfg = json.load(open(run / "run_config.json"))
    lg = RunLogger(run, cfg, use_wandb=True)              # creates/resumes the W&B run (id in wandb_id.txt)
    if lg.wandb is None:
        raise RuntimeError("W&B unavailable; see events.jsonl")
    for rec in _jsonl_safe(run / "steps.jsonl"):
        lg._wb({f"step/{k}": v for k, v in rec.items() if isinstance(v, (int, float))}, rec.get("global_step"))
    for rec in _jsonl_safe(run / "epochs.jsonl"):
        lg._wb({f"epoch/{k}": v for k, v in rec.items() if isinstance(v, (int, float))}, rec.get("global_step"))
    pics = {p.stem: p for p in (run / "monitor_png").glob("*.png")}
    for rec in _jsonl_safe(run / "monitor.jsonl"):
        flat = {f"monitor/{k}": v for k, v in rec.get("summary", {}).items()}
        if C.WANDB_LOG_IMAGES:
            import wandb
            for stem, p in pics.items():
                if stem.startswith(f"ep{rec['epoch']:04d}_"):
                    flat[f"monitor_img/{stem.split('_', 1)[1]}"] = wandb.Image(str(p))
        lg._wb(flat, rec.get("global_step"))
    for rec in _jsonl_safe(run / "system.jsonl"):
        lg._wb({f"system/{k}": v for k, v in rec.items() if isinstance(v, (int, float))}, None)
    lg.wandb.summary["backfilled_from_local_logs"] = True
    lg.close()


if __name__ == "__main__":
    for d in sys.argv[1:]:
        backfill(d)
        print("backfilled", d)
