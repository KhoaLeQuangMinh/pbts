"""Run queue for the main study: keeps at most MAX_JOBS of our SLURM jobs busy, in the pre-declared order,
and applies the protocol's rescue rule. Light-weight polling loop; safe to stop and restart at any time
(state is re-derived from the run directories and squeue on every poll).

    python -m pbts.final.queue plans/main_study.json --dry_run      # show what it would submit
    nohup python -m pbts.final.queue plans/main_study.json > ../logs/queue.out 2>&1 < /dev/null &

Per model: fold 0 (tag 'default') first. If it writes RESCUE_TRIGGER.json, ONE rescue run (tag 'rescue_lr3',
lr / 3) is submitted for fold 0; if that passes, folds 1-4 use the rescue recipe; if it also fails, the model
is reported as not trainable and skipped. Folds 1-4 start only after fold 0 passed. A job that ends without
COMPLETE/INCOMPLETE/RESCUE (crash, node failure) is resubmitted once (it resumes); then it is left for a human.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from pbts import config as C
from pbts import protocol as P

STATE = C.ROOT / "logs" / "queue_state.json"
LOG = C.ROOT / "logs" / "queue.log"
SLICE = "a100_3g.20gb"


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def our_jobs():
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%b"], capture_output=True, text=True).stdout
    jobs = []
    for l in out.splitlines():
        i, name, state, gres = l.split("|")
        jobs.append({"id": i, "name": name, "state": state, "gres": gres})
    return jobs


def run_dir(model, tag, k):
    return C.RUNS_DIR / model / tag / f"fold_{k}"


def status(model, tag, k, jobs):
    d = run_dir(model, tag, k)
    name = f"{model}__{tag}__f{k}"
    if any(j["name"] == name for j in jobs):
        return "queued_or_running"
    if (d / "RESCUE_TRIGGER.json").exists():
        return "rescue"
    if (d / "COMPLETE.json").exists():
        return "complete"
    if (d / "INCOMPLETE.json").exists():
        return "incomplete"
    if (d / "run_config.json").exists():
        return "stopped"            # started before, ended without a verdict (crash / killed / requeue lost)
    return "todo"


def plan_runs(plan, jobs, state):
    """Ordered list of (model, tag, fold, lr_factor, gres_options) that should run now."""
    todo, report = [], {}
    for m in plan["models"]:
        model, gres = m["model"], m["gres"]
        s0 = status(model, "default", 0, jobs)
        tag, lr = "default", 1.0
        if s0 == "rescue":
            tag, lr = "rescue_lr3", 1 / 3
            s0 = status(model, tag, 0, jobs)
            if s0 == "rescue":
                report[model] = "NOT TRAINABLE under the protocol (fold 0 failed with default and lr/3)"
                continue
        if s0 in ("todo", "stopped"):
            todo.append((model, tag, 0, lr, gres))
            report[model] = f"fold 0 ({tag}) to run"
            continue
        if s0 != "complete":
            report[model] = f"fold 0 ({tag}): {s0}"
            continue
        sts = {k: status(model, tag, k, jobs) for k in range(1, C.N_INNER)}
        for k, s in sts.items():
            if s in ("todo", "stopped"):
                todo.append((model, tag, k, lr, gres))
        report[model] = f"recipe '{tag}': " + ", ".join(f"f{k}={s}" for k, s in {0: s0, **sts}.items())
    return todo, report


def submit(model, tag, k, lr, gres, samples, wandb, dry):
    name = f"{model}__{tag}__f{k}"
    cmd = ["sbatch", "--parsable", "-J", name, f"--gres=gpu:{gres}:1"] + (["-c", "14"] if gres == SLICE else []) + \
          ["slurm/train.sbatch", "--model", model, "--fold", str(k), "--tag", tag, "--samples", str(samples),
           "--lr_factor", repr(lr)] + ([] if wandb else ["--no_wandb"])
    if dry:
        log("DRY " + " ".join(cmd))
        return "dry"
    jid = subprocess.run(cmd, capture_output=True, text=True, cwd=C.ROOT / "code", check=True).stdout.strip()
    log(f"submitted {name} on {gres}: job {jid}")
    return jid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--poll_s", type=int, default=120)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    samples = plan.get("samples") or P.TRAINING_SAMPLES
    if not samples:
        raise SystemExit("training length not fixed: set protocol.TRAINING_SAMPLES (after the pilot) or plan['samples']")
    state = json.load(open(STATE)) if STATE.exists() else {"submissions": {}}
    while True:
        jobs = our_jobs()
        todo, report = plan_runs(plan, jobs, state)
        busy = {g: sum(1 for j in jobs if g in j["gres"]) for g in ("a100:", SLICE)}
        free_slots = plan.get("max_jobs", 2) - len(jobs)
        for model, tag, k, lr, gres_opts in todo:
            if free_slots <= 0:
                break
            name = f"{model}__{tag}__f{k}"
            n_prev = len(state["submissions"].get(name, []))
            if status(model, tag, k, jobs) == "stopped" and n_prev >= 2:
                report[model] = report.get(model, "") + f" | {name} stopped twice: needs a human"
                continue
            free = [g for g in gres_opts if busy.get("a100:" if g == "a100" else g, 0) == 0]
            if not free:
                continue
            g = free[0]
            jid = submit(model, tag, k, lr, g, samples, plan.get("wandb", True), a.dry_run)
            state["submissions"].setdefault(name, []).append({"job": jid, "gres": g, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            busy["a100:" if g == "a100" else g] = busy.get("a100:" if g == "a100" else g, 0) + 1
            free_slots -= 1
        if not a.dry_run:
            json.dump(state, open(STATE, "w"), indent=1)
        log("status: " + " || ".join(f"{m}: {r}" for m, r in report.items()))
        if a.dry_run or (not todo and not jobs):
            break
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
