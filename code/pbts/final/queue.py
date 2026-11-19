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

# GPU choice: the smallest GPU the run fits on (user rule, 2026-11-20). Usable memory per SLURM gres type, GiB.
# The a100_2g.10gb_mps slice is shared with other users through MPS and is never used.
GPU_GB = {"a100_2g.10gb": 9.75, "a100_3g.20gb": 19.5, "a100": 39.4}
PEAK_MARGIN, CONTEXT_GB = 1.15, 1.0   # measured torch peak x 1.15 + 1 GiB (CUDA context, allocator fragmentation)


def fitting_gres(m):
    """GPU types a plan entry fits on, smallest first, from its measured training peak (`peak_GB`, torch
    max_memory_allocated at the configuration and batch size it will train with). Unmeasured -> full A100 only.
    An explicit `gres` list in the plan overrides this."""
    if m.get("gres"):
        return list(m["gres"])
    peak = m.get("peak_GB")
    if peak is None:
        return ["a100"]
    need = peak * PEAK_MARGIN + CONTEXT_GB
    fits = [g for g, cap in sorted(GPU_GB.items(), key=lambda kv: kv[1]) if need <= cap]
    if not fits:
        raise SystemExit(f"{m['model']}: needs ~{need:.1f} GiB, more than any GPU here")
    return fits


def gres_type(job_gres: str) -> str:
    """'gres/gpu:a100_3g.20gb:1' -> 'a100_3g.20gb'."""
    return job_gres.split("gpu:", 1)[-1].rsplit(":", 1)[0]


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


ACTIVE_S = 45 * 60


def run_dir(model, tag, k):
    return C.RUNS_DIR / model / tag / f"fold_{k}"


def recently_written(d: Path) -> bool:
    """True if anything under the run dir changed in the last ACTIVE_S seconds (covers runs started outside the
    queue, e.g. the pilots, and the scoring phase after training)."""
    if not d.exists():
        return False
    now = time.time()
    for p in d.rglob("*"):
        try:
            if now - p.stat().st_mtime < ACTIVE_S:
                return True
        except FileNotFoundError:
            pass
    return False


def lock_alive(d: Path, jobs) -> bool:
    lk = d / "RUNNING.lock"
    if not lk.exists():
        return False
    try:
        job = json.load(open(lk)).get("slurm_job_id")
    except Exception:
        return True
    return job is None or any(j["id"] == str(job) for j in jobs)


def tag_for(model, base, samples):
    """'default' if the existing fold-0 run (e.g. the pilot) used this training length, else a length-specific tag."""
    d = run_dir(model, base, 0)
    if (d / "run_config.json").exists() and json.load(open(d / "run_config.json")).get("samples") != samples:
        return f"{base}_s{samples // 1000}k"
    return base


def status(model, tag, k, jobs):
    d = run_dir(model, tag, k)
    name = f"{model}__{tag}__f{k}"
    if any(j["name"] == name for j in jobs) or lock_alive(d, jobs) or recently_written(d):
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


def plan_runs(plan, jobs, state, samples):
    """Ordered list of (model, tag, fold, lr_factor, gres_options) that should run now."""
    todo, report = [], {}
    for m in plan["models"]:
        model, gres = m["model"], fitting_gres(m)
        base = tag_for(model, "default", samples)
        s0 = status(model, base, 0, jobs)
        tag, lr = base, 1.0
        if s0 == "rescue":
            tag, lr = tag_for(model, "rescue_lr3", samples), 1 / 3
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
    cmd = ["sbatch", "--parsable", "-J", name, f"--gres=gpu:{gres}:1"] + (["-c", "14"] if gres != "a100" else []) + \
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
        todo, report = plan_runs(plan, jobs, state, samples)
        busy = {g: sum(1 for j in jobs if gres_type(j["gres"]) == g) for g in GPU_GB}
        free_slots = plan.get("max_jobs", 2) - len(jobs)
        for model, tag, k, lr, gres_opts in todo:
            if free_slots <= 0:
                break
            name = f"{model}__{tag}__f{k}"
            n_prev = len(state["submissions"].get(name, []))
            if status(model, tag, k, jobs) == "stopped" and n_prev >= 2:
                report[model] = report.get(model, "") + f" | {name} stopped twice: needs a human"
                continue
            free = [g for g in gres_opts if busy.get(g, 0) == 0]   # smallest fitting GPU that is free
            if not free:
                continue
            g = free[0]
            jid = submit(model, tag, k, lr, g, samples, plan.get("wandb", True), a.dry_run)
            state["submissions"].setdefault(name, []).append({"job": jid, "gres": g, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            busy[g] = busy.get(g, 0) + 1
            free_slots -= 1
        if not a.dry_run:
            json.dump(state, open(STATE, "w"), indent=1)
        log("status: " + " || ".join(f"{m}: {r}" for m, r in report.items()))
        if a.dry_run or (not todo and not jobs):
            break
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
