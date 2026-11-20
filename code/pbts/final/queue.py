"""Run queue for the main study: keeps this machine's GPU slots busy, in the pre-declared order, and applies the
protocol's rescue rule. Light-weight polling loop; safe to stop and restart at any time (state is re-derived from
the run directories and the job list on every poll). One queue per machine; each runs only the models the plan
assigns to it ("machine" per model; the machines cannot see each other's files).

    python -m pbts.final.queue plans/main_study.json --machine gpu-a240 --dry_run     # show what it would submit
    nohup python -m pbts.final.queue plans/main_study.json --machine blackwell > ../logs/queue.out 2>&1 < /dev/null &

Backends: "slurm" (gpu-a240: sbatch/squeue, smallest fitting GPU type) and "local" (blackwell: no scheduler; a
job is started as its own process with local/run_job.sh, only when the GPU has enough free memory, because other
users share it).

Per model: fold 0 (tag 'default') first. If it writes RESCUE_TRIGGER.json, ONE rescue run (tag 'rescue_lr3',
lr / 3) is submitted for fold 0; if that passes, folds 1-4 use the rescue recipe; if it also fails, the model
is reported as not trainable and skipped. Folds 1-4 start only after fold 0 passed (a fold 0 that ran on the
other machine is declared in the plan as "fold0_external"). A job that ends without COMPLETE/INCOMPLETE/RESCUE
(crash, node failure) is resubmitted (it resumes) at most twice; then it is left for a human.
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
JOBS_DIR = C.ROOT / "logs" / "jobs"          # local backend: one JSON per started job (pid, args, log)

# GPU choice: the smallest GPU the run fits on (user rule, 2026-11-20). Usable memory per SLURM gres type, GiB.
# The a100_2g.10gb_mps slice is shared with other users through MPS and is never used.
GPU_GB = {"a100_2g.10gb": 9.75, "a100_3g.20gb": 19.5, "a100": 39.4}
PEAK_MARGIN, CONTEXT_GB = 1.15, 1.0   # measured torch peak x 1.15 + 1 GiB (CUDA context, allocator fragmentation)
DEFAULT_MACHINES = {"gpu-a240": {"backend": "slurm", "max_jobs": 2}}


def need_GB(m):
    return None if m.get("peak_GB") is None else m["peak_GB"] * PEAK_MARGIN + CONTEXT_GB


def fitting_gres(m):
    """SLURM GPU types a plan entry fits on, smallest first, from its measured training peak (`peak_GB`, torch
    max_memory_allocated at the configuration and batch size it will train with). Unmeasured -> full A100 only.
    An explicit `gres` list in the plan overrides this."""
    if m.get("gres"):
        return list(m["gres"])
    need = need_GB(m)
    if need is None:
        return ["a100"]
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


# ------------------------------------------------------------------ job lists (per backend)
def _pid_alive(pid) -> bool:
    """Running (not a zombie) process on THIS machine."""
    try:
        st = open(f"/proc/{int(pid)}/stat").read()
        return st.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def slurm_jobs():
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%b"], capture_output=True, text=True).stdout
    jobs = []
    for l in out.splitlines():
        i, name, state, gres = l.split("|")
        jobs.append({"id": i, "name": name, "state": state, "gres": gres})
    return jobs


_CHILDREN = []                                # local backend: Popen objects, polled so finished jobs are reaped


def local_jobs():
    for p in list(_CHILDREN):
        if p.poll() is not None:
            _CHILDREN.remove(p)
    jobs = []
    for f in sorted(JOBS_DIR.glob("*.json")) if JOBS_DIR.exists() else []:
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if _pid_alive(j.get("pid")):
            jobs.append({"id": str(j["pid"]), "name": j["name"], "state": "RUNNING", "gres": "local"})
    return jobs


def gpu_free_GB():
    """Free memory on the local GPU (other users included), GiB."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return float(out[0]) / 1024 if out else 0.0


# ------------------------------------------------------------------ run state
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
        lock = json.load(open(lk))
    except Exception:
        return True
    job = lock.get("slurm_job_id")
    if job is None:                           # started without SLURM: alive only if its process still runs here
        return _pid_alive(lock.get("pid"))
    return any(j["id"] == str(job) for j in jobs)


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


def machine_of(m, k):
    """Machine a fold runs on. Plan entry 'machine' is a name, or {"default": name, "<fold>": name, ...}."""
    mm = m.get("machine", "gpu-a240")
    if isinstance(mm, dict):
        return mm.get(str(k), mm.get("default", "gpu-a240"))
    return mm


def assigned(m, machine):
    return any(machine_of(m, k) == machine for k in range(C.N_INNER))


def plan_runs(plan, jobs, state, samples, machine="gpu-a240"):
    """Ordered list of (model, tag, fold, lr_factor, plan_entry) that should run now on this machine."""
    todo, report = [], {}
    for x in plan.get("extra_runs", []):          # one-off runs outside the protocol grid (e.g. reproducibility)
        if x.get("machine", "gpu-a240") != machine:
            continue
        m = next(e for e in plan["models"] if e["model"] == x["model"])
        s = status(x["model"], x["tag"], x["fold"], jobs)
        report[f"{x['model']}/{x['tag']}"] = f"extra run f{x['fold']}: {s}"
        if s in ("todo", "stopped"):
            todo.append((x["model"], x["tag"], x["fold"], x.get("lr_factor", 1.0), m))
    for m in plan["models"]:
        if not assigned(m, machine):
            report[m["model"]] = f"runs on {m.get('machine', 'gpu-a240')}"
            continue
        if m.get("hold"):                   # e.g. not validated / memory not measured yet
            report[m["model"]] = "on hold: " + m["hold"]
            continue
        model = m["model"]
        f0_here = machine_of(m, 0) == machine
        # fold 0 on the other machine: its verdict is copied into the plan once known, as
        # {"verdict": "complete" | "not_trainable", "tag": "default" | "rescue_lr3", "lr_factor": 1.0 | 1/3, "where": ...}
        ext = None if f0_here else m.get("fold0_external")
        if not f0_here and not ext:
            report[model] = f"waiting for fold 0 on {machine_of(m, 0)}"
            continue
        if ext and ext.get("verdict") == "not_trainable":
            report[model] = f"NOT TRAINABLE under the protocol (fold 0 on {ext.get('where')})"
            continue
        base = ext["tag"] if ext else tag_for(model, "default", samples)
        s0 = ext["verdict"] if ext else status(model, base, 0, jobs)
        tag, lr = base, (ext.get("lr_factor", 1.0) if ext else 1.0)
        if s0 == "rescue":
            tag, lr = tag_for(model, "rescue_lr3", samples), 1 / 3
            s0 = status(model, tag, 0, jobs)
            if s0 == "rescue":
                report[model] = "NOT TRAINABLE under the protocol (fold 0 failed with default and lr/3)"
                continue
        if s0 in ("todo", "stopped"):
            todo.append((model, tag, 0, lr, m))
            report[model] = f"fold 0 ({tag}) to run"
            continue
        if s0 != "complete":
            report[model] = f"fold 0 ({tag}): {s0}"
            continue
        sts = {k: status(model, tag, k, jobs) for k in range(1, C.N_INNER) if machine_of(m, k) == machine}
        for k, s in sts.items():
            if s in ("todo", "stopped"):
                todo.append((model, tag, k, lr, m))
        f0 = f"{s0} on {ext['where']}" if ext else s0
        report[model] = f"recipe '{tag}': " + ", ".join(f"f{k}={s}" for k, s in {0: f0, **sts}.items()) + \
            (f" (other folds on the other machine)" if len(sts) < C.N_INNER - 1 else "")
    # Fold 0 of every model before any later fold (2026-11-20): the rescue rule is decided on fold 0, so a broken
    # model is found before days are spent on its other folds, and fold 0 alone gives a first 4-model comparison.
    # Stable sort: plan order is kept within each fold. Scheduling only; no effect on any result.
    todo.sort(key=lambda t: t[2])
    return todo, report


def train_args(model, tag, k, lr, samples, wandb):
    return ["--model", model, "--fold", str(k), "--tag", tag, "--samples", str(samples), "--lr_factor", repr(lr)] + \
           ([] if wandb else ["--no_wandb"])


def submit_slurm(name, args, gres, dry):
    cmd = ["sbatch", "--parsable", "-J", name, f"--gres=gpu:{gres}:1"] + (["-c", "14"] if gres != "a100" else []) + \
          ["slurm/train.sbatch"] + args
    if dry:
        log("DRY " + " ".join(cmd))
        return "dry"
    jid = subprocess.run(cmd, capture_output=True, text=True, cwd=C.ROOT / "code", check=True).stdout.strip()
    log(f"submitted {name} on {gres}: job {jid}")
    return jid


def submit_local(name, args, dry):
    cmd = ["bash", str(C.ROOT / "code/local/run_job.sh")] + args
    if dry:
        log("DRY " + " ".join(cmd))
        return "dry"
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.ROOT / "logs" / f"local-{name}-{time.strftime('%Y%m%dT%H%M%S')}.out"
    p = subprocess.Popen(cmd, stdout=open(out, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         cwd=C.ROOT / "code", start_new_session=True)
    _CHILDREN.append(p)
    json.dump({"pid": p.pid, "name": name, "args": args, "log": str(out), "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(JOBS_DIR / f"{name}.json", "w"), indent=1)
    log(f"started {name} locally: pid {p.pid}, log {out.name}")
    return str(p.pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--machine", default="gpu-a240")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--poll_s", type=int, default=120)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    mc = {**DEFAULT_MACHINES, **plan.get("machines", {})}[a.machine]
    local = mc["backend"] == "local"
    samples = plan.get("samples") or P.TRAINING_SAMPLES
    if not samples:
        raise SystemExit("training length not fixed: set protocol.TRAINING_SAMPLES (after the pilot) or plan['samples']")
    state = json.load(open(STATE)) if STATE.exists() else {"submissions": {}}
    log(f"queue started on {a.machine} ({mc['backend']}, max {mc.get('max_jobs', 1)} jobs)")
    while True:
        jobs = local_jobs() if local else slurm_jobs()
        todo, report = plan_runs(plan, jobs, state, samples, a.machine)
        busy = {g: sum(1 for j in jobs if gres_type(j["gres"]) == g) for g in GPU_GB}
        free_slots = mc.get("max_jobs", 1) - len(jobs)
        for model, tag, k, lr, m in todo:
            if free_slots <= 0:
                break
            name = f"{model}__{tag}__f{k}"
            n_prev = len(state["submissions"].get(name, []))
            if status(model, tag, k, jobs) == "stopped" and n_prev >= 2:
                report[model] = report.get(model, "") + f" | {name} stopped twice: needs a human"
                continue
            args = train_args(model, tag, k, lr, samples, plan.get("wandb", True))
            if local:
                need = need_GB(m) or mc.get("gpu_GB", 0)
                if need > mc.get("gpu_GB", 0):
                    report[model] = report.get(model, "") + f" | needs {need:.1f} GiB > this GPU"
                    continue
                free_gb = gpu_free_GB()
                if free_gb < need:                 # someone else is on the GPU: wait, never squeeze in
                    report[model] = report.get(model, "") + f" | waiting: GPU has {free_gb:.1f} GiB free, needs {need:.1f}"
                    break
                jid, g = submit_local(name, args, a.dry_run), "local"
            else:
                free = [g for g in fitting_gres(m) if busy.get(g, 0) == 0]   # smallest fitting GPU that is free
                if not free:
                    continue
                g = free[0]
                jid = submit_slurm(name, args, g, a.dry_run)
                busy[g] = busy.get(g, 0) + 1
            state["submissions"].setdefault(name, []).append({"job": jid, "gres": g, "machine": a.machine,
                                                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            free_slots -= 1
        if not a.dry_run:
            json.dump(state, open(STATE, "w"), indent=1)
        log("status: " + " || ".join(f"{m}: {r}" for m, r in report.items()))
        if a.dry_run or (not todo and not jobs):
            break
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
