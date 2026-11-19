"""Study status as JSON (read-only: run directories, the plan, squeue, nvidia-smi, df). Used by the local dashboard
(scripts/dashboard.py). Never touches a run.

    python -m pbts.final.status [--plan plans/main_study.json]
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from pbts import config as C
from pbts import protocol as P
from pbts.final import queue as Q

TEST_TAGS = ("smoke", "resume_test", "throughput", "smoke128")
PROVISIONAL_SAMPLES = 100000


def _jsonl_tail(p, n=12):
    p = Path(p)
    if not p.exists():
        return []
    lines = p.read_text().strip().splitlines()[-n:]
    out = []
    for l in lines:
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


def _ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp()


def squeue_all():
    fmt = "%i|%j|%T|%b|%M|%L|%R|%C|%m"
    out = subprocess.run(["squeue", "--me", "-h", "-o", fmt], capture_output=True, text=True).stdout
    jobs = []
    for l in out.splitlines():
        i, name, state, gres, elapsed, left, reason, cpus, mem = l.split("|")
        jobs.append({"id": i, "name": name, "state": state, "gres": gres.replace("gres/gpu:", ""), "elapsed": elapsed,
                     "time_left": left, "reason": reason, "cpus": cpus, "mem": mem})
    return jobs


def run_job_ids(d: Path):
    ids = []
    for h in d.glob("provenance/**/host.json"):
        try:
            j = json.load(open(h)).get("slurm", {}).get("SLURM_JOB_ID")
            if j:
                ids.append(j)
        except Exception:
            pass
    lk = d / "RUNNING.lock"
    if lk.exists():
        try:
            ids.append(str(json.load(open(lk)).get("slurm_job_id")))
        except Exception:
            pass
    return ids


def describe_run(d: Path, jobs):
    r = {"dir": str(d.relative_to(C.RUNS_DIR)), "exists": d.exists()}
    if not d.exists():
        return r
    cfg = json.load(open(d / "run_config.json")) if (d / "run_config.json").exists() else {}
    r.update({k: cfg.get(k) for k in ("model", "tag", "fold", "samples", "configuration", "lr_factor")})
    ids = set(run_job_ids(d))
    job = next((j for j in jobs if j["id"] in ids), None)
    r["job"] = job
    summ = d / "provenance" / "model_summary.json"
    n_ep = json.load(open(summ))["num_epochs"] if summ.exists() else None
    ep = _jsonl_tail(d / "epochs.jsonl", 12)
    if ep:
        last = ep[-1]
        r["epoch"], r["num_epochs"] = last["epoch"] + 1, n_ep
        r["last"] = {k: last.get(k) for k in ("train_loss", "val_loss", "val_pseudo_dice_ET", "val_pseudo_dice_NET",
                                             "val_pseudo_dice_CC", "val_pseudo_dice_ED", "samples_per_s", "gpu_peak_GB",
                                             "lr", "time")}
        if len(ep) >= 2:
            per = (_ts(ep[-1]["time"]) - _ts(ep[0]["time"])) / (len(ep) - 1)
            r["s_per_epoch"] = per
            if n_ep:
                r["eta_h"] = max(0, n_ep - last["epoch"] - 1) * per / 3600 + 0.35   # + checkpoint scoring
    mon = _jsonl_tail(d / "monitor.jsonl", 1)
    if mon:
        s = mon[-1]["summary"]
        r["monitor"] = {"epoch": mon[-1]["epoch"], **{k: s.get(k) for k in s if k.endswith("_dice_when_present")
                                                     or k.endswith("_false_alarm_rate_when_absent")}}
    ev = _jsonl_tail(d / "events.jsonl", 3)
    r["last_event"] = ev[-1]["event"] + " @ " + ev[-1]["time"] if ev else None
    if (d / "COMPLETE.json").exists():
        r["state"] = "complete"
        st = json.load(open(d / "COMPLETE.json"))
        r["final"] = {f: v.get("WT_off_global_dsc_mean") for f, v in (st.get("checkpoint_scores") or {}).items()}
        r["finished"] = st.get("time")
    elif (d / "RESCUE_TRIGGER.json").exists():
        r["state"] = "rescue"
        r["rescue"] = json.load(open(d / "RESCUE_TRIGGER.json"))
    elif (d / "INCOMPLETE.json").exists():
        r["state"] = "incomplete"
        r["missing"] = json.load(open(d / "INCOMPLETE.json")).get("missing")
    elif job and job["state"] == "PENDING":
        r["state"] = "queued"
    elif job or Q.recently_written(d):
        r["state"] = "running"
        if (d / "eval").exists() and not (d / "COMPLETE.json").exists() and r.get("epoch") == r.get("num_epochs"):
            r["phase"] = "scoring checkpoints"
    else:
        r["state"] = "stopped"
    return r


def study_grid(plan, jobs, samples):
    rows = []
    for m in plan["models"]:
        model = m["model"]
        base = Q.tag_for(model, "default", samples)
        d0 = Q.run_dir(model, base, 0)
        r0 = describe_run(d0, jobs)
        tag = base
        rescue = None
        if r0.get("state") == "rescue":
            rescue = Q.tag_for(model, "rescue_lr3", samples)
            tag = rescue
        cells = []
        for k in range(C.N_INNER):
            d = Q.run_dir(model, tag, k)
            r = describe_run(d, jobs) if d.exists() else {"dir": str(d.relative_to(C.RUNS_DIR)), "exists": False}
            if not d.exists():
                r["state"] = "todo" if k == 0 else "blocked"
                if k > 0:
                    f0 = describe_run(Q.run_dir(model, tag, 0), jobs)
                    r["state"] = "todo" if f0.get("state") == "complete" else "blocked"
                    r["blocked_by"] = None if r["state"] == "todo" else f"fold 0 ({f0.get('state', 'not started')})"
            cells.append(r)
        rows.append({"model": model, "tag": tag, "rescue_tag": rescue, "gres": Q.fitting_gres(m), "peak_GB": m.get("peak_GB"),
                     "fold0_default": r0 if rescue else None, "folds": cells})
    return rows


def server_info():
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    g = []
    for l in gpus.strip().splitlines():
        i, n, mu, mt, u = [x.strip() for x in l.split(",")]
        g.append({"index": i, "name": n, "mem_used_GB": round(float(mu) / 1024, 1), "mem_total_GB": round(float(mt) / 1024, 1),
                  "util": None if u.startswith("[") else int(u)})
    df = subprocess.run(["df", "-BG", "/data"], capture_output=True, text=True).stdout.splitlines()[-1].split()
    runs_size = subprocess.run(["du", "-sh", str(C.RUNS_DIR)], capture_output=True, text=True).stdout.split("\t")[0]
    queue_alive = bool(subprocess.run(["pgrep", "-u", "khoalq", "-f", "pbts.final.queue"], capture_output=True, text=True).stdout.strip())
    node = subprocess.run(["sinfo", "-N", "-h", "-o", "%C|%m|%e"], capture_output=True, text=True).stdout.strip().split("|")
    return {"gpus": g, "data_disk": {"size": df[1], "used": df[2], "free": df[3], "use": df[4]}, "runs_size": runs_size,
            "queue_daemon_running": queue_alive,
            "node_cpus_alloc_idle_other_total": node[0] if node else None,
            "node_free_mem_GB": round(int(node[2]) / 1024, 1) if len(node) > 2 and node[2].isdigit() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(C.ROOT / "code/plans/main_study.json"))
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    samples = plan.get("samples") or P.TRAINING_SAMPLES
    jobs = squeue_all()
    grid = study_grid(plan, jobs, samples or PROVISIONAL_SAMPLES)
    planned = {c["dir"] for row in grid for c in row["folds"]} | \
              {row["fold0_default"]["dir"] for row in grid if row.get("fold0_default")}
    extra, tests = [], []
    for d in sorted(C.RUNS_DIR.glob("*/*/fold_*")):
        rel = str(d.relative_to(C.RUNS_DIR))
        if rel in planned:
            continue
        r = describe_run(d, jobs)
        (tests if any(t == d.parent.name for t in TEST_TAGS) else extra).append(r)
    known = {str(j) for row in grid for c in row["folds"] for j in run_job_ids(C.RUNS_DIR / c["dir"])} | \
            {str(j) for r in extra + tests for j in run_job_ids(C.RUNS_DIR / r["dir"])}
    other_jobs = [j for j in jobs if j["id"] not in known]
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "training_samples": samples,
           "training_samples_fixed": bool(samples), "provisional_samples": None if samples else PROVISIONAL_SAMPLES,
           "plan": plan, "grid": grid, "pilot_and_extra": extra, "tests": tests, "jobs": jobs, "other_jobs": other_jobs,
           "server": server_info(), "locked_test_done": (C.RUNS_DIR / "final_locked" / "FINAL_DONE.json").exists()}
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
