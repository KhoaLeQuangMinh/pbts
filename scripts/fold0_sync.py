#!/usr/bin/env python3
"""Copy each fold-0 verdict to the machine that runs that model's other folds (the two servers cannot see each other).

For every model whose fold 0 has finished on one machine while some of its folds 1-4 are assigned to the other machine,
and whose plan entry has no 'fold0_external' yet: read the verdict where fold 0 ran (COMPLETE.json, or the protocol's
rescue: RESCUE_TRIGGER.json -> the 'rescue_lr3' run), write 'fold0_external' into code/plans/main_study.json, commit,
sync both servers, and restart the queue on the other machine so it picks up the change (queues are stateless; running
jobs are not touched).

    python3 scripts/fold0_sync.py               # once
    python3 scripts/fold0_sync.py --dry_run     # show what it would do
    python3 scripts/fold0_sync.py --watch       # every 15 minutes

It also restarts a machine's run queue when that machine rebooted after the queue last logged (the queue then
resumes the interrupted runs from their last checkpoint). A queue that stopped by itself is left alone.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "code/plans/main_study.json"
ROOTS = {"gpu-a240": "/data/khoalq/pbts", "blackwell": "/home/khoalq/pbts"}
N_FOLDS = 5
LOG = ROOT / "dashboard" / "fold0_sync.log"
PENDING = ROOT / "dashboard" / "fold0_sync_pending.json"   # hosts whose queue still has to receive a plan update


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ssh(host, cmd, check=True):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, cmd],
                       capture_output=True, text=True, timeout=180)
    if check and r.returncode != 0:
        raise RuntimeError(f"{host}: {r.stderr.strip()[-300:]}")
    return r.stdout


def machine_of(m, k):
    mm = m.get("machine", "gpu-a240")
    return mm.get(str(k), mm.get("default", "gpu-a240")) if isinstance(mm, dict) else mm


def files(host, model, tag):
    out = ssh(host, f"ls {ROOTS[host]}/runs/{model}/{tag}/fold_0 2>/dev/null || true")
    return set(out.split())


def verdict(host, model):
    """The protocol's fold-0 outcome, or None while undecided (running / needs a human)."""
    d = files(host, model, "default")
    if "COMPLETE.json" in d:
        return {"verdict": "complete", "tag": "default", "lr_factor": 1.0}
    if "RESCUE_TRIGGER.json" in d:
        r = files(host, model, "rescue_lr3")
        if "COMPLETE.json" in r:
            return {"verdict": "complete", "tag": "rescue_lr3", "lr_factor": 1 / 3}
        if "RESCUE_TRIGGER.json" in r:
            return {"verdict": "not_trainable", "tag": "rescue_lr3", "lr_factor": 1 / 3}
        return None                                  # rescue run still to run / running
    if "INCOMPLETE.json" in d:
        log(f"{model}: fold 0 on {host} is INCOMPLETE -> needs a human, not declared")
    return None


def restart_queue(host):
    r = ROOTS[host]
    ssh(host, 'pkill -u khoalq -f "^../env/bin/python -m pbts.final.queue" ; sleep 2; true', check=False)
    subprocess.run(["ssh", "-f", "-o", "BatchMode=yes", host,
                    f"cd {r}/code && setsid nohup ../env/bin/python -m pbts.final.queue plans/main_study.json "
                    f"--machine {host} >> ../logs/queue.out 2>&1 < /dev/null &"], check=True, timeout=60)
    time.sleep(8)
    alive = ssh(host, 'ps -u khoalq -o args | grep "^../env/bin/python -m pbts.final.queue" || true').strip()
    if not alive:
        raise RuntimeError(f"queue on {host} did not come back up")
    log(f"queue on {host} restarted: running")


def queue_killed_by_reboot(host):
    """True if no run queue is running on `host` and the machine booted after the queue last wrote to its log, i.e.
    the queue was killed by a reboot rather than having stopped by itself. Raises if the host is unreachable."""
    out = ssh(host, f'ps -u khoalq -o args | grep -c "^../env/bin/python -m pbts.final.queue"; grep "^btime" /proc/stat; '
                    f'stat -c %Y {ROOTS[host]}/logs/queue.out 2>/dev/null || echo 0').split()
    n_queues, boot, last_log = int(out[0]), int(out[2]), int(out[3])
    return n_queues == 0 and boot > last_log


def load_pending():
    return set(json.load(open(PENDING))) if PENDING.exists() else set()


def save_pending(hosts):
    PENDING.write_text(json.dumps(sorted(hosts)) + "\n")


def deliver(hosts):
    """Sync the code (with the updated plan) to each host and restart its queue. A host that cannot be reached stays
    pending in PENDING and is retried on every pass until it succeeds, so a verdict declared while a server is down
    still reaches its queue once the server is back."""
    pending = load_pending() | set(hosts)
    save_pending(pending)
    for h in sorted(pending):
        try:
            r = subprocess.run(["bash", "scripts/sync.sh", h], cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"sync failed: {r.stderr.strip()[-200:]}")
            restart_queue(h)
            pending.discard(h)
            save_pending(pending)
        except Exception as ex:
            log(f"{h}: plan update not delivered yet, will retry ({ex})")


def once(dry):
    plan = json.load(open(PLAN))
    changed, restart = [], set()
    for m in plan["models"]:
        f0 = machine_of(m, 0)
        others = {machine_of(m, k) for k in range(1, N_FOLDS)} - {f0}
        if not others or m.get("fold0_external", {}).get("verdict") in ("complete", "not_trainable"):
            continue
        try:
            v = verdict(f0, m["model"])
        except Exception as ex:                      # one unreachable server must not block the other models
            log(f"{m['model']}: cannot read fold 0 on {f0} ({ex})")
            continue
        if v is None:
            continue
        m["fold0_external"] = {**v, "where": f0, "declared": time.strftime("%Y-%m-%dT%H:%M:%S") + " by scripts/fold0_sync.py"}
        changed.append(f"{m['model']}: fold 0 on {f0} -> {v['verdict']} ({v['tag']})")
        restart |= others
    if changed:
        log("new fold-0 verdicts: " + "; ".join(changed) + (" [dry run]" if dry else ""))
        if dry:
            return True
        PLAN.write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
        msg = "Plan: fold-0 verdicts copied to the other machine\n\n" + "\n".join(changed) + \
              "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
        subprocess.run(["git", "add", str(PLAN)], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "-q"], cwd=ROOT, check=False)
    for h in ROOTS:                                  # a reboot kills the queue; bring it back (its jobs then resume)
        try:
            if queue_killed_by_reboot(h):
                log(f"{h} rebooted after its run queue last logged -> restarting the queue "
                    f"(interrupted runs resume from their last checkpoint)" + (" [dry run]" if dry else ""))
                restart.add(h)
        except Exception:
            pass                                     # unreachable: checked again next pass
    if (changed or restart or load_pending()) and not dry:
        deliver(restart)
    return bool(changed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=900)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            once(a.dry_run)
        except Exception as ex:
            log(f"error: {ex}")
        if not a.watch:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
