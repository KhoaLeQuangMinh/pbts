#!/usr/bin/env python3
"""Bring every finished main-study run from blackwell to gpu-a240, so all runs (predictions, probabilities, weights,
logs) end up in one place for the analysis and the final locked-test evaluation.

Copies runs/<model>/<tag>/fold_k that have COMPLETE.json on blackwell and none on gpu-a240 (main-study tags only:
default*, rescue_lr3*), as an exact replica (tar stream through this Mac; the servers cannot reach each other), then
verifies the copy file by file (relative path + size of every file must match). Read-only on blackwell.

    python3 scripts/gather_runs.py             # once
    python3 scripts/gather_runs.py --watch     # every 30 minutes
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ("blackwell", "/home/khoalq/pbts/runs"), ("gpu-a240", "/data/khoalq/pbts/runs")
LOG = ROOT / "dashboard" / "gather_runs.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ssh(host, cmd):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, cmd],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"{host}: {r.stderr.strip()[-300:]}")
    return r.stdout


def complete_runs(host, root):
    out = ssh(host, f"cd {root} 2>/dev/null && ls -d */*/fold_*/COMPLETE.json 2>/dev/null || true")
    return {p.rsplit("/", 1)[0] for p in out.split() if p.split("/")[1].startswith(("default", "rescue_lr3"))}


def manifest(host, root, run):
    return ssh(host, f"cd {root}/{run} && find . -type f -printf '%P %s\\n' | sort")


def copy(run):
    t = time.time()
    model_tag = run.rsplit("/", 1)[0]
    ssh(DST[0], f"mkdir -p {DST[1]}/{model_tag}")
    src = subprocess.Popen(["ssh", "-o", "BatchMode=yes", SRC[0], f"tar -C {SRC[1]} -cf - {run}"], stdout=subprocess.PIPE)
    dst = subprocess.run(["ssh", "-o", "BatchMode=yes", DST[0], f"tar -C {DST[1]} -xf -"], stdin=src.stdout,
                         capture_output=True, text=True)
    src.wait()
    if src.returncode or dst.returncode:
        raise RuntimeError(f"copy of {run} failed: {dst.stderr[-300:]}")
    a, b = manifest(*SRC, run), manifest(*DST, run)
    if a != b:
        raise RuntimeError(f"copy of {run} differs from the original (file list / sizes)")
    n = len(a.splitlines())
    log(f"copied {run} blackwell -> gpu-a240: {n} files verified, {(time.time() - t) / 60:.1f} min")


def once():
    todo = sorted(complete_runs(*SRC) - complete_runs(*DST))
    for run in todo:
        try:
            copy(run)
        except Exception as ex:
            log(f"error: {ex}")
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=1800)
    a = ap.parse_args()
    while True:
        try:
            once()
        except Exception as ex:
            log(f"error: {ex}")
        if not a.watch:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
