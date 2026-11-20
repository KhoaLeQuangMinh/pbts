#!/usr/bin/env python3
"""Local study dashboard: what still needs to run, what is running, what is done.

    python3 scripts/dashboard.py            # refresh once and open the page
    python3 scripts/dashboard.py --watch    # keep refreshing every 60 s (Ctrl+C to stop)

Fetches read-only status from the server (python -m pbts.final.status over ssh) and writes
dashboard/index.html, which reloads itself every minute. No scan images; nothing on the server is changed.
"""
import argparse
import html
import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "index.html"
CACHE = ROOT / "dashboard" / "last_status.json"
REMOTE = "cd /data/khoalq/pbts/code && ../env/bin/python -m pbts.final.status"
STATE_LABEL = {"complete": "done", "running": "running", "queued": "queued", "todo": "to run", "blocked": "waiting",
               "rescue": "rescue triggered", "incomplete": "incomplete", "stopped": "stopped"}
NAMES = {"nnunet_resenc_l": "nnU-Net ResEnc L", "primus_v2b": "PrimusV2-B", "segmamba": "SegMamba", "swinunetr": "Swin UNETR"}


def fetch():
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "gpu-a240", REMOTE],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-400:] or f"ssh exit {r.returncode}")
    return json.loads(r.stdout.strip().splitlines()[-1])


REASONS = {"QOSMaxJobsPerUserLimit": "waiting for a free job slot (your 2-job limit)",
           "Resources": "waiting for a free GPU", "Priority": "waiting behind other users' jobs",
           "None": "", "(None)": ""}


def reason(r):
    r = (r or "").strip("()")
    return REASONS.get(r, r)


def e(x):
    return html.escape("" if x is None else str(x))


def pct(a, b):
    return 0 if not b else max(0, min(100, 100 * a / b))


def fmt_h(h):
    if h is None:
        return "—"
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def dice_chips(last):
    if not last:
        return ""
    out = []
    for n in ("ET", "NET", "CC", "ED"):
        v = last.get(f"val_pseudo_dice_{n}")
        out.append(f'<span class="chip">{n} {("—" if v is None or v != v else f"{v:.2f}")}</span>')
    return "".join(out)


def run_card(r, title):
    job = r.get("job") or {}
    ep, n = r.get("epoch"), r.get("num_epochs")
    last = r.get("last") or {}
    mon = r.get("monitor") or {}
    mon_txt = " · ".join(f"{k.split('_')[0]} {v:.2f}" for k, v in mon.items()
                         if k.endswith("_dice_when_present") and v == v and v is not None)
    phase = r.get("phase")
    return f"""
    <article class="card">
      <header><b>{e(title)}</b><span class="tag">{e(job.get('gres', ''))}{' · job ' + e(job['id']) if job.get('id') else ''}</span></header>
      <div class="bar"><i style="width:{pct(ep or 0, n)}%"></i></div>
      <div class="row"><span>epoch {e(ep)} / {e(n)}{' · ' + e(phase) if phase else ''}</span><span>ETA {fmt_h(r.get('eta_h'))}</span></div>
      <div class="row muted"><span>{(r.get('s_per_epoch') or 0):.0f} s/epoch · {(last.get('samples_per_s') or 0):.2f} samples/s · {(last.get('gpu_peak_GB') or 0):.1f} GB</span>
        <span>elapsed {e(job.get('elapsed', '—'))}</span></div>
      <div class="row"><span>train {(last.get('train_loss') or 0):.3f} · val {(last.get('val_loss') or 0):.3f}</span><span>{dice_chips(last)}</span></div>
      {f'<div class="row muted"><span>monitor (epoch {mon.get("epoch")}) Dice when present: {e(mon_txt)}</span></div>' if mon_txt else ''}
    </article>"""


def cell(r, length_fixed=True):
    st = r.get("state", "todo")
    sub = ""
    if st == "running":
        sub = f"{r.get('epoch', 0)}/{r.get('num_epochs', '?')} · {fmt_h(r.get('eta_h'))}"
    elif st == "complete" and r.get("final"):
        wt = r["final"].get("100") or r["final"].get(100)
        sub = f"WT Dice {wt:.3f}" if wt is not None else ""
    elif st == "queued":
        sub = reason((r.get("job") or {}).get("reason", ""))
    elif st == "todo" and not length_fixed:
        sub = "after the training length is fixed"
    elif st == "blocked":
        sub = r.get("blocked_by") or ""
    elif st == "rescue":
        sub = (r.get("rescue") or {}).get("trigger", "")
    elif st in ("incomplete", "stopped"):
        sub = "needs a look"
    return f'<td class="s-{st}"><b>{STATE_LABEL.get(st, st)}</b><small>{e(sub)}</small></td>'


def next_actions(s):
    acts = []
    grid = s["grid"]
    if not s["training_samples_fixed"]:
        pilot = grid[0]["folds"][0] if grid else {}
        acts.append(f"Waiting for the whole-head pilot to fix the training length "
                    f"(ETA {fmt_h(pilot.get('eta_h'))}). Main-study runs are shown with a provisional "
                    f"{s['provisional_samples']:,} samples.")
    elif not s["server"]["queue_daemon_running"]:
        todo = sum(1 for row in grid for c in row["folds"] if c.get("state") in ("todo", "stopped"))
        if todo:
            acts.append(f"The run queue is not running and {todo} run(s) are ready: start it on the server.")
    for row in grid:
        for k, c in enumerate(row["folds"]):
            if c.get("state") in ("incomplete", "stopped"):
                acts.append(f"{NAMES.get(row['model'], row['model'])} fold {k}: {c['state']} — check {c['dir']}")
        if row.get("rescue_tag"):
            acts.append(f"{NAMES.get(row['model'], row['model'])}: rescue triggered on fold 0 "
                        f"({(row['fold0_default'] or {}).get('rescue', {}).get('trigger')}); continuing with lr/3.")
    for r in s["pilot_and_extra"]:
        if r.get("state") in ("incomplete", "stopped"):
            acts.append(f"{r['dir']}: {r['state']}")
    pend = [j for j in s["jobs"] if j["state"] == "PENDING"]
    if pend:
        acts.append("Pending SLURM jobs: " + ", ".join(f"{j['name']} ({reason(j['reason'])})" for j in pend))
    return acts or ["Nothing needs attention."]


CSS = """
:root{--bg:#f7f6f2;--fg:#1c1c1a;--muted:#6d6c66;--line:#e1dfd8;--card:#ffffff;--accent:#2f5d8a;
 --done:#d9efdf;--done-b:#2e7d4f;--run:#dce8f7;--run-b:#2f5d8a;--queue:#f3ecd7;--queue-b:#9a7414;
 --todo:#efeee9;--todo-b:#8a8983;--warn:#f8dfd5;--warn-b:#b4442a}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--fg:#ecebe6;--muted:#a19f98;--line:#32312d;--card:#1e1e1c;
 --accent:#8db6e0;--done:#1d3326;--done-b:#7fcb98;--run:#1c2a3b;--run-b:#8db6e0;--queue:#332b16;--queue-b:#e0bf62;
 --todo:#262624;--todo-b:#8f8d86;--warn:#3a221b;--warn-b:#f0937a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 56px}
h1{font-size:20px;margin:0}h2{font-size:15px;margin:28px 0 10px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.muted{color:var(--muted)}small{display:block;color:var(--muted);font-size:11.5px;margin-top:2px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile b{font-size:22px;display:block}.tile span{color:var(--muted);font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card header{display:flex;justify-content:space-between;gap:8px;margin-bottom:8px}.tag{color:var(--muted);font-size:12px}
.row{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:6px;font-size:13px}
.bar{height:8px;background:var(--todo);border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;background:var(--run-b)}
.chip{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 7px;margin-left:4px;font-size:12px}
.tablewrap{overflow-x:auto}table{border-collapse:separate;border-spacing:4px;width:100%;min-width:640px}
th{font-weight:600;text-align:left;font-size:12.5px;color:var(--muted);padding:2px 6px}
td{border-radius:8px;padding:8px 10px;vertical-align:top;font-size:13px;background:var(--card);border:1px solid var(--line)}
td.s-complete{background:var(--done);border-color:var(--done-b)}td.s-running{background:var(--run);border-color:var(--run-b)}
td.s-queued{background:var(--queue);border-color:var(--queue-b)}td.s-todo{background:var(--todo)}
td.s-blocked{background:var(--todo);color:var(--muted)}td.s-rescue,td.s-incomplete,td.s-stopped{background:var(--warn);border-color:var(--warn-b)}
table.plain{border-spacing:0}table.plain td,table.plain th{background:none;border:0;border-bottom:1px solid var(--line);border-radius:0;padding:6px 8px}
ul.acts{margin:0;padding-left:18px}ul.acts li{margin:4px 0}
.banner{background:var(--warn);border:1px solid var(--warn-b);border-radius:10px;padding:10px 12px;margin-top:12px}
details summary{cursor:pointer;color:var(--muted)}
"""


def render(s, error=None):
    grid = s["grid"]
    cells = [c for row in grid for c in row["folds"]]
    n_total = len(cells)
    count = lambda st: sum(1 for c in cells if c.get("state") == st)
    running = [(f"{NAMES.get(row['model'], row['model'])} · fold {k} · {row['tag']}", c)
               for row in grid for k, c in enumerate(row["folds"]) if c.get("state") == "running"]
    running += [(r["dir"], r) for r in s["pilot_and_extra"] if r.get("state") == "running"]
    gpus = "".join(f"<tr><td>GPU {e(g['index'])}</td><td>{e(g['name'])}</td><td>{g['mem_used_GB']} / {g['mem_total_GB']} GB</td>"
                   f"<td>{'—' if g['util'] is None else str(g['util']) + '%'}</td></tr>" for g in s["server"]["gpus"])
    jobs = "".join(f"<tr><td>{e(j['id'])}</td><td>{e(j['name'])}</td><td>{e(j['state'])}</td><td>{e(j['gres'])}</td>"
                   f"<td>{e(j['elapsed'])}</td><td>{e(j['time_left'])}</td><td>{e(reason(j['reason']) if j['state'] == 'PENDING' else j['reason'])}</td></tr>" for j in s["jobs"]) \
        or "<tr><td colspan=7 class='muted'>No jobs.</td></tr>"
    matrix = "".join(f"<tr><th>{e(NAMES.get(row['model'], row['model']))}<small>{e(row['tag'])}"
                     f"{' · rescue (lr/3)' if row.get('rescue_tag') else ''}</small></th>"
                     + "".join(cell(c, s["training_samples_fixed"]) for c in row["folds"]) + "</tr>" for row in grid)
    extra = "".join(f"<tr><td>{e(r['dir'])}</td><td>{e(STATE_LABEL.get(r.get('state'), r.get('state')))}</td>"
                    f"<td>{e(r.get('epoch'))}/{e(r.get('num_epochs'))}</td><td>{fmt_h(r.get('eta_h')) if r.get('state') == 'running' else ''}</td>"
                    f"<td>{e(r.get('last_event'))}</td></tr>" for r in s["pilot_and_extra"]) \
        or "<tr><td colspan=5 class='muted'>None.</td></tr>"
    tests = "".join(f"<tr><td>{e(r['dir'])}</td><td>{e(STATE_LABEL.get(r.get('state'), r.get('state')))}</td><td>{e(r.get('last_event'))}</td></tr>"
                    for r in s["tests"])
    srv = s["server"]
    samples = f"{s['training_samples']:,}" if s["training_samples_fixed"] else f"not fixed yet (provisional {s['provisional_samples']:,})"
    banner = f'<div class="banner">Could not refresh: {e(error)}. Showing data from {e(s["generated"])}.</div>' if error else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60"><title>PBTS study status</title><style>{CSS}</style></head><body><main>
<div class="top"><h1>PBTS study status</h1><span class="muted">server status at {e(s['generated'])} · page reloads every minute</span></div>
{banner}
<div class="tiles">
 <div class="tile"><b>{count('complete')}/{n_total}</b><span>main-study runs done</span></div>
 <div class="tile"><b>{count('running')}</b><span>running</span></div>
 <div class="tile"><b>{count('queued')}</b><span>queued in SLURM</span></div>
 <div class="tile"><b>{count('todo')}</b><span>{'ready to run' if s['training_samples_fixed'] else 'to run (after length is fixed)'}</span></div>
 <div class="tile"><b>{count('blocked')}</b><span>waiting (on fold 0, or model on hold)</span></div>
 <div class="tile"><b style="font-size:15px">{e(samples)}</b><span>training length (samples)</span></div>
 <div class="tile"><b style="font-size:15px">{'running' if srv['queue_daemon_running'] else 'not running'}</b><span>run queue</span></div>
 <div class="tile"><b style="font-size:15px">{'done' if s['locked_test_done'] else 'not yet'}</b><span>locked test (once, at the end)</span></div>
</div>
<h2>Needs attention / next</h2><ul class="acts">{''.join(f'<li>{e(a)}</li>' for a in next_actions(s))}</ul>
<h2>Running now</h2><div class="cards">{''.join(run_card(r, t) for t, r in running) or '<p class="muted">Nothing running.</p>'}</div>
<h2>Main study: models × inner folds</h2><div class="tablewrap"><table><tr><th></th>{''.join(f'<th>fold {k}</th>' for k in range(5))}</tr>{matrix}</table></div>
<p class="muted" style="font-size:12px">Order: ResEnc → Primus → SegMamba → Swin UNETR. Folds 1–4 start only after fold 0 passes the rescue check. ResEnc fold 0 is the whole-head pilot run.</p>
<h2>Pilot and other runs</h2><div class="tablewrap"><table class="plain"><tr><th>run</th><th>state</th><th>epoch</th><th>ETA</th><th>last event</th></tr>{extra}</table></div>
<h2>SLURM jobs (yours)</h2><div class="tablewrap"><table class="plain"><tr><th>job</th><th>name</th><th>state</th><th>GPU</th><th>elapsed</th><th>time left</th><th>reason / node</th></tr>{jobs}</table></div>
<h2>Server</h2><div class="tablewrap"><table class="plain"><tr><th>GPU</th><th>model</th><th>memory</th><th>utilisation</th></tr>{gpus}</table></div>
<p class="muted">/data: {e(srv['data_disk']['used'])} used of {e(srv['data_disk']['size'])} ({e(srv['data_disk']['use'])}), runs folder {e(srv['runs_size'])} · node CPUs alloc/idle/other/total {e(srv['node_cpus_alloc_idle_other_total'])} · free RAM {e(srv['node_free_mem_GB'])} GB</p>
<details><summary>Test and smoke runs ({len(s['tests'])})</summary><table class="plain"><tr><th>run</th><th>state</th><th>last event</th></tr>{tests}</table></details>
</main></body></html>"""


def refresh():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        s = fetch()
        CACHE.write_text(json.dumps(s))
        OUT.write_text(render(s))
        return True
    except Exception as ex:
        if CACHE.exists():
            OUT.write_text(render(json.loads(CACHE.read_text()), error=str(ex)))
        else:
            OUT.write_text(f"<!doctype html><meta http-equiv='refresh' content='60'><p>Could not reach the server: {e(ex)}</p>")
        print("refresh failed:", ex, file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--no_open", action="store_true")
    a = ap.parse_args()
    refresh()
    if not a.no_open:
        webbrowser.open(OUT.as_uri())
    print(f"dashboard: {OUT}")
    while a.watch:
        time.sleep(a.every)
        ok = refresh()
        print(time.strftime("%H:%M:%S"), "refreshed" if ok else "refresh failed (showing last data)")


if __name__ == "__main__":
    main()
