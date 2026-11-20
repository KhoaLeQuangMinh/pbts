"""Per-run HTML report: one self-contained file (charts and pictures embedded, no internet needed).

    python -m pbts.report.run_report <run_dir>          -> <run_dir>/report.html (+ copy in reports/runs/)

Built only from what the run saved (logs, checkpoint evaluations, predictions), so it can be regenerated any time.
Contains MRI pictures: keep it on the server / your own machine, never publish it.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pbts import config as C

REGIONS = list(C.REGIONS)
LABEL_COLORS = {1: (0.2, 0.4, 1.0), 2: (1.0, 0.45, 0.2), 3: (0.1, 0.75, 0.5), 4: (0.95, 0.75, 0.1)}


# ---------------------------------------------------------------- helpers
def _jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def _fig_png(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(src, alt=""):
    return f'<img src="{src}" alt="{alt}" loading="lazy">'


def _file_png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def patient(case):
    return case.rsplit("-", 1)[0]


def bootstrap_ci(values, cases, n=2000, seed=0):
    """Mean with a 95% CI, resampling PATIENTS (a child's repeat scans move together). NaNs are dropped first."""
    v = pd.Series(values, index=cases).dropna()
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    groups = v.groupby(v.index.map(patient)).apply(list)
    rng = np.random.default_rng(seed)
    keys = groups.index.to_numpy()
    means = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys), replace=True)
        vals = [x for k in pick for x in groups[k]]
        means.append(np.mean(vals))
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _fmt(m, lo=None, hi=None, d=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "—"
    return f"{m:.{d}f}" if lo is None or np.isnan(lo) else f"{m:.{d}f} <small>[{lo:.{d}f}, {hi:.{d}f}]</small>"


# ---------------------------------------------------------------- sections
def training_curves(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = pd.DataFrame(_jsonl(run / "epochs.jsonl"))
    if ep.empty:
        return "<p>No epochs logged.</p>"
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    a = ax[0, 0]; a.plot(ep.epoch, ep.train_loss, label="train total"); a.plot(ep.epoch, ep.val_loss, label="val total")
    a.plot(ep.epoch, ep.train_ce, label="train CE", alpha=.6); a.set_title("loss"); a.legend(fontsize=7)
    a = ax[0, 1]
    for n in ["ET", "NET", "CC", "ED"]:
        if f"val_pseudo_dice_{n}" in ep:
            a.plot(ep.epoch, ep[f"val_pseudo_dice_{n}"], label=n)
    a.plot(ep.epoch, ep.train_soft_dice, "k--", alpha=.5, label="train soft Dice")
    a.set_title("validation pseudo-Dice (nnU-Net, random crops)"); a.set_ylim(0, 1); a.legend(fontsize=7)
    a = ax[0, 2]; a.plot(ep.epoch, ep.lr); a.set_title("learning rate")
    a = ax[1, 0]; a.plot(ep.epoch, ep.grad_norm_mean, label="mean"); a.plot(ep.epoch, ep.grad_norm_max, label="max", alpha=.6)
    a2 = a.twinx(); a2.plot(ep.epoch, ep.clipped_fraction, "r:", label="clipped fraction"); a2.set_ylim(0, 1)
    a.set_title("gradient norm (pre-clip) · red: fraction of steps clipped"); a.legend(fontsize=7)
    a = ax[1, 1]; a.plot(ep.epoch, ep.samples_per_s, label="samples/s")
    a2 = a.twinx(); a2.plot(ep.epoch, ep.data_wait_s / ep.epoch_s.clip(lower=1e-9), "r:"); a2.set_ylim(0, 1)
    a.set_title("throughput · red: share of time waiting for data")
    a = ax[1, 2]; a.plot(ep.epoch, ep.gpu_peak_GB); a.set_title("GPU peak memory per epoch (GB)")
    for a in ax.flat:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.tight_layout()
    return _img(_fig_png(fig), "training curves")


def monitor_curves(run):
    import matplotlib.pyplot as plt
    mon = _jsonl(run / "monitor.jsonl")
    if not mon:
        return "<p>No monitor records.</p>"
    s = pd.DataFrame([{"epoch": m["epoch"], **m["summary"]} for m in mon])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for r in REGIONS:
        if f"{r}_dice_when_present" in s:
            ax[0].plot(s.epoch, s[f"{r}_dice_when_present"], marker=".", label=r)
        if f"{r}_false_alarm_rate_when_absent" in s and s[f"{r}_false_alarm_rate_when_absent"].notna().any():
            ax[1].plot(s.epoch, s[f"{r}_false_alarm_rate_when_absent"], marker=".", label=r)
    ax[0].set_title("monitor cases: Dice when the region is present"); ax[0].set_ylim(0, 1)
    ax[1].set_title("monitor cases: false-alarm rate when the region is absent"); ax[1].set_ylim(0, 1)
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=.3); a.set_xlabel("epoch")
    fig.tight_layout()
    pics = sorted((run / "monitor_png").glob("*.png"))
    by_case = {}
    for p in pics:
        by_case.setdefault(p.stem.split("_", 1)[1], []).append(p)
    strip = "".join(f"<figure>{_img(_file_png(v[-1]))}<figcaption>{k} · last monitor epoch</figcaption></figure>"
                    for k, v in by_case.items())
    return _img(_fig_png(fig)) + f'<div class="gallery">{strip}</div>'


def load_eval(run, frac=100):
    d = run / "eval" / f"ckpt{frac:03d}"
    if not (d / "EVAL_DONE.json").exists():
        return None, None, None
    return (pd.read_csv(d / "per_case_region.csv"), pd.read_csv(d / "per_lesion.csv"),
            pd.read_csv(d / "confusion.csv", index_col=0))


def length_curve(run):
    import matplotlib.pyplot as plt
    rows = []
    for f in (25, 50, 75, 100):
        df, _, _ = load_eval(run, f)
        if df is None:
            continue
        for r, d in df.groupby("region"):
            rows.append({"fraction": f, "region": r, "dsc": d.off_global_dsc.mean(), "f1": d.off_all_instance_f1.mean(),
                         "lw2023": d.lw2023_dice.mean()})
    if not rows:
        return "<p>No checkpoint evaluations.</p>"
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r, d in t.groupby("region"):
        for a, k, name in zip(ax, ["dsc", "f1", "lw2023"], ["official whole-volume Dice", "official detection F1",
                                                             "2023 lesion-wise Dice"]):
            a.plot(d.fraction, d[k], marker="o", label=r); a.set_title(name)
    for a in ax:
        a.set_xlabel("% of training"); a.set_xticks([25, 50, 75, 100]); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout()
    return _img(_fig_png(fig)) + "<p class='note'>Each point is a separately saved checkpoint of THIS run, scored on all " \
                                 "validation cases. It shows whether shorter training would have been enough.</p>"


def score_table(df):
    cases = df.case
    rows = []
    for r in REGIONS:
        d = df[df.region == r]
        pres, absn = d[d.gt_present], d[~d.gt_present]
        cells = {
            "region": r, "n present / absent": f"{len(pres)} / {len(absn)}",
            "official Dice": _fmt(*bootstrap_ci(d.off_global_dsc.values, d.case.values)),
            "official NSD": _fmt(*bootstrap_ci(d.off_global_nsd.values, d.case.values)),
            "official HD95": _fmt(*bootstrap_ci(d.off_global_hd95.values, d.case.values), d=1),
            "detection F1": _fmt(*bootstrap_ci(d.off_all_instance_f1.values, d.case.values)),
            "2023 lesion-wise Dice": _fmt(*bootstrap_ci(d.lw2023_dice.values, d.case.values)),
            "Dice when present": _fmt(*bootstrap_ci(pres.off_global_dsc.values, pres.case.values)),
            "false alarm when absent": f"{int((absn.presence == 'FP').sum())} / {len(absn)}" if len(absn) else "—",
            "missed when present": f"{int((pres.presence == 'FN').sum())} / {len(pres)}" if len(pres) else "—",
            "official cases counted": int(d.off_global_dsc.notna().sum()),
        }
        rows.append(cells)
    t = pd.DataFrame(rows)
    note = ("<p class='note'>Means with 95% CIs from resampling patients. Official scorer: a region absent in both GT "
            "and prediction is NaN and drops out of its mean ('official cases counted'); any prediction on an absent "
            "region scores 0. 2023 lesion-wise: absent in both scores 1, lesions ≤50 voxels are not counted.</p>")
    return t.to_html(index=False, escape=False, classes="t") + note


def detection_by_size(les):
    import matplotlib.pyplot as plt
    g = les[les.kind == "gt_lesion"].copy()
    if g.empty:
        return "<p>No GT lesions.</p>"
    bins = [0, 50, 200, 1000, 5000, 1e9]
    names = ["≤50", "51–200", "201–1k", "1k–5k", ">5k"]
    g["size"] = pd.cut(g.vol, bins, labels=names)
    t = g.groupby(["region", "size"], observed=False).detected.agg(mean="mean", n="size").reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.13
    for i, r in enumerate(REGIONS):
        d = t[t.region == r].set_index("size")["mean"].reindex(names)   # a region may have no GT lesions
        ax.bar(np.arange(len(names)) + i * w, d.fillna(0).to_numpy(), w, label=r)
    ax.set_xticks(np.arange(len(names)) + 2.5 * w); ax.set_xticklabels(names); ax.set_ylim(0, 1)
    ax.set_xlabel("GT lesion size (voxels = mm³)"); ax.set_ylabel("fraction detected"); ax.legend(fontsize=7, ncol=6)
    ax.grid(alpha=.3, axis="y")
    fp = les[les.kind == "pred_fp"]
    fp_t = fp.groupby("region").agg(n=("vol", "size"), median_vol=("vol", "median"),
                                    median_dist_mm=("dist_to_gt_mm", "median")).reset_index() if len(fp) else pd.DataFrame()
    counts = t.pivot(index="region", columns="size", values="n").reindex(REGIONS)
    return (_img(_fig_png(fig)) + "<h4>GT lesions per size bin</h4>" + counts.to_html(classes="t") +
            "<h4>False-positive pieces</h4>" + (fp_t.to_html(index=False, classes="t") if len(fp_t) else "<p>None.</p>"))


def confusion_section(conf):
    import matplotlib.pyplot as plt
    m = conf.values.astype(float)
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    names = [c.replace("pred_", "") for c in conf.columns]
    ax.set_xticks(range(5)); ax.set_xticklabels(names); ax.set_yticks(range(5)); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > .5 else "black")
    fig.colorbar(im, ax=ax, fraction=.046)
    ax.set_title("voxel label confusion inside the tumor union (row-normalised)", fontsize=9)
    return _img(_fig_png(fig))


def calibration_section(run, cases, n_vox=200_000, seed=0):
    import matplotlib.pyplot as plt
    pdir = run / "eval" / "ckpt100" / "probabilities"
    if not pdir.exists():
        return "<p>No probabilities saved.</p>"
    import nibabel as nib
    rng = np.random.default_rng(seed)
    conf_all, corr_all = [], []
    for c in cases:
        f = pdir / f"{c}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        probs = z["probs"].astype(np.float32)
        lo, hi = z["box_lo"], z["box_hi"]
        gt = np.asanyarray(nib.load(str(C.RAW_DIR / "labelsTr" / f"{c}.nii.gz")).dataobj)
        gt = gt[tuple(slice(a, b + 1) for a, b in zip(lo, hi))]
        pred = probs.argmax(0)
        conf = probs.max(0)
        m = (gt > 0) | (pred > 0)                       # tumor union: background voxels would swamp the curve
        idx = np.flatnonzero(m)
        if idx.size > n_vox // max(1, len(cases)):
            idx = rng.choice(idx, n_vox // max(1, len(cases)), replace=False)
        conf_all.append(conf.ravel()[idx]); corr_all.append((pred.ravel()[idx] == gt.ravel()[idx]))
    if not conf_all:
        return "<p>No probabilities found.</p>"
    conf, corr = np.concatenate(conf_all), np.concatenate(corr_all)
    bins = np.linspace(0.2, 1, 17)
    which = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
    acc = np.array([corr[which == b].mean() if (which == b).any() else np.nan for b in range(len(bins) - 1)])
    frac = np.array([(which == b).mean() for b in range(len(bins) - 1)])
    ece = np.nansum(frac * np.abs(acc - (bins[:-1] + bins[1:]) / 2))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot([0, 1], [0, 1], "k:"); ax.plot((bins[:-1] + bins[1:]) / 2, acc, marker="o")
    ax.set_xlabel("confidence of the predicted label"); ax.set_ylabel("fraction correct")
    ax.set_title(f"calibration inside the tumor union · ECE {ece:.3f}", fontsize=9); ax.grid(alpha=.3)
    return _img(_fig_png(fig))


def worst_cases(run, df, n=6):
    import matplotlib.pyplot as plt
    import nibabel as nib
    wt = df[df.region == "WT"].sort_values("off_global_dsc").head(n)
    out = []
    for _, r in wt.iterrows():
        c = r.case
        p = run / "eval" / "ckpt100" / "predictions" / f"{c}.nii.gz"
        if not p.exists():
            continue
        pred = np.asanyarray(nib.load(str(p)).dataobj)
        gt = np.asanyarray(nib.load(str(C.RAW_DIR / "labelsTr" / f"{c}.nii.gz")).dataobj)
        fl = nib.load(str(C.RAW_DIR / "imagesTr" / f"{c}_0000.nii.gz")).get_fdata(dtype=np.float32)
        t = (gt > 0) | (pred > 0)
        ctr = np.round(np.argwhere(t).mean(0)).astype(int) if t.any() else np.array(gt.shape) // 2
        lo, hi = np.percentile(fl[fl > 0], [1, 99.5]) if (fl > 0).any() else (0, 1)
        fig, ax = plt.subplots(2, 3, figsize=(8, 5.4))
        for row, lab in enumerate([gt, pred]):
            for col, sl in enumerate([(slice(None), slice(None), ctr[2]), (slice(None), ctr[1], slice(None)),
                                      (ctr[0], slice(None), slice(None))]):
                img = np.clip((fl[sl] - lo) / (hi - lo + 1e-6), 0, 1)
                rgb = np.stack([img] * 3, -1)
                for k, col_rgb in LABEL_COLORS.items():
                    mm = lab[sl] == k
                    rgb[mm] = .45 * rgb[mm] + .55 * np.array(col_rgb)
                ax[row, col].imshow(np.rot90(rgb)); ax[row, col].set_axis_off()
        ax[0, 0].set_title("ground truth", fontsize=8, loc="left"); ax[1, 0].set_title("prediction", fontsize=8, loc="left")
        fig.suptitle(f"{c} · WT Dice {r.off_global_dsc:.3f}", fontsize=9)
        fig.tight_layout()
        out.append(f"<figure>{_img(_fig_png(fig))}</figure>")
    return '<div class="gallery">' + "".join(out) + "</div><p class='note'>ET blue · NET orange · CC green · ED yellow</p>"


def hardest_training_cases(run, k=10):
    cl = _jsonl(run / "case_losses.jsonl")
    if not cl:
        return "<p>No per-case losses.</p>"
    last = cl[-max(1, len(cl) // 10):]
    agg = {}
    for rec in last:
        for c, v in rec["losses"].items():
            agg.setdefault(c, []).append(v)
    t = pd.Series({c: np.mean(v) for c, v in agg.items()}).sort_values(ascending=False).head(k)
    return t.rename("mean loss, last 10% of epochs").to_frame().to_html(classes="t")


CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--accent:#2f5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29c;--line:#33332f;--card:#1f1f1d;--accent:#8db6e0}
 img{filter:none}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:0 16px}
main{max-width:1180px;margin:0 auto;padding:24px 0 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h4{margin:16px 0 4px;font-size:13px;color:var(--muted)}
.meta{color:var(--muted);font-size:13px}.note{color:var(--muted);font-size:12px}
img{max-width:100%;height:auto;background:#fff;border-radius:4px}
table.t{border-collapse:collapse;font-size:12.5px;margin:6px 0;width:100%;overflow-x:auto;display:block}
table.t th,table.t td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px 16px;font-size:13px}
.kv div span{color:var(--muted)}.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;
 background:var(--accent);color:var(--bg)}.gallery{display:flex;flex-wrap:wrap;gap:8px}.gallery figure{margin:0;flex:1 1 360px}
figcaption{font-size:12px;color:var(--muted)}
"""


def build(run_dir):
    run = Path(run_dir)
    cfg = json.load(open(run / "run_config.json"))
    summ = json.load(open(run / "provenance" / "model_summary.json"))
    host = json.load(open(run / "provenance" / "host.json"))
    ver = json.load(open(run / "provenance" / "code_version.json")) if (run / "provenance/code_version.json").exists() else {}
    status = json.load(open(run / "COMPLETE.json")) if (run / "COMPLETE.json").exists() else \
        (json.load(open(run / "INCOMPLETE.json")) if (run / "INCOMPLETE.json").exists() else {"complete": False})
    events = _jsonl(run / "events.jsonl")
    rescue = json.load(open(run / "RESCUE_TRIGGER.json")) if (run / "RESCUE_TRIGGER.json").exists() else None
    df, les, conf = load_eval(run, 100)

    kv = {"model": cfg["model"], "tag": cfg["tag"], "fold": cfg["fold"], "configuration": cfg["configuration"],
          "training samples": cfg["samples"], "epochs × steps": f"{summ['num_epochs']} × {summ['iters_per_epoch']}",
          "batch": summ["batch_size"], "lr × rescue factor": f"{summ['initial_lr']:.2e} (× {cfg['lr_factor']})",
          "weight decay": summ["weight_decay"], "grad clip": summ["grad_clip"], "parameters": f"{summ['params'] / 1e6:.1f} M",
          "deep supervision": summ["deep_supervision"], "GPU": host.get("gpu"), "code commit": ver.get("commit", "?")[:10],
          "uncommitted files": ver.get("uncommitted_files", "?"), "protocol sha": cfg["protocol_sha"],
          "train / val cases": f"{len(cfg['train_cases'])} / {len(cfg['val_cases'])}",
          "started": events[0]["time"] if events else "?", "finished": events[-1]["time"] if events else "?"}
    head = "".join(f"<div><span>{k}</span><br>{v}</div>" for k, v in kv.items())
    badge = "complete" if status.get("complete") else "INCOMPLETE: " + ", ".join(status.get("missing", []))[:200]
    resc = f"<p><b>Rescue triggered:</b> {rescue['trigger']} — {rescue['detail']}</p>" if rescue else ""

    parts = [f"<h1>{cfg['run_name']}</h1><p class='meta'><span class='badge'>{badge}</span></p>{resc}<div class='kv'>{head}</div>",
             "<h2>Training</h2>", training_curves(run),
             "<h2>Monitor cases (every few epochs, no flips)</h2>", monitor_curves(run),
             "<h2>Training length</h2>", length_curve(run)]
    if df is not None:
        parts += ["<h2>Final scores (100% checkpoint, 8-flip prediction, raw — no post-processing)</h2>", score_table(df),
                  "<h2>Detection by lesion size</h2>", detection_by_size(les),
                  "<h2>Label confusion</h2>", confusion_section(conf),
                  "<h2>Calibration</h2>", calibration_section(run, sorted(df.case.unique())),
                  "<h2>Lowest whole-tumor Dice</h2>", worst_cases(run, df)]
    parts += ["<h2>Hardest training cases</h2>", hardest_training_cases(run),
              "<h2>Events</h2>", pd.DataFrame(events)[["time", "event"]].to_html(index=False, classes="t") if events else ""]
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{cfg['run_name']}</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(parts)}</main></body></html>")
    out = run / "report.html"
    out.write_text(html)
    copy = C.REPORTS_DIR / "runs" / f"{cfg['model']}__{cfg['tag']}__fold{cfg['fold']}.html"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(html)
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    print(build(sys.argv[1]))
