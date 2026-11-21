"""Head-to-head comparison of models on the SAME inner fold (e.g. the fold-0 review, while folds 1-4 are on hold).

    python -m pbts.analysis.fold_compare --models nnunet_resenc_l primus_v2b [segmamba ...] [--fold 0] [--workers 4]

The question it answers: are the models really different, or do they succeed and fail on the same scans, regions
and voxels? Built only from what the runs saved (per-case scores, per-lesion tables, label maps and probability
maps of the final checkpoint); nothing is retrained or re-predicted.

Writes <ROOT>/analysis/fold<k>_<models>/: results.json, CSV tables, and report.html (contains MRI slices: keep it
private). One fold is ~47 scans: numbers here are a first look, not the study result (that is all 5 folds).
"""
import argparse
import html
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats as sps

from pbts import config as C
from pbts.analysis import stats as S
from pbts.analysis.strata import STRATA, case_features

REGIONS = list(C.REGIONS)                       # ET NET CC ED TC WT
RARE = ["ET", "CC", "ED"]
NAMES = {"nnunet_resenc_l": "ResEnc", "primus_v2b": "Primus", "segmamba": "SegMamba", "swinunetr": "SwinUNETR"}
METRICS = [("off_global_dsc", "official Dice", True), ("lw2023_dice", "lesion-wise Dice (2023)", True),
           ("off_global_nsd", "official NSD", True), ("off_global_hd95", "official HD95 (mm)", False)]
FAIL = 0.5          # a scan 'fails' a region when its lesion-wise Dice is below this
CKPTS = (25, 50, 75, 100)


def nm(m):
    return NAMES.get(m, m)


def eval_dir(model, fold, ck=100, tag="default"):
    return C.RUNS_DIR / model / tag / f"fold_{fold}" / "eval" / f"ckpt{ck:03d}"


def _nii(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def dice(a, b):
    sa, sb = int(a.sum()), int(b.sum())
    return np.nan if sa + sb == 0 else 2.0 * int((a & b).sum()) / (sa + sb)


def subsets(models):
    """Ensembles to evaluate: every pair, plus all models when there are more than two."""
    out = [tuple(p) for p in itertools.combinations(models, 2)]
    return out + ([tuple(models)] if len(models) > 2 else [])


# ------------------------------------------------------------------ per-case voxel work (parallel)
def case_work(args):
    from pbts.eval.diagnostics import evaluate_case
    case, models, fold = args
    gt = _nii(C.RAW_DIR / "labelsTr" / f"{case}.nii.gz").astype(np.uint8)
    pred = {m: _nii(eval_dir(m, fold) / "predictions" / f"{case}.nii.gz").astype(np.uint8) for m in models}
    probs, box = {}, None
    for m in models:
        z = np.load(eval_dir(m, fold) / "probabilities" / f"{case}.npz")
        probs[m] = z["probs"].astype(np.float32)
        box = tuple(slice(int(a), int(b) + 1) for a, b in zip(z["box_lo"], z["box_hi"]))
    pairs = list(itertools.combinations(models, 2))

    vox = []
    for r in REGIONS:
        g = np.isin(gt, C.REGIONS[r])
        pm = {m: np.isin(pred[m], C.REGIONS[r]) for m in models}
        row = {"case": case, "region": r, "gt_vox": int(g.sum())}
        for m in models:
            row[f"dice|{m}"] = dice(pm[m], g)
            row[f"fn|{m}"], row[f"fp|{m}"] = int((g & ~pm[m]).sum()), int((pm[m] & ~g).sum())
        for a, b in pairs:
            k = f"{a}+{b}"
            row[f"agree|{k}"] = dice(pm[a], pm[b])
            row[f"fn_both|{k}"] = int((g & ~pm[a] & ~pm[b]).sum())
            row[f"fp_both|{k}"] = int((pm[a] & pm[b] & ~g).sum())
        vox.append(row)

    union = gt > 0
    for m in models:
        union |= pred[m] > 0
    wrong = {m: (pred[m] != gt) & union for m in models}
    lab = {"case": case, "union_vox": int(union.sum()), **{f"wrong|{m}": int(wrong[m].sum()) for m in models}}
    for a, b in pairs:
        both = wrong[a] & wrong[b]
        lab[f"wrong_both|{a}+{b}"] = int(both.sum())
        lab[f"wrong_same|{a}+{b}"] = int((both & (pred[a] == pred[b])).sum())

    ens = []
    for sub in subsets(models):
        lab_ens = np.zeros_like(gt)
        lab_ens[box] = sum(probs[m] for m in sub).argmax(0).astype(np.uint8)
        rows, _, _ = evaluate_case(case, lab_ens, gt)
        for rr in rows:
            rr["dice_vox"] = dice(np.isin(lab_ens, C.REGIONS[rr["region"]]), np.isin(gt, C.REGIONS[rr["region"]]))
            rr["ensemble"] = "+".join(sub)
            ens.append({k: rr[k] for k in ("case", "region", "ensemble", "lw2023_dice", "dice_vox", "gt_present",
                                           "pred_present")})
    # sanity: argmax of the saved probabilities reproduces the saved label map (float16 ties aside)
    m0 = models[0]
    single = np.zeros_like(gt)
    single[box] = probs[m0].argmax(0).astype(np.uint8)
    lab["sanity_vox_differ"] = int((single != pred[m0]).sum())
    return vox, lab, ens


# ------------------------------------------------------------------ analyses
def load_scores(models, fold, ck=100):
    out = []
    for m in models:
        d = pd.read_csv(eval_dir(m, fold, ck) / "per_case_region.csv")
        d["model"] = m
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    for c in ("gt_present", "pred_present"):
        d[c] = d[c].astype(bool)
    return d


def wide(d, metric, models):
    return d.pivot_table(index=["case", "region"], columns="model", values=metric, aggfunc="first")[models]


def score_table(d, models):
    rows = []
    for metric, label, hib in METRICS:
        w = wide(d, metric, models)
        for r in REGIONS:
            x = w.xs(r, level="region")
            row = {"metric": label, "region": r}
            for m in models:
                v = x[m].dropna()
                mean, lo, hi = S.bootstrap_ci(v.values, v.index) if len(v) >= 3 else (v.mean(), np.nan, np.nan)
                row[nm(m)] = mean
                row[f"{nm(m)} CI"] = f"[{lo:.3f}, {hi:.3f}]" if lo == lo else ""
            rows.append(row)
    return pd.DataFrame(rows)


def paired_table(d, models):
    rows = []
    for a, b in itertools.combinations(models, 2):
        for metric, label, hib in METRICS:
            w = wide(d, metric, models)
            res = []
            for r in REGIONS:
                x = w.xs(r, level="region")
                pc = S.paired_compare(x[a], x[b], x.index)
                res.append({"pair": f"{nm(a)} − {nm(b)}", "metric": label, "region": r, "n": pc["n"],
                            "mean diff": pc["mean_diff"], "95% CI": f"[{pc['ci_lo']:.3f}, {pc['ci_hi']:.3f}]",
                            "p (Wilcoxon)": pc["p_wilcoxon"],
                            f"{nm(a)} better": pc["a_better"], f"{nm(b)} better": pc["b_better"],
                            "_lo": pc["ci_lo"], "_hi": pc["ci_hi"], "_hib": hib})
            adj = S.holm([x["p (Wilcoxon)"] for x in res])
            for x, p in zip(res, adj):
                x["p (Holm)"] = p
                sig = x["_lo"] > 0 or x["_hi"] < 0
                if not sig:
                    x["verdict"] = "no clear difference"
                else:
                    a_wins = (x["mean diff"] > 0) == hib
                    x["verdict"] = f"{nm(a) if a_wins else nm(b)} better" + ("" if p < 0.05 else " (CI only)")
            rows += res
    return pd.DataFrame(rows)


def same_hard_cases(d, models):
    """Spearman correlation of per-case scores, and how much of the score variance is explained by WHICH SCAN it is
    (shared difficulty) vs WHICH MODEL (model differences) vs the rest (scan-specific model differences)."""
    rho, var = [], []
    for metric in ("lw2023_dice", "off_global_dsc"):
        w = wide(d, metric, models)
        for r in REGIONS:
            x = w.xs(r, level="region").dropna()
            for a, b in itertools.combinations(models, 2):
                s = sps.spearmanr(x[a], x[b]) if len(x) > 3 and x[a].nunique() > 1 and x[b].nunique() > 1 else None
                rho.append({"metric": metric, "region": r, "pair": f"{nm(a)} vs {nm(b)}", "n": len(x),
                            "spearman": s.statistic if s is not None else np.nan})
            if len(x) > 3:
                v = x.to_numpy(float)
                g = v.mean()
                ss_case = v.shape[1] * ((v.mean(1) - g) ** 2).sum()
                ss_model = v.shape[0] * ((v.mean(0) - g) ** 2).sum()
                ss_tot = ((v - g) ** 2).sum()
                if ss_tot > 0:
                    var.append({"metric": metric, "region": r, "n": len(x), "scan": ss_case / ss_tot,
                                "model": ss_model / ss_tot, "scan × model": 1 - (ss_case + ss_model) / ss_tot})
    return pd.DataFrame(rho), pd.DataFrame(var)


def failure_overlap(d, models):
    w = wide(d, "lw2023_dice", models)
    rows = []
    for r in REGIONS:
        x = w.xs(r, level="region").dropna()
        f = x < FAIL
        for a, b in itertools.combinations(models, 2):
            n = len(x)
            both, oa, ob = int((f[a] & f[b]).sum()), int((f[a] & ~f[b]).sum()), int((~f[a] & f[b]).sum())
            exp = f[a].mean() * f[b].mean() * n
            table = [[both, oa], [ob, n - both - oa - ob]]
            p = sps.fisher_exact(table)[1] if (both + oa) and (both + ob) else np.nan
            rows.append({"region": r, "pair": f"{nm(a)} vs {nm(b)}", "scans": n, "both fail": both,
                         f"only {nm(a)} fails": oa, f"only {nm(b)} fails": ob, "neither": n - both - oa - ob,
                         "both fail if independent": exp, "Fisher p": p,
                         "share of failures shared": both / (both + oa + ob) if both + oa + ob else np.nan})
    return pd.DataFrame(rows)


def presence_tables(d, models, ens=None):
    rows = []
    for r in RARE:
        x = d[d.region == r]
        pres = x.pivot_table(index="case", columns="model", values="pred_present", aggfunc="first")[models].astype(bool)
        gtp = x.groupby("case").gt_present.first()
        if ens is not None:
            e = ens[(ens.region == r)].pivot_table(index="case", columns="ensemble", values="pred_present", aggfunc="first")
            for c in e.columns:
                pres[f"ens:{c}"] = e[c].astype(bool)
        for status, sub in (("present in ground truth", gtp[gtp].index), ("absent in ground truth", gtp[~gtp].index)):
            p = pres.loc[sub]
            row = {"region": r, "scans": status, "n": len(sub)}
            for m in pres.columns:
                label = ("ensemble " + "+".join(nm(x) for x in m[4:].split("+"))) if m.startswith("ens:") else nm(m)
                row[label] = int(p[m].sum())
            if len(models) == 2:
                a, b = models
                row["both"], row[f"only {nm(a)}"] = int((p[a] & p[b]).sum()), int((p[a] & ~p[b]).sum())
                row[f"only {nm(b)}"], row["neither"] = int((~p[a] & p[b]).sum()), int((~p[a] & ~p[b]).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def voxel_tables(vox, lab, models):
    pairs = list(itertools.combinations(models, 2))
    rows = []
    for r in REGIONS:
        v = vox[vox.region == r]
        row = {"region": r}
        for m in models:
            row[f"Dice {nm(m)}–truth"] = v[f"dice|{m}"].mean()
        for a, b in pairs:
            k = f"{a}+{b}"
            row[f"Dice {nm(a)}–{nm(b)}"] = v[f"agree|{k}"].mean()
            fn_a, fn_b, fn_ab = v[f"fn|{a}"].sum(), v[f"fn|{b}"].sum(), v[f"fn_both|{k}"].sum()
            fp_a, fp_b, fp_ab = v[f"fp|{a}"].sum(), v[f"fp|{b}"].sum(), v[f"fp_both|{k}"].sum()
            gt = v.gt_vox.sum()
            row[f"missed voxels shared ({nm(a)},{nm(b)})"] = fn_ab / max(1, fn_a + fn_b - fn_ab)
            row[f"false voxels shared ({nm(a)},{nm(b)})"] = fp_ab / max(1, fp_a + fp_b - fp_ab)
            row[f"P({nm(b)} misses | {nm(a)} misses)"] = fn_ab / max(1, fn_a)
            row[f"P({nm(b)} misses)"] = fn_b / max(1, gt)
        rows.append(row)
    lrows = []
    for a, b in pairs:
        k = f"{a}+{b}"
        wa, wb, wab, same = (lab[f"wrong|{a}"].sum(), lab[f"wrong|{b}"].sum(), lab[f"wrong_both|{k}"].sum(),
                             lab[f"wrong_same|{k}"].sum())
        lrows.append({"pair": f"{nm(a)} vs {nm(b)}", "tumour voxels considered": int(lab.union_vox.sum()),
                      f"wrong label: {nm(a)}": int(wa), f"wrong label: {nm(b)}": int(wb), "wrong in both": int(wab),
                      f"share of {nm(a)}'s errors also made by {nm(b)}": wab / max(1, wa),
                      f"share of {nm(b)}'s errors also made by {nm(a)}": wab / max(1, wb),
                      "both wrong with the SAME wrong label": same / max(1, wab)})
    return pd.DataFrame(rows), pd.DataFrame(lrows)


def error_types(vox, models, fold):
    """Missed voxels split into whole missed lesions vs edges of found lesions; false voxels split into false-alarm
    pieces (touching no true lesion) vs over-reach at the edges of true lesions."""
    rows = []
    for m in models:
        les = pd.read_csv(eval_dir(m, fold) / "per_lesion.csv")
        for r in REGIONS:
            v = vox[vox.region == r]
            l = les[les.region == r]
            fn, fp = v[f"fn|{m}"].sum(), v[f"fp|{m}"].sum()
            miss = l[(l.kind == "gt_lesion") & ~l.detected.astype(bool)]
            fa = l[l.kind == "pred_fp"]
            rows.append({"model": nm(m), "region": r, "true lesions": int((l.kind == "gt_lesion").sum()),
                         "lesions missed": len(miss), "false-alarm pieces": len(fa),
                         "missed voxels": int(fn), "… in wholly missed lesions": miss.vol.sum() / max(1, fn),
                         "false voxels": int(fp), "… in false-alarm pieces": fa.vol.sum() / max(1, fp),
                         "median size of missed lesions (voxels)": miss.vol.median() if len(miss) else np.nan,
                         "median size of false-alarm pieces (voxels)": fa.vol.median() if len(fa) else np.nan})
    return pd.DataFrame(rows)


def confusion(models, fold):
    out = {}
    for m in models:
        c = pd.read_csv(eval_dir(m, fold) / "confusion.csv", index_col=0)
        out[m] = c.div(c.sum(axis=1).replace(0, np.nan), axis=0)
    return out


def ensemble_table(d, ens, models):
    rows = []
    lw = wide(d, "lw2023_dice", models)
    for r in REGIONS:
        x = lw.xs(r, level="region")
        row = {"region": r}
        for m in models:
            row[nm(m)] = x[m].mean()
        e = ens[ens.region == r]
        for sub, g in e.groupby("ensemble"):
            row["ensemble " + "+".join(nm(s) for s in sub.split("+"))] = g.lw2023_dice.mean()
        row["oracle (best model per scan)"] = x.max(axis=1).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_table(d, models):
    f = case_features()
    lw = d[d.region.isin(["WT", "TC", "ET", "CC", "ED"])].pivot_table(index="case", columns=["region", "model"],
                                                                      values="lw2023_dice", aggfunc="first")
    rows = []
    for s in STRATA:
        if s not in f.columns:
            continue
        lv = f.reindex(lw.index)[s].fillna("unknown")
        for level, idx in lw.groupby(lv).groups.items():
            if len(idx) < 4:
                continue
            row = {"subgroup": s, "level": level, "n": len(idx)}
            for r in ("WT", "TC", "ET", "CC", "ED"):
                vals = [lw.loc[idx, (r, m)].mean() for m in models]
                row[r] = " / ".join(f"{v:.2f}" for v in vals)
                if len(models) == 2:
                    row[f"Δ{r}"] = vals[0] - vals[1]
            rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_table(models, fold):
    rows = []
    for m in models:
        for ck in CKPTS:
            p = eval_dir(m, fold, ck) / "per_case_region.csv"
            if not p.exists():
                continue
            x = pd.read_csv(p)
            for r in REGIONS:
                y = x[x.region == r]
                rows.append({"model": nm(m), "checkpoint %": ck, "region": r,
                             "official Dice": y.off_global_dsc.mean(), "lesion-wise Dice": y.lw2023_dice.mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_scatter(d, a, b):
    plt = _plt()
    from pbts.report.run_report import _fig_png
    f = case_features()
    w = wide(d, "lw2023_dice", [a, b])
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.6))
    cols = {"DIPG registry": "#2f5d8a", "HGG (DFCI/BCH/BWH)": "#b4442a", "CBTN (mixed types)": "#2e7d4f"}
    for ax, r in zip(axs.flat, ["WT", "TC", "NET", "ET", "CC", "ED"]):
        x = w.xs(r, level="region").dropna()
        col = [cols.get(f.collection.get(c, ""), "#999999") for c in x.index]
        jit = np.random.default_rng(0).normal(0, 0.008, (2, len(x)))
        ax.scatter(x[a] + jit[0], x[b] + jit[1], s=18, c=col, alpha=.8, edgecolors="none")
        ax.plot([0, 1], [0, 1], color="#888", lw=.8, ls="--")
        rho = sps.spearmanr(x[a], x[b]).statistic if x[a].nunique() > 1 and x[b].nunique() > 1 else np.nan
        ax.set_title(f"{r} · Spearman {rho:.2f}", fontsize=9)
        ax.set_xlim(-.05, 1.05); ax.set_ylim(-.05, 1.05)
        ax.set_xlabel(nm(a), fontsize=8); ax.set_ylabel(nm(b), fontsize=8); ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in cols.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Lesion-wise Dice per scan (one dot = one scan; on the dashed line = both models scored the same)", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    return _fig_png(fig)


def fig_variance(var):
    plt = _plt()
    from pbts.report.run_report import _fig_png
    v = var[var.metric == "lw2023_dice"].set_index("region").reindex(REGIONS)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    left = np.zeros(len(v))
    for c, col in (("scan", "#2f5d8a"), ("model", "#b4442a"), ("scan × model", "#c9c6bd")):
        ax.barh(v.index, v[c] * 100, left=left, color=col, label={"scan": "which scan (shared difficulty)",
                "model": "which model", "scan × model": "model differences specific to a scan"}[c])
        left += (v[c] * 100).fillna(0).to_numpy()
    ax.set_xlabel("% of the variance of per-scan lesion-wise Dice", fontsize=8); ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False); ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_png(fig)


def fig_confusion(conf, models):
    plt = _plt()
    from pbts.report.run_report import _fig_png
    labs = ["background", "ET", "NET", "CC", "ED"]
    fig, axs = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.4))
    for ax, m in zip(np.atleast_1d(axs), models):
        c = conf[m].to_numpy(float)
        ax.imshow(c, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                if i > 0 and c[i, j] == c[i, j]:
                    ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if c[i, j] > .5 else "black")
        ax.set_xticks(range(5), labs, fontsize=7, rotation=30); ax.set_yticks(range(5), labs, fontsize=7)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("ground truth", fontsize=8); ax.set_title(nm(m), fontsize=9)
    fig.suptitle("Where each true label's voxels end up (rows sum to 1, inside the tumour of truth or prediction)", fontsize=8)
    fig.tight_layout()
    return _fig_png(fig)


def case_figure(case, models, fold, region, mode, title):
    """Axial slice through the region: ground truth and every model, on T1CE (FLAIR for oedema)."""
    plt = _plt()
    from pbts.report.run_report import LABEL_COLORS, _fig_png
    gt = _nii(C.RAW_DIR / "labelsTr" / f"{case}.nii.gz")
    preds = [_nii(eval_dir(m, fold) / "predictions" / f"{case}.nii.gz") for m in models]
    ch = 0 if region == "ED" else 2
    img = nib.load(str(C.RAW_DIR / "imagesTr" / f"{case}_{ch:04d}.nii.gz")).get_fdata(dtype=np.float32)
    labs = C.REGIONS[region]
    g = np.isin(gt, labs)
    ps = [np.isin(p, labs) for p in preds]
    if mode == "disagree" and len(ps) >= 2:
        t = np.logical_xor(ps[0], ps[1])
    else:
        t = g if g.any() else np.any(ps, axis=0)
    z = int(np.argmax(t.sum((0, 1)))) if t.any() else gt.shape[2] // 2
    lo, hi = np.percentile(img[img > 0], [1, 99.5]) if (img > 0).any() else (0, 1)
    base = np.clip((img[:, :, z] - lo) / (hi - lo + 1e-6), 0, 1)
    fig, axs = plt.subplots(1, 1 + len(models), figsize=(2.6 * (1 + len(models)), 2.9))
    for ax, lab, name in zip(axs, [gt] + preds, ["ground truth"] + [nm(m) for m in models]):
        rgb = np.stack([base] * 3, -1)
        for k, col in LABEL_COLORS.items():
            mm = lab[:, :, z] == k
            rgb[mm] = .45 * rgb[mm] + .55 * np.array(col)
        ax.imshow(np.rot90(rgb)); ax.set_axis_off(); ax.set_title(name, fontsize=8)
    fig.suptitle(title, fontsize=8.5)
    fig.tight_layout()
    return _fig_png(fig)


# ------------------------------------------------------------------ report
def _table(df, pct=(), floatfmt="{:.3f}"):
    df = df[[c for c in df.columns if not str(c).startswith("_")]]
    fmt = {c: (lambda v: "—" if v != v else f"{100 * v:.0f}%") for c in df.columns if c in pct or str(c).startswith(
        ("share", "P(", "missed voxels shared", "false voxels shared", "… in", "both wrong with", "scan", "model"))
        and df[c].dtype.kind == "f"}
    for c in df.columns:
        if c not in fmt and df[c].dtype.kind == "f":
            fmt[c] = (lambda v, f=floatfmt: "—" if v != v else f.format(v))
    return df.to_html(index=False, escape=True, formatters=fmt, border=0, classes="t")


def build_report(out, models, fold, parts, figs, cases_html, key):
    from pbts.report.run_report import CSS, LEGEND
    title = f"Fold {fold}: " + " vs ".join(nm(m) for m in models)
    sec = []
    def s(h, intro, body):
        sec.append(f"<h2>{html.escape(h)}</h2><p class='intro'>{intro}</p>{body}")
    s("Key numbers", "Computed automatically from the sections below.", "<ul>" + "".join(f"<li>{k}</li>" for k in key) + "</ul>")
    s("1. Scores", "Mean per scan over the fold's validation scans, with patient-bootstrap 95% CI. Official Dice/NSD: a "
      "region absent in both truth and prediction is left out (NaN); predicting an absent region scores 0. Lesion-wise "
      "Dice (BraTS 2023): absent in both = 1, every missed or false lesion counts as 0.", _table(parts["scores"]))
    s("2. Paired differences", "Same scans, model A minus model B. 'Better' only when the 95% CI excludes 0; Holm-adjusted "
      "Wilcoxon p across the 6 regions. For HD95 lower is better. Official Dice compares only scans where both models "
      "have a value, so a false alarm against a correct 'nothing' drops out; lesion-wise Dice keeps those scans.",
      _table(parts["paired"], pct=tuple(c for c in parts["paired"].columns if str(c).endswith(" better"))))
    s("3. Do the same scans trip both models?", "Spearman correlation of per-scan scores, and how much of the spread of "
      "per-scan scores comes from which scan it is (shared difficulty), from which model, and from model differences that "
      "depend on the scan.", f"<figure><img src='{figs['variance']}'></figure>" + _table(parts["rho"]) + _table(parts["var"]))
    s("Per-scan scores", "", f"<figure><img src='{figs['scatter']}'></figure>")
    s(f"4. Same failures? (lesion-wise Dice < {FAIL})", "Scans failed by both, by one, by neither, against how many both "
      "would fail if failures were unrelated (independent).", _table(parts["fail"]))
    s("5. Rare regions: found and falsely predicted", "Counts of scans in which each model predicted the region at all.",
      _table(parts["presence"]))
    s("6. Same voxels?", "Agreement between the two models' predictions vs each model's agreement with the truth (mean "
      "per-scan Dice), and how much of the missed / falsely predicted tumour is shared.", _table(parts["voxel"]) +
      _table(parts["labels"]))
    s("7. Error types", "Missed tumour from lesions missed entirely vs from edges of lesions that were found; false tumour "
      "from false-alarm pieces vs over-reach around true lesions.", _table(parts["errtypes"]))
    s("8. Label confusion", "", f"<figure><img src='{figs['confusion']}'></figure>")
    s("9. Complementarity", "Lesion-wise Dice of each model, of the probability-averaged ensemble, and of an oracle that "
      "picks the better model for each scan (an upper bound on what combining them could give).", _table(parts["ensemble"]))
    s("10. Subgroups", "Lesion-wise Dice per model (in the order of the title) for groups of at least 4 scans; Δ = first "
      "minus second model. Small groups: read as hints only.", _table(parts["subgroups"], floatfmt="{:+.2f}"))
    s("11. Training length", "Mean scores of the 25/50/75/100% checkpoints on the same scans.",
      _table(parts["ckpt"].pivot_table(index=["region", "checkpoint %"], columns="model", values="lesion-wise Dice")
             .reset_index()))
    s("12. Cases", "Scans both models fail, and scans where they disagree most. T1CE (FLAIR for oedema), axial slice "
      "through the region.", cases_html + LEGEND)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}"
            ".intro{color:var(--muted,#666);max-width:900px} table.t{border-collapse:collapse;font-size:12.5px;margin:8px 0 18px}"
            "table.t td,table.t th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:right}"
            "table.t td:first-child,table.t th:first-child{text-align:left} figure img{max-width:100%}</style></head><body><main>"
            f"<h1>{html.escape(title)}</h1><p class='intro'>Generated {time.strftime('%Y-%m-%d %H:%M')} by "
            f"pbts.analysis.fold_compare from the saved predictions (final checkpoint, 8-flip TTA, no post-processing). "
            f"One fold only: a first look, not the study result. Contains MRI slices: keep private.</p>"
            + "".join(sec) + "</main></body></html>")
    (out / "report.html").write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    models, fold = a.models, a.fold
    out = Path(a.out) if a.out else C.ROOT / "analysis" / f"fold{fold}_{'_vs_'.join(nm(m) for m in models)}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    d = load_scores(models, fold)
    cases = sorted(set.intersection(*[set(d[d.model == m].case) for m in models]))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(case_work, [(c, models, fold) for c in cases]))
    vox = pd.DataFrame([r for v, _, _ in res for r in v])
    lab = pd.DataFrame([l for _, l, _ in res])
    ens = pd.DataFrame([r for _, _, e in res for r in e])

    # our whole-volume Dice must equal the official one (same predictions, same definition)
    chk = vox.melt(id_vars=["case", "region"], value_vars=[f"dice|{m}" for m in models], var_name="m", value_name="ours")
    chk["model"] = chk.m.str.split("|").str[1]
    chk = chk.merge(d[["case", "region", "model", "off_global_dsc"]], on=["case", "region", "model"])
    dice_gap = float((chk.ours - chk.off_global_dsc).abs().max())

    parts = {"scores": score_table(d, models), "paired": paired_table(d, models)}
    parts["rho"], parts["var"] = same_hard_cases(d, models)
    parts["fail"] = failure_overlap(d, models)
    parts["presence"] = presence_tables(d, models, ens)
    parts["voxel"], parts["labels"] = voxel_tables(vox, lab, models)
    parts["errtypes"] = error_types(vox, models, fold)
    parts["ensemble"] = ensemble_table(d, ens, models)
    parts["subgroups"] = subgroup_table(d, models)
    parts["ckpt"] = checkpoint_table(models, fold)
    conf = confusion(models, fold)
    for k, v in parts.items():
        v.to_csv(out / f"{k}.csv", index=False)
    vox.to_csv(out / "voxel_per_case.csv", index=False)
    ens.to_csv(out / "ensemble_per_case.csv", index=False)

    figs = {"scatter": fig_scatter(d, models[0], models[1]), "variance": fig_variance(parts["var"]),
            "confusion": fig_confusion(conf, models)}

    # cases: shared failures (worst mean lesion-wise Dice among scans both fail) and largest disagreements
    lw = wide(d, "lw2023_dice", models).reset_index()
    a0, b0 = models[0], models[1]
    lw["both_fail"] = (lw[a0] < FAIL) & (lw[b0] < FAIL)
    lw["gap"] = (lw[a0] - lw[b0]).abs()
    info = d[d.model == a0].set_index(["case", "region"])
    pp = d.pivot_table(index=["case", "region"], columns="model", values="pred_present", aggfunc="first")
    lw["truth_has_it"] = [bool(info.loc[(c, r), "gt_present"]) for c, r in zip(lw.case, lw.region)]

    def what(c, r):
        if not info.loc[(c, r), "gt_present"]:
            return "both predict it, but the ground truth has none"
        return "both miss it entirely" if not pp.loc[(c, r)].astype(bool).any() else "both score below 0.5"
    picks = []
    for r, present in (("CC", True), ("CC", False), ("ED", True), ("ED", False), ("ET", None), ("WT", None)):
        x = lw[(lw.region == r) & lw.both_fail & ((lw["truth_has_it"] == present) if present is not None else True)]
        x = x.assign(m=lambda t: t[[a0, b0]].mean(axis=1)).sort_values("m")
        picks += [(c, r, "shared", f"{c} · {r}: {what(c, r)} (lesion-wise {x.loc[i, a0]:.2f} / {x.loc[i, b0]:.2f})")
                  for i, c in zip(x.index[:1], x.case[:1])]
    for r in ("ET", "CC", "ED", "NET"):
        x = lw[lw.region == r].sort_values("gap", ascending=False)
        picks += [(c, r, "disagree", f"{c} · {r}: largest disagreement ({nm(a0)} {x.loc[i, a0]:.2f} vs {nm(b0)} "
                   f"{x.loc[i, b0]:.2f})") for i, c in zip(x.index[:1], x.case[:1])]
    cases_html = "".join(f"<figure><img src='{case_figure(c, models, fold, r, mode, t)}'></figure>" for c, r, mode, t in picks)

    # key numbers
    pr = parts["paired"]
    sig = pr[pr.verdict != "no clear difference"]
    pair = f"{nm(a0)} vs {nm(b0)}"
    var = parts["var"][parts["var"].metric == "lw2023_dice"].set_index("region")
    rho = parts["rho"][(parts["rho"].metric == "lw2023_dice") & (parts["rho"].pair == pair)].set_index("region")
    fo = parts["fail"][parts["fail"].pair == pair].set_index("region")
    en = parts["ensemble"].set_index("region")
    best_single = en[[nm(m) for m in models]].max(axis=1)
    ens_col = [c for c in en.columns if c.startswith("ensemble")][-1]
    key = [f"{len(sig)} of {len(pr)} paired region × metric comparisons show a clear difference: " +
           ("; ".join(f"{r.region} {r.metric}: {r.verdict} ({r['mean diff']:+.3f})" for _, r in sig.iterrows()) or "none"),
           "Spearman correlation of per-scan lesion-wise Dice: " + ", ".join(f"{r} {rho.loc[r, 'spearman']:.2f}"
                                                                           for r in REGIONS if r in rho.index),
           "Share of per-scan score variance explained by WHICH SCAN (shared difficulty) vs WHICH MODEL: " +
           ", ".join(f"{r} {100 * var.loc[r, 'scan']:.0f}% vs {100 * var.loc[r, 'model']:.0f}%" for r in REGIONS if r in var.index),
           "Scans failed by both (vs if independent): " + ", ".join(
               f"{r} {int(fo.loc[r, 'both fail'])} ({fo.loc[r, 'both fail if independent']:.1f})" for r in REGIONS if r in fo.index),
           f"{ens_col} vs best single model, lesion-wise Dice: " + ", ".join(
               f"{r} {en.loc[r, ens_col] - best_single[r]:+.3f}" for r in REGIONS),
           "Oracle (best model per scan) vs best single model: " + ", ".join(
               f"{r} {en.loc[r, 'oracle (best model per scan)'] - best_single[r]:+.3f}" for r in REGIONS),
           f"Checks: our whole-volume Dice vs the official scorer, max difference {dice_gap:.2e}; argmax of saved "
           f"probabilities vs saved labels, {int(lab.sanity_vox_differ.sum())} voxels differ over {len(cases)} scans."]
    build_report(out, models, fold, parts, figs, cases_html, key)
    json.dump({"models": models, "fold": fold, "cases": len(cases), "key": key, "dice_gap_vs_official": dice_gap,
               "minutes": (time.time() - t0) / 60}, open(out / "results.json", "w"), indent=1)
    print("\n".join(key))
    print(f"report: {out / 'report.html'} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
