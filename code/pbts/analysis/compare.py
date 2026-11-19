"""Comparison dashboard across models (out-of-fold results on the 234 development cases).

    python -m pbts.analysis.compare --models nnunet_resenc_l primus_v2b segmamba [swinunetr] \
        [--tag segmamba=rescue_lr3] [--ensemble NAME ...] [--out reports/compare]

Writes index.html (self-contained) and the underlying tables as CSV. Only models whose 5 folds are complete
enter the statistics (partial models are listed but excluded), so every comparison is on identical cases.
Raw predictions only; no post-processing.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pbts import config as C
from pbts.analysis.collect import collect_model
from pbts.analysis.stats import bootstrap_ci, holm, paired_compare
from pbts.analysis.strata import STRATA, case_features
from pbts.report.run_report import CSS, LABEL_COLORS, _fig_png, _fmt, _img

REGIONS = list(C.REGIONS)
METRICS = {"lw2023_dice": "2023 lesion-wise Dice", "off_global_dsc": "official whole-volume Dice",
           "off_global_nsd": "official NSD", "off_all_instance_f1": "official detection F1"}


def composite(pcr):
    """Per-case score = mean over the 6 regions of the 2023 lesion-wise Dice (absent-in-both counts 1)."""
    return pcr.groupby(["model", "case"]).lw2023_dice.mean().rename("composite").reset_index()


def scoreboard(pcr):
    rows = []
    for (m, r), d in pcr.groupby(["model", "region"]):
        pres, absn = d[d.gt_present], d[~d.gt_present]
        row = {"model": m, "region": r}
        for k, name in METRICS.items():
            row[name] = bootstrap_ci(d[k].values, d.case.values)
        row["Dice when present"] = bootstrap_ci(pres.off_global_dsc.values, pres.case.values)
        row["false-alarm rate (absent)"] = (absn.presence == "FP").mean() if len(absn) else np.nan
        row["miss rate (present)"] = (pres.presence == "FN").mean() if len(pres) else np.nan
        row["n present/absent"] = f"{len(pres)}/{len(absn)}"
        rows.append(row)
    return pd.DataFrame(rows)


def scoreboard_html(sb):
    show = sb.copy()
    for c in list(METRICS.values()) + ["Dice when present"]:
        show[c] = show[c].map(lambda t: _fmt(*t))
    for c in ["false-alarm rate (absent)", "miss rate (present)"]:
        show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    return "".join(f"<h4>{r}</h4>" + show[show.region == r].drop(columns="region").to_html(index=False, escape=False, classes="t")
                   for r in REGIONS)


def pairwise(pcr, models):
    rows = []
    for k in METRICS:
        for r in REGIONS:
            piv = pcr[pcr.region == r].pivot(index="case", columns="model", values=k)
            for a, b in itertools.combinations(models, 2):
                res = paired_compare(piv[a].values, piv[b].values, piv.index)
                rows.append({"metric": METRICS[k], "region": r, "A": a, "B": b, **res})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for k in t.metric.unique():                      # one family per metric: regions x pairs
        m = t.metric == k
        t.loc[m, "p_holm"] = holm(t.loc[m, "p_wilcoxon"].values)
    return t


def pairwise_html(t):
    s = t.copy()
    s["A − B"] = [_fmt(m, lo, hi) for m, lo, hi in zip(s.mean_diff, s.ci_lo, s.ci_hi)]
    s["p (Holm)"] = s.p_holm.map(lambda p: "—" if pd.isna(p) else (f"<b>{p:.3g}</b>" if p < 0.05 else f"{p:.3g}"))
    s["cases A better / B better"] = [f"{a:.0%} / {b:.0%}" if not pd.isna(a) else "—" for a, b in zip(s.a_better, s.b_better)]
    keep = ["region", "A", "B", "n", "A − B", "p (Holm)", "cases A better / B better"]
    return "".join(f"<h4>{m}</h4>" + s[s.metric == m][keep].to_html(index=False, escape=False, classes="t")
                   for m in s.metric.unique()) + \
        "<p class='note'>Paired over identical cases; CI from resampling patients; Wilcoxon signed-rank; " \
        "Holm correction within each metric across all regions and model pairs. Bold: p &lt; 0.05 after correction.</p>"


def strata_tables(pcr, feats, models):
    comp = composite(pcr).merge(feats, left_on="case", right_index=True)
    out = []
    for s in STRATA:
        g = comp.groupby([s, "model"]).composite.mean().unstack("model").reindex(columns=models)
        n = comp[comp.model == models[0]].groupby(s).size().rename("n cases")
        g = g.join(n)
        g.index = [f"{s}: {v}" for v in g.index]
        out.append(g)
    return pd.concat(out)


def strata_heatmap(tab, models):
    import matplotlib.pyplot as plt
    vals = tab[models].values.astype(float)
    fig, ax = plt.subplots(figsize=(2.2 + 1.6 * len(models), 0.32 * len(tab) + 1.2))
    im = ax.imshow(vals, aspect="auto", cmap="viridis", vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(tab))); ax.set_yticklabels([f"{i}  (n={int(n)})" for i, n in zip(tab.index, tab["n cases"])], fontsize=7)
    for i in range(vals.shape[0]):
        if np.all(np.isnan(vals[i])):
            continue
        best = np.nanargmax(vals[i])
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.3f}{' ★' if j == best else ''}", ha="center", va="center", fontsize=7,
                    color="white" if vals[i, j] < np.nanmean(vals) else "black")
    fig.colorbar(im, ax=ax, fraction=.03)
    ax.set_title("mean per-case score (6-region 2023 lesion-wise Dice) by subgroup · ★ best in row", fontsize=9)
    return _img(_fig_png(fig))


def detection_by_size(les, models):
    import matplotlib.pyplot as plt
    g = les[les.kind == "gt_lesion"].copy()
    bins, names = [0, 50, 200, 1000, 5000, 1e9], ["≤50", "51–200", "201–1k", "1k–5k", ">5k"]
    g["size"] = pd.cut(g.vol, bins, labels=names)
    fig, ax = plt.subplots(1, len(REGIONS), figsize=(3.2 * len(REGIONS), 3.2), sharey=True)
    for a, r in zip(ax, REGIONS):
        for m in models:
            d = g[(g.region == r) & (g.model == m)].groupby("size", observed=False).detected.mean()
            a.plot(names, d.values, marker="o", label=m)
        a.set_title(r); a.set_ylim(0, 1); a.tick_params(axis="x", labelsize=7, rotation=30); a.grid(alpha=.3)
    ax[0].set_ylabel("fraction of GT lesions detected"); ax[-1].legend(fontsize=7)
    fig.tight_layout()
    return _img(_fig_png(fig))


def false_alarms(pcr, les, models):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        d = pcr[pcr.model == m]
        fa = [((d.region == r) & ~d.gt_present & (d.presence == "FP")).sum() / max(1, ((d.region == r) & ~d.gt_present).sum())
              for r in ["ET", "CC", "ED"]]
        ax[0].bar(np.arange(3) + i * w, fa, w, label=m)
        n_fp = [les[(les.model == m) & (les.kind == "pred_fp") & (les.region == r)].shape[0] for r in REGIONS]
        ax[1].bar(np.arange(len(REGIONS)) + i * w, n_fp, w, label=m)
    ax[0].set_xticks(np.arange(3) + 0.4 - w / 2); ax[0].set_xticklabels(["ET", "CC", "ED"])
    ax[0].set_title("scans where an ABSENT region was predicted (fraction)"); ax[0].set_ylim(0, 1)
    ax[1].set_xticks(np.arange(len(REGIONS)) + 0.4 - w / 2); ax[1].set_xticklabels(REGIONS)
    ax[1].set_title("number of false-positive pieces (all scans)")
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=.3, axis="y")
    fig.tight_layout()
    return _img(_fig_png(fig))


def confusions(confs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(confs), figsize=(4.2 * len(confs), 3.8))
    ax = np.atleast_1d(ax)
    for a, (m, c) in zip(ax, confs.items()):
        v = c.values.astype(float); v = v / np.maximum(v.sum(1, keepdims=True), 1)
        a.imshow(v, vmin=0, vmax=1, cmap="Blues")
        names = [x.replace("pred_", "") for x in c.columns]
        a.set_xticks(range(5)); a.set_xticklabels(names, fontsize=7); a.set_yticks(range(5)); a.set_yticklabels(names, fontsize=7)
        for i in range(5):
            for j in range(5):
                a.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if v[i, j] > .5 else "black")
        a.set_title(m, fontsize=9); a.set_xlabel("predicted", fontsize=8)
    ax[0].set_ylabel("ground truth", fontsize=8)
    fig.tight_layout()
    return _img(_fig_png(fig))


def win_matrix(pcr, models):
    comp = composite(pcr).pivot(index="case", columns="model", values="composite")[models]
    t = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            t.loc[a, b] = np.nan if a == b else float((comp[a] > comp[b]).mean())
    oracle = comp.max(1)
    best_single = comp.mean().idxmax()
    head = pd.DataFrame({"mean per-case score": comp.mean(), "best on # cases": comp.idxmax(1).value_counts().reindex(models).fillna(0).astype(int)})
    note = (f"<p>Oracle (best model chosen per case): <b>{oracle.mean():.3f}</b> vs best single model "
            f"<b>{best_single}</b> {comp[best_single].mean():.3f}. The gap is the most any per-case selection or "
            f"ensemble of these models could gain.</p>")
    return head.to_html(classes="t", float_format="%.3f") + "<h4>fraction of cases where row beats column</h4>" + \
        t.to_html(classes="t", float_format="%.2f", na_rep="—") + note


def cost_table(infos):
    rows = []
    for m, inf in infos.items():
        c = inf.get("cost", {})
        rows.append({"model": m, "params (M)": np.mean(c.get("params_M", [np.nan])),
                     "train GPU-h (5 folds)": np.sum(c.get("train_gpu_h", [np.nan])),
                     "peak GPU GB": np.max(c.get("peak_GB", [np.nan])),
                     "predict s/case (8-flip)": np.mean(c.get("predict_s_per_case", [np.nan]))})
    return pd.DataFrame(rows).to_html(index=False, classes="t", float_format="%.2f")


def disagreement_gallery(pcr, infos, models, n=6):
    import matplotlib.pyplot as plt
    import nibabel as nib
    comp = composite(pcr).pivot(index="case", columns="model", values="composite")[models]
    spread = (comp.max(1) - comp.min(1)).sort_values(ascending=False)
    worst_all = comp.max(1).sort_values()
    picks = list(dict.fromkeys(list(spread.index[: n // 2]) + list(worst_all.index[: n - n // 2])))
    fold_of = pcr.drop_duplicates("case").set_index("case").fold
    out = []
    for c in picks:
        gt = np.asanyarray(nib.load(str(C.RAW_DIR / "labelsTr" / f"{c}.nii.gz")).dataobj)
        fl = nib.load(str(C.RAW_DIR / "imagesTr" / f"{c}_0000.nii.gz")).get_fdata(dtype=np.float32)
        preds = {m: np.asanyarray(nib.load(str(Path(infos[m]["pred_dirs"][int(fold_of[c])]) / f"{c}.nii.gz")).dataobj)
                 for m in models}
        t = gt > 0
        for p in preds.values():
            t |= p > 0
        z = int(np.round(np.argwhere(t)[:, 2].mean())) if t.any() else gt.shape[2] // 2
        lo, hi = np.percentile(fl[fl > 0], [1, 99.5]) if (fl > 0).any() else (0, 1)
        fig, ax = plt.subplots(1, 1 + len(models), figsize=(2.6 * (1 + len(models)), 2.9))
        for a, (name, lab) in zip(ax, [("ground truth", gt)] + [(m, preds[m]) for m in models]):
            img = np.clip((fl[:, :, z] - lo) / (hi - lo + 1e-6), 0, 1)
            rgb = np.stack([img] * 3, -1)
            for k, col in LABEL_COLORS.items():
                mm = lab[:, :, z] == k
                rgb[mm] = .45 * rgb[mm] + .55 * np.array(col)
            a.imshow(np.rot90(rgb)); a.set_axis_off()
            score = "" if name == "ground truth" else f" · {comp.at[c, name]:.2f}"
            a.set_title(name + score, fontsize=8)
        fig.suptitle(c, fontsize=9); fig.tight_layout()
        out.append(f"<figure>{_img(_fig_png(fig))}</figure>")
    return '<div class="gallery">' + "".join(out) + "</div><p class='note'>First half: cases where models disagree " \
                                                     "most. Second half: cases where even the best model scores lowest. " \
                                                     "ET blue · NET orange · CC green · ED yellow.</p>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tag", nargs="*", default=[], help="model=tag overrides (default tag 'default')")
    ap.add_argument("--out", default=str(C.REPORTS_DIR / "compare"))
    ap.add_argument("--allow_partial", action="store_true", help="include models with missing folds (smoke tests only)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    tags = dict(t.split("=") for t in a.tag)
    pcrs, less, confs, infos, excluded = [], [], {}, {}, []
    for m in a.models:
        p, l, c, inf = collect_model(m, tags.get(m, "default"))
        if p is None or (inf["folds_missing"] and not a.allow_partial):
            excluded.append(f"{m} (folds missing: {inf['folds_missing']})")
            continue
        pcrs.append(p); less.append(l); confs[m] = c; infos[m] = inf
    models = list(infos)
    if not models:
        raise SystemExit(f"no model has complete results: {excluded}")
    pcr, les = pd.concat(pcrs, ignore_index=True), pd.concat(less, ignore_index=True)
    common = set.intersection(*[set(pcr[pcr.model == m].case) for m in models])
    pcr, les = pcr[pcr.case.isin(common)], les[les.case.isin(common)]
    feats = case_features()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sb = scoreboard(pcr)
    sb.assign(**{k: sb[k].map(lambda t: t[0]) for k in list(METRICS.values()) + ["Dice when present"]}).to_csv(out / "scoreboard.csv", index=False)
    pw = pairwise(pcr, models) if len(models) > 1 else pd.DataFrame()
    pw.to_csv(out / "pairwise.csv", index=False)
    st = strata_tables(pcr, feats, models); st.to_csv(out / "strata.csv")
    composite(pcr).to_csv(out / "per_case_composite.csv", index=False)

    parts = [f"<h1>Model comparison</h1><p class='meta'>{len(common)} development cases (out-of-fold) · models: "
             f"{', '.join(models)}{' · excluded: ' + '; '.join(excluded) if excluded else ''} · raw predictions, "
             f"no post-processing</p>",
             "<h2>Cost</h2>", cost_table(infos),
             "<h2>Scoreboard</h2>", scoreboard_html(sb)]
    if len(models) > 1:
        parts += ["<h2>Paired differences</h2>", pairwise_html(pw), "<h2>Case-level wins and oracle headroom</h2>",
                  win_matrix(pcr, models)]
    parts += ["<h2>Who wins where (subgroups)</h2>", strata_heatmap(st, models),
              "<h2>Detection by lesion size</h2>", detection_by_size(les, models),
              "<h2>False alarms</h2>", false_alarms(pcr, les, models),
              "<h2>Label confusion</h2>", confusions(confs),
              "<h2>Cases to look at</h2>", disagreement_gallery(pcr, infos, models)]
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Model comparison</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(parts)}</main></body></html>")
    (out / "index.html").write_text(html)
    json.dump({"models": models, "excluded": excluded, "n_cases": len(common)}, open(out / "compare_info.json", "w"), indent=1)
    print(out / "index.html")


if __name__ == "__main__":
    main()
