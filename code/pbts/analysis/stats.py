"""Statistics for comparing models on the same cases.

- bootstrap_ci: mean with 95% CI, resampling PATIENTS (a child's repeat scans move together)
- paired_compare: per-case paired difference A-B with patient-bootstrap CI and Wilcoxon signed-rank p
- holm: Holm-Bonferroni adjustment over a family of tests
NaNs (e.g. the official scorer's 'absent in both') are dropped pairwise, never imputed.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps


def patient(case: str) -> str:
    return case.rsplit("-", 1)[0]


def _groups(series: pd.Series):
    return series.groupby(series.index.map(patient)).apply(lambda s: s.to_numpy())


def bootstrap_ci(values, cases, n=2000, seed=0):
    v = pd.Series(np.asarray(values, float), index=list(cases)).dropna()
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    g = _groups(v)
    arrs = list(g.values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arrs), size=(n, len(arrs)))
    sums = np.array([a.sum() for a in arrs]); counts = np.array([len(a) for a in arrs])
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_compare(a, b, cases, n=2000, seed=0):
    """a, b: per-case values of two models on the SAME cases (aligned)."""
    d = pd.DataFrame({"a": np.asarray(a, float), "b": np.asarray(b, float)}, index=list(cases)).dropna()
    if len(d) < 3:
        return {"n": len(d), "mean_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_wilcoxon": np.nan,
                "a_better": np.nan}
    diff = d.a - d.b
    m, lo, hi = bootstrap_ci(diff.values, diff.index, n=n, seed=seed)
    if np.allclose(diff.values, 0):
        p = 1.0
    else:
        p = float(sps.wilcoxon(d.a, d.b, zero_method="wilcox", alternative="two-sided").pvalue)
    return {"n": len(d), "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p,
            "a_better": float((diff > 0).mean()), "b_better": float((diff < 0).mean())}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input). NaNs stay NaN."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out
