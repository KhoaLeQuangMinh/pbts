"""Pure-python tests of the statistics (run anywhere: python -m pytest tests/test_analysis.py)."""
import numpy as np
import pytest

from pbts.analysis.stats import bootstrap_ci, holm, paired_compare


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, np.nan, 0.5]
    adj = holm(p)
    # sorted 0.01,0.03,0.04,0.5 (m=4): 0.04, max(0.04,0.09)=0.09, max(0.09,0.08)=0.09, max(0.09,0.5)=0.5
    assert np.allclose(adj[[0, 2, 1, 4]], [0.04, 0.09, 0.09, 0.5]) and np.isnan(adj[3])


def test_bootstrap_resamples_patients_not_scans():
    cases = ["BraTS-PED-00001-000", "BraTS-PED-00001-100", "BraTS-PED-00001-101", "BraTS-PED-00002-000"]
    vals = [1.0, 1.0, 1.0, 0.0]
    m, lo, hi = bootstrap_ci(vals, cases, n=4000, seed=1)
    assert m == pytest.approx(0.75)
    # only 2 patients: the resampled mean is 1.0 (both draws patient 1), 0.0 (both patient 2) or 0.75
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)


def test_paired_compare_sign_and_nan_handling():
    cases = [f"BraTS-PED-{i:05d}-000" for i in range(30)]
    a = np.linspace(0.5, 0.9, 30); b = a - 0.05; b[3] = np.nan
    r = paired_compare(a, b, cases)
    assert r["n"] == 29 and r["mean_diff"] == pytest.approx(0.05) and r["a_better"] == 1.0 and r["p_wilcoxon"] < 1e-4
