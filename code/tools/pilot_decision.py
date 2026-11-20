"""Decision 1 evidence (protocol.PILOT): whole-head vs 128^3 patches, and training length from the 25/50/75/100%
checkpoints of the whole-head run. Read-only; prints tables and writes reports/pilot_decision.json.

Every checkpoint is compared, case by case on the same 48 validation scans, against the whole-head 100% checkpoint
(paired difference with a patient-bootstrap 95% CI). Regions: ET NET CC ED TC WT.
    official DSC   = official 2026 scorer, whole-volume Dice; a region absent in both is NaN and dropped
    lesion-wise    = BraTS 2023-25 lesion-wise Dice (absent in both = 1, missed or false lesions = 0)
    false alarms   = share of scans WITHOUT the region where the model still predicted some of it
"""
import json
import sys

import numpy as np
import pandas as pd

from pbts import config as C
from pbts.analysis.stats import bootstrap_ci, paired_compare

REGIONS = ["ET", "NET", "CC", "ED", "TC", "WT"]
RUNS = {"wholehead": C.RUNS_DIR / "nnunet_resenc_l/default/fold_0",
        "patch128": C.RUNS_DIR / "nnunet_resenc_l/pilot_patch128/fold_0"}
CKPTS = ["025", "050", "075", "100"]
METRICS = {"official DSC": "off_global_dsc", "lesion-wise": "lw2023_dice", "official NSD": "off_global_nsd"}


def load(run, ck):
    df = pd.read_csv(RUNS[run] / f"eval/ckpt{ck}/per_case_region.csv")
    return df.pivot(index="case", columns="region")


def main():
    data = {(r, c): load(r, c) for r in RUNS for c in CKPTS}
    ref = data[("wholehead", "100")]
    cases = ref.index
    out = {"n_cases": len(cases), "rows": []}
    for (run, ck), df in data.items():
        df = df.loc[cases]
        for reg in REGIONS:
            row = {"run": run, "ckpt": ck, "region": reg}
            for name, col in METRICS.items():
                v = df[(col, reg)].astype(float)
                m, lo, hi = bootstrap_ci(v.values, cases)
                row[name] = m
                row[name + " CI"] = (lo, hi)
                if (run, ck) != ("wholehead", "100"):
                    pc = paired_compare(v.values, ref[(col, reg)].astype(float).values, cases)
                    row[name + " diff"] = pc["mean_diff"]
                    row[name + " diff CI"] = (pc["ci_lo"], pc["ci_hi"])
                    row[name + " p"] = pc["p_wilcoxon"]
            absent = ~df[("gt_present", reg)].astype(bool)
            row["n_absent"] = int(absent.sum())
            row["false alarms"] = float(df[("pred_present", reg)].astype(bool)[absent].mean()) if absent.any() else np.nan
            out["rows"].append(row)
    rows = pd.DataFrame(out["rows"])

    def fmt_diff(r, k):
        if pd.isna(r.get(k + " diff", np.nan)):
            return "      ref       "
        lo, hi = r[k + " diff CI"]
        star = "*" if (lo > 0 or hi < 0) else " "
        return f"{r[k + ' diff']:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"

    for k in METRICS:
        print(f"\n=== {k}: mean (diff vs whole-head 100%, 95% patient-bootstrap CI; * = CI excludes 0)")
        print(f"{'run':10s} {'ckpt':4s} " + " ".join(f"{reg:>30s}" for reg in REGIONS))
        for (run, ck), g in rows.groupby(["run", "ckpt"], sort=False):
            g = g.set_index("region")
            print(f"{run:10s} {ck:4s} " + " ".join(f"{g.loc[reg, k]:.3f} {fmt_diff(g.loc[reg], k):>24s}" for reg in REGIONS))
    print("\n=== false alarms on scans without the region (share; n absent in brackets)")
    for (run, ck), g in rows.groupby(["run", "ckpt"], sort=False):
        g = g.set_index("region")
        print(f"{run:10s} {ck:4s} " + " ".join(f"{reg} {g.loc[reg, 'false alarms']:.2f} ({g.loc[reg, 'n_absent']})"
                                              for reg in REGIONS))
    dest = C.REPORTS_DIR / "pilot_decision.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1, default=lambda o: None if pd.isna(o) else o)
    print("\nwritten", dest, file=sys.stderr)


if __name__ == "__main__":
    main()
