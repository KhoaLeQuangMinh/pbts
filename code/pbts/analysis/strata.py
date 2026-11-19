"""Per-case features used to break results down ("who wins where").

Sources: the EDA per-case table (copied to data/head/eda_per_case.csv), the dataset manifest, the BraTS metadata
(hospital, age), and our preprocessing records (skull-stripped, oversized head). Image-quality flags come from
the visual QC of all 294 head masks (2026-11-19).
"""
import json

import numpy as np
import pandas as pd

from pbts import config as C

EDA_FILE = C.HEAD_DIR / "eda_per_case.csv"
QUALITY_FLAGS = {"BraTS-PED-00042-000": "motion/blur", "BraTS-PED-00228-000": "motion/blur"}


def _tertile(s: pd.Series, names=("small", "medium", "large")):
    q = s.quantile([1 / 3, 2 / 3]).values
    return pd.cut(s, [-np.inf, q[0], q[1], np.inf], labels=list(names)).astype(str)


def case_features() -> pd.DataFrame:
    eda = pd.read_csv(EDA_FILE).set_index("subject_id")
    recs = {p.stem: json.load(open(p)) for p in (C.HEAD_DIR / "records").glob("*.json")}
    f = pd.DataFrame(index=eda.index)
    f["follow_up"] = np.where(eda.timepoint >= 100, "follow-up", "baseline/naive")
    f["cohort"] = eda.cohort
    f["hospital"] = eda.data_source.fillna("unknown")
    f["tumor_size"] = _tertile(eda.WT_vol_ml)
    f["WT_ml"] = eda.WT_vol_ml
    for r in ("ET", "CC", "ED"):
        f[f"{r}_present"] = np.where(eda[f"{r}_present"].astype(bool), f"{r} present", f"{r} absent")
    f["side"] = eda.WT_side
    f["height_in_head"] = _tertile(eda.WT_centroid_z_rel_head, ("low (posterior fossa)", "middle", "high"))
    age = eda.age_years
    f["age_group"] = pd.cut(age, [0, 5, 10, 15, 100], labels=["<5", "5–10", "10–15", "≥15"]).astype(str) \
        .where(age.notna(), "unknown")
    f["n_labels_present"] = eda.n_labels_present.astype(int).astype(str) + " labels"
    f["skull_stripped"] = [("already stripped" if recs.get(c, {}).get("skull_stripped") else "not stripped") for c in f.index]
    f["head_oversized"] = [("oversized head" if recs.get(c, {}).get("oversized_vs_frame") else "fits frame") for c in f.index]
    f["image_quality"] = [QUALITY_FLAGS.get(c, "no flag") for c in f.index]
    return f


STRATA = ["follow_up", "hospital", "tumor_size", "ET_present", "CC_present", "ED_present", "side",
          "height_in_head", "age_group", "n_labels_present", "skull_stripped", "head_oversized", "image_quality"]
