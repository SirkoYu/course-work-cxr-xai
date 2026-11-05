"""Поділ на вибірки без витоку."""
import pandas as pd

from cxr_xai.data import splitting


def test_group_split_has_no_leakage():
    rows = []
    gid = 0
    for label in ["COVID-19", "Lung Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"]:
        for g in range(40):
            for k in range(1 + (g % 3 == 0)):              # частина груп має 2 зображення
                rows.append({"path": f"{label}/{g}_{k}.png", "label": label, "group_id": gid,
                             "sha256": f"{gid}_{k}", "phash": f"{gid}"})
            gid += 1
    df = pd.DataFrame(rows)
    out = splitting.group_stratified_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    assert splitting.check_no_leakage(out) == []
    counts = out.groupby("group_id")["split"].first().value_counts(normalize=True)
    assert abs(counts["train"] - 0.70) < 0.05
    assert set(out.groupby("split")["label"].nunique()) == {5}


def test_leakage_is_detected():
    df = pd.DataFrame({"path": ["a", "b"], "group_id": [1, 1], "sha256": ["x", "y"],
                       "phash": ["p", "q"], "split": ["train", "test"]})
    assert splitting.check_no_leakage(df)
