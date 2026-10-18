"""Run categorization and detection on a public transaction CSV.

Usage: python -m eval.real_data path.csv eval/real_map.json
real_map.json: {"date": "<col>", "description": "<col>", "amount": "<col>",
                "label": "<col or null>", "label_map": {"<dataset label>": "<our category>"},
                "negate_amount": false, "source": "<dataset name + licence>"}
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from eval import metrics as M
from finpilot.agents.categorizer import categorize_df
from finpilot.engines import analyze as A
from finpilot.llm import LLM


def run(csv_path, mapping_path, llm):
    m = json.load(open(mapping_path))
    if m.get("label") and not m.get("label_map"):
        raise ValueError("mapping sets 'label' but has no 'label_map' (dataset label -> FinPilot category)")
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame({"date": pd.to_datetime(raw[m["date"]], errors="coerce"),
                       "description": raw[m["description"]].astype(str),
                       "amount": pd.to_numeric(raw[m["amount"]], errors="coerce")})
    if m.get("negate_amount"):
        df["amount"] = -df["amount"]
    if m.get("label"):
        df["_label"] = raw[m["label"]]
    df = df.dropna(subset=["date", "amount"]).sort_values("date", kind="stable").reset_index(drop=True)
    labels = df.pop("_label") if m.get("label") else None
    cat = categorize_df(df, llm)
    out = {"source": m.get("source"), "rows": len(df),
           "source_mix": cat["category_source"].value_counts().to_dict()}
    if labels is not None:
        truth = labels.map(m["label_map"])
        mask = truth.notna()
        out["labelled_rows"] = int(mask.sum())
        out["accuracy_vs_dataset_labels"] = round(M.accuracy(truth[mask], cat["category"][mask]), 4)
        out["macro_f1_vs_dataset_labels"] = round(M.macro_f1(truth[mask], cat["category"][mask]), 4)
    findings = A.analyze_all(cat)
    out["findings_by_type"] = pd.Series([f["type"] for f in findings]).value_counts().to_dict()
    out["sample_for_manual_review"] = [f.get("merchant") or f.get("category") for f in findings[:10]]
    return out


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2], LLM()), indent=2, default=str))
