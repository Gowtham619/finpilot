"""Score the categorizer on good_05 against its truth labels. NEEDS a Mesh key (MESH_API_KEY in .env).

Run:  .venv/bin/python testdata/score_categories.py
Prints rules-only vs rules+LLM accuracy and how many rows the LLM labelled. Costs a few cents.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finpilot.agents.categorizer import categorize_df  # noqa: E402
from finpilot.llm import LLM  # noqa: E402

HERE = pathlib.Path(__file__).parent


def main():
    df = pd.read_csv(HERE / "good_05_llm_categorization_stress.csv")
    df["date"] = pd.to_datetime(df["date"])
    truth = pd.read_csv(HERE / "good_05_truth_labels.csv").drop_duplicates("description").set_index("description")
    exp = df["description"].map(truth["expected_category"])
    llm = LLM()
    try:
        llm.json("Return {\"ok\":true}", "ping")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"LLM preflight failed ({type(e).__name__}): set MESH_API_KEY in .env first")
    for label, use in (("rules only", None), ("rules + LLM", llm)):
        out = categorize_df(df, use)
        acc = (out["category"] == exp).mean()
        n_llm = int((out["category_source"] == "llm").sum())
        unc = int((out["category"] == "uncategorized").sum())
        print(f"{label:12} accuracy {acc:.3f} | LLM-labelled rows {n_llm} | uncategorized {unc} | n={len(df)}")
    wrong = out[out["category"] != exp].assign(expected=exp)[["description", "category", "expected"]].head(15)
    print("\nFirst mislabelled rows (rules + LLM):")
    print(wrong.to_string(index=False))


if __name__ == "__main__":
    main()
