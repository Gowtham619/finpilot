import json

from ..config import CATEGORIES
from ..engines.categorize import normalize_merchant, rule_category
from ..llm import LLMUnavailable

SYSTEM = ("You classify bank-statement merchants. Allowed categories: "
          + ", ".join(CATEGORIES) + ". Reply with a JSON object mapping each given merchant "
          "string exactly to one category. No other keys.")


def _llm_map(unknown, llm):
    mapping = {}
    if not unknown or llm is None:
        return mapping
    unknown = sorted(unknown)
    for i in range(0, len(unknown), 40):
        batch = unknown[i:i + 40]
        try:
            res = llm.json(SYSTEM, json.dumps(batch))
        except (LLMUnavailable, ValueError):
            continue
        if not isinstance(res, dict):
            continue
        for m in batch:
            c = res.get(m)
            if isinstance(c, str):
                mapping[m] = c if c in CATEGORIES else "other"
    return mapping


def categorize_df(df, llm=None, overrides=None):
    out = df.copy()
    out["merchant"] = out["description"].map(normalize_merchant)
    overrides = overrides or {}
    cats, srcs, confs, unknown = [], [], [], set()
    for m, amt in zip(out["merchant"], out["amount"]):
        if m in overrides:
            cats.append(overrides[m]); srcs.append("user"); confs.append(1.0)
            continue
        c = rule_category(m, amt)
        if c:
            cats.append(c); srcs.append("rule"); confs.append(0.95)
        else:
            cats.append(None); srcs.append(None); confs.append(0.0)
            unknown.add(m)
    mapping = _llm_map(unknown, llm)
    for i, m in enumerate(out["merchant"]):
        if cats[i] is None:
            if m in mapping:
                cats[i], srcs[i], confs[i] = mapping[m], "llm", 0.75
            else:
                cats[i], srcs[i] = "uncategorized", "none"
    out["category"], out["category_source"], out["confidence"] = cats, srcs, confs
    return out
