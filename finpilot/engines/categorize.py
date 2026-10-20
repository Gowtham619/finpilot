import re

STOP = {"UPI", "POS", "NEFT", "IMPS", "ACH", "PAYMENT", "BANGALORE", "BLR", "MUMBAI", "DELHI", "PUNE"}

RULES = {  # order matters: first match wins
    "rent": ["RENT"],
    "utilities": ["ELECTRICITY", "BROADBAND", "WATER", "GAS", "BESCOM", "AIRTEL", "JIO FIBER"],
    "subscriptions": ["NETFLIX", "SPOTIFY", "HOTSTAR", "PRIME VIDEO", "CULT FIT", "ADOBE", "YOUTUBE"],
    "groceries": ["BIGBASKET", "DMART", "GROFERS", "SUPERMARKET"],
    "food_delivery": ["SWIGGY", "ZOMATO"],
    "dining": ["RESTAURANT", "CAFE"],
    "transport": ["UBER", "OLA", "METRO", "RAPIDO"],
    "shopping": ["AMAZON", "FLIPKART", "MYNTRA"],
    "health": ["PHARMACY", "HOSPITAL"],
}
_COMPILED = [(c, re.compile(r"\b(" + "|".join(re.escape(k) for k in ks) + r")\b"))
             for c, ks in RULES.items()]


def normalize_merchant(desc: str) -> str:
    toks = re.findall(r"[A-Za-z]+", str(desc).upper())
    return " ".join(t for t in toks if t not in STOP and len(t) > 1)


def rule_category(merchant: str, amount: float):
    if amount > 0:
        return "income"
    for cat, rx in _COMPILED:
        if rx.search(merchant):
            return cat
    return None
