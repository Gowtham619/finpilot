def accuracy(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    f1s = []
    for c in sorted(set(y_true)):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def precision_recall(pred, truth):
    pred, truth = set(pred), set(truth)
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f, 6)}


def brier(probs, outcomes):
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)
