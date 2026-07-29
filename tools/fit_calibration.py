"""Fit the confidence model on cached training extractions.

Reports 5-fold cross-validated Brier error (the honest estimate of what the
model buys on unseen packets) and then refits on everything and writes
mib/calibration.json.
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mib.confidence import features
from mib.rules import adjudicate, batch_reference_date, batch_revoked_sponsors, resolve_fields

L2 = 1.0
STEPS = 4000
LR = 0.5


def build(truth_path, cache_path):
    truth = {r["case_id"]: r for r in csv.DictReader(open(truth_path))}
    records = [json.loads(line) for line in open(cache_path)]
    resolved = [(rec, resolve_fields(rec)) for rec in records]
    reference = batch_reference_date([f for _, f in resolved])
    revoked_sponsors = batch_revoked_sponsors([f for _, f in resolved])
    rows, labels = [], []
    for rec, fields in resolved:
        gold = truth.get(rec["case_id"])
        if not gold:
            continue
        adj, reasons = adjudicate(rec, fields, reference_date=reference, revoked_sponsors=revoked_sponsors)
        rows.append(features(rec, fields, adj, reasons))
        labels.append(1.0 if adj == gold["adjudication"] else 0.0)
    names = sorted({k for r in rows for k in r})
    X = np.array([[r.get(n, 0.0) for n in names] for r in rows])
    y = np.array(labels)
    return names, X, y


def train(X, y, steps=STEPS, lr=LR, l2=L2):
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) / n + l2 * w / n
        grad[0] -= l2 * w[0] / n            # do not regularise the bias
        w -= lr * grad
    return w


def predict(X, w, clip=(0.03, 0.97)):
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
    return np.clip(p, *clip)


def main():
    names, X, y = build(sys.argv[1], sys.argv[2])
    print(f"{len(y)} cases, {X.shape[1]} features, base accuracy {y.mean():.1%}")

    rng = np.random.default_rng(0)
    folds = rng.permutation(len(y)) % 5
    oof = np.zeros(len(y))
    for k in range(5):
        train_idx, test_idx = folds != k, folds == k
        w = train(X[train_idx], y[train_idx])
        oof[test_idx] = predict(X[test_idx], w)
    cv_brier = float(np.mean((oof - y) ** 2))
    const_brier = float(np.mean((y.mean() - y) ** 2))
    print(f"cross-validated Brier {cv_brier:.4f}  ->  calibration {20*max(0,1-2*cv_brier):.2f}/20")
    print(f"constant-p baseline  {const_brier:.4f}  ->  calibration {20*max(0,1-2*const_brier):.2f}/20")

    w = train(X, y)
    out = {"weights": {n: float(v) for n, v in zip(names, w)}, "clip": [0.03, 0.97],
           "cv_brier": cv_brier, "n_train": int(len(y))}
    path = Path(__file__).resolve().parents[1] / "mib" / "calibration.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
