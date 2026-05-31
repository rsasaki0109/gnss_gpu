"""Quick test: can elev + snr + d_snr predict NLOS pseudo-label?

Random 80/20 split within one run. If AUC > 0.75, there's predictive signal
from inference-time features.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


def load(p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            lbl = int(r["is_nlos"])
            if lbl < 0:
                continue
            rows.append((
                float(r["elev_deg"]), float(r["snr_dbhz"]), float(r["d_snr_dbhz"]),
                abs(float(r["pr_res_m"])), float(r["pr_res_std_m"]),
                lbl
            ))
    arr = np.array(rows, dtype=np.float64)
    X = arr[:, :3]   # elev, snr, d_snr
    X_with_pr_res = arr[:, :5]
    y = arr[:, 5].astype(np.int32)
    return X, X_with_pr_res, y


def evaluate(model, X_tr, X_te, y_tr, y_te, label):
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba)
    print(f"\n{label}  AUC={auc:.4f}")
    if hasattr(model, "feature_importances_"):
        print(f"  importances: {model.feature_importances_}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    X, X_full, y = load(args.csv)
    print(f"n={len(y)}  LOS={np.sum(y==0)} NLOS={np.sum(y==1)}")
    print(f"baseline (always predict NLOS): acc={np.sum(y==1)/len(y):.4f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_tr_full, X_te_full, _, _ = train_test_split(X_full, y, test_size=0.2, random_state=0, stratify=y)

    print("\n=== Inference-time features [elev, snr, d_snr] ===")
    evaluate(LogisticRegression(max_iter=200), X_tr, X_te, y_tr, y_te, "  LogReg")
    evaluate(RandomForestClassifier(n_estimators=200, max_depth=12,
                                     random_state=0, n_jobs=-1, class_weight="balanced"),
              X_tr, X_te, y_tr, y_te, "  RF")
    evaluate(GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=0),
              X_tr, X_te, y_tr, y_te, "  GBDT")

    print("\n=== Cheat features [elev, snr, d_snr, |pr_res|, pr_std] (label leak) ===")
    evaluate(RandomForestClassifier(n_estimators=200, max_depth=12,
                                     random_state=0, n_jobs=-1, class_weight="balanced"),
              X_tr_full, X_te_full, y_tr, y_te, "  RF-cheat")


if __name__ == "__main__":
    main()
