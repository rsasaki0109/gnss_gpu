"""Train a RandomForest NLOS classifier on extracted features.

Reads CSV from extract_nlos_features.py; trains RF; reports accuracy + feature importance.

Usage:
    python experiments/train_nlos_classifier.py --train tokyo_run1.csv [more...] --test holdout.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix


def load_csv(p: Path, include_pr_res: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load features + labels.

    If include_pr_res=False (default), the pr_res features are EXCLUDED from
    X — they leak the label since `is_nlos` is derived from |pr_res|>8.
    At inference time, truth is unavailable so pr_res cannot be computed.
    """
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            lbl = int(r["is_nlos"])
            if lbl < 0:
                continue
            feats = [float(r["elev_deg"]), float(r["snr_dbhz"]), float(r["d_snr_dbhz"])]
            if include_pr_res:
                feats.extend([abs(float(r["pr_res_m"])), float(r["pr_res_std_m"])])
            rows.append(feats + [lbl])
    arr = np.array(rows, dtype=np.float64)
    if len(arr) == 0:
        nc = 5 if include_pr_res else 3
        return np.empty((0, nc)), np.empty((0,), dtype=np.int32)
    nc = arr.shape[1] - 1
    X = arr[:, :nc]
    y = arr[:, -1].astype(np.int32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=10)
    args = ap.parse_args()

    X_tr_list, y_tr_list = [], []
    for p in args.train:
        X, y = load_csv(p, include_pr_res=False)
        X_tr_list.append(X); y_tr_list.append(y)
    X_tr = np.vstack(X_tr_list)
    y_tr = np.concatenate(y_tr_list)
    X_te, y_te = load_csv(args.test, include_pr_res=False)

    print(f"Train: n={len(y_tr)}  LOS={np.sum(y_tr==0)} NLOS={np.sum(y_tr==1)}")
    print(f"Test : n={len(y_te)}  LOS={np.sum(y_te==0)} NLOS={np.sum(y_te==1)}")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        random_state=0, n_jobs=-1, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"Confusion (rows=true, cols=pred):\n{cm}")
    print(classification_report(y_te, y_pred, target_names=["LOS", "NLOS"]))

    feat_names = ["elev_deg", "snr_dbhz", "d_snr_dbhz"]
    print("Feature importance:")
    for name, imp in sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s}  {imp:.4f}")


if __name__ == "__main__":
    main()
