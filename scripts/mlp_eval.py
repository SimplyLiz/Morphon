#!/usr/bin/env python3
"""
MLP vs Linear Readout diagnostic on frozen morphon activations.

Usage:
    python3 scripts/mlp_eval.py docs/benchmark_results/v4.4.0/v3_activations.csv

Tests whether morphon features support higher accuracy with a non-linear readout.
If MLP >> linear accuracy: readout is the bottleneck, not features.
If MLP ≈ linear accuracy: features themselves are the ceiling.
"""

import sys
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def load_csv(path):
    print(f"Loading {path} ...", flush=True)
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    labels = np.array([int(r[0]) for r in rows])
    features = np.array([[float(x) for x in r[1:]] for r in rows])
    print(f"  {len(labels)} samples, {features.shape[1]} features, {len(set(labels))} classes")
    return features, labels

def eval_classifier(name, clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) * 100
    print(f"\n{'━'*50}")
    print(f"  {name}: {acc:.1f}%")
    preds = clf.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    print("\n  Confusion matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{i:>4}" for i in range(10)))
    for i, row in enumerate(cm):
        recall = cm[i, i] / row.sum() * 100 if row.sum() > 0 else 0
        print(f"{i} " + "".join(f"{v:>5}" for v in row) + f"  | {recall:.0f}%")
    # Per-class precision
    col_sums = cm.sum(axis=0)
    prec = [cm[i,i]/col_sums[i]*100 if col_sums[i] > 0 else 0 for i in range(10)]
    print("  " + "".join(f"{int(p):>4}%" for p in prec) + "  (prec)")
    return acc

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/benchmark_results/v4.4.0/v3_activations.csv"
    X, y = load_csv(path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use 80% for training the readout, 20% for test — matches morphon training split roughly
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(y_train)}, Test: {len(y_test)}")

    results = {}

    # 1. Logistic regression — equivalent to the linear readout in Rust
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    results["LogReg (≈linear readout)"] = eval_classifier(
        "LogReg (≈linear readout)", lr, X_train, X_test, y_train, y_test
    )

    # 2. MLP — 1 hidden layer, same as adding one non-linear layer over frozen features
    mlp_1 = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42,
                           learning_rate_init=0.001, early_stopping=True)
    results["MLP 256"] = eval_classifier(
        "MLP (256 hidden)", mlp_1, X_train, X_test, y_train, y_test
    )

    # 3. MLP — 2 hidden layers
    mlp_2 = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500, random_state=42,
                           learning_rate_init=0.001, early_stopping=True)
    results["MLP 512→256"] = eval_classifier(
        "MLP (512→256)", mlp_2, X_train, X_test, y_train, y_test
    )

    print(f"\n{'━'*50}")
    print("  SUMMARY")
    print(f"{'━'*50}")
    for name, acc in results.items():
        print(f"  {name:<30} {acc:.1f}%")

    best_mlp = max(results["MLP 256"], results["MLP 512→256"])
    linear = results["LogReg (≈linear readout)"]
    delta = best_mlp - linear
    print(f"\n  MLP gain over linear: {delta:+.1f}pp")
    if delta > 5:
        print("  → Features are sufficient. Linear readout is the bottleneck.")
        print("    Recommendation: implement 1-hidden-layer readout in Rust.")
    elif delta > 2:
        print("  → Small MLP gain. Both features and readout contribute to the ceiling.")
    else:
        print("  → No MLP gain. Feature representation is the bottleneck.")
        print("    Recommendation: improve morphon feature quality (more morphons, better iSTDP).")

if __name__ == "__main__":
    main()
