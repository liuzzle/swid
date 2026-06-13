#!/usr/bin/env python3
"""
Train and evaluate speaker-identification classifiers on pre-extracted embeddings.

Implements three classifiers:
  cosine   — nearest-centroid using cosine similarity (no training required)
  knn      — k-nearest-neighbours (sklearn, cosine metric)
  svm      — LinearSVC with L2-normalised embeddings

Embeddings are L2-normalised before all classifiers.  For Whisper embeddings
(shape: layers × hidden) a single layer is selected via ``--whisper-layer``
(default: -1, last transformer layer) or ``mean`` to average across layers.

Usage:
  python scripts/classify_speakers.py \\
    --train metadata/dur_1s/xvector_train.csv \\
    --test  metadata/dur_1s/xvector_test.csv  \\
    --emb-type xvector \\
    --out   results/dur_1s_xvector.json

  python scripts/classify_speakers.py \\
    --train metadata/dur_1s/whisper_train.csv \\
    --test  metadata/dur_1s/whisper_test.csv  \\
    --emb-type whisper --whisper-layer -1 \\
    --out   results/dur_1s_whisper_last.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(
    csv_path: Path,
    emb_type: str,
    whisper_layer: str = "-1",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays from a manifest CSV.

    X: float32 array of shape (n_clips, hidden_dim)
    y: string array of speaker IDs
    """
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            arr = np.load(row["embedding_path"])

            if emb_type == "xvector":
                vec = arr.reshape(-1)   # (1, 512) → (512,)
            else:
                # whisper: (layers, hidden)
                if whisper_layer == "mean":
                    vec = arr.mean(axis=0)
                else:
                    vec = arr[int(whisper_layer)]

            X_list.append(vec.astype(np.float32))
            y_list.append(row["speaker_id"])

    return np.stack(X_list), np.array(y_list)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def cosine_centroid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Nearest-centroid classifier using cosine similarity.

    Per-speaker centroids are computed from train embeddings and L2-normalised.
    Test embeddings are also L2-normalised; prediction = argmax dot-product.
    """
    t0 = time.perf_counter()
    speakers = np.unique(y_train)
    centroids = np.stack([
        X_train[y_train == spk].mean(axis=0) for spk in speakers
    ])
    centroids = l2_normalize(centroids)
    X_test_n = l2_normalize(X_test)
    sims = X_test_n @ centroids.T          # (n_test, n_speakers)
    preds = speakers[sims.argmax(axis=1)]
    acc = float((preds == y_test).mean())
    elapsed = time.perf_counter() - t0
    return {"accuracy": acc, "elapsed_sec": elapsed, "predictions": preds}


def knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
) -> Dict:
    t0 = time.perf_counter()
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float((preds == y_test).mean())
    elapsed = time.perf_counter() - t0
    return {"accuracy": acc, "k": k, "elapsed_sec": elapsed, "predictions": preds}


def svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
) -> Dict:
    # L2-normalise so cosine ≈ dot product (standard practice for speaker SVM)
    X_tr = l2_normalize(X_train)
    X_te = l2_normalize(X_test)
    t0 = time.perf_counter()
    clf = LinearSVC(C=C, max_iter=2000)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    acc = float((preds == y_test).mean())
    elapsed = time.perf_counter() - t0
    return {"accuracy": acc, "C": C, "elapsed_sec": elapsed, "predictions": preds}


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute per-speaker and aggregate precision/recall/F1.

    Returns a dict with:
      per_class: {speaker_id: {precision, recall, f1, support}}
      macro_precision/macro_recall/macro_f1: unweighted mean over speakers
      weighted_precision/weighted_recall/weighted_f1: support-weighted mean
    Labels are fixed to the speakers present in y_true so every test speaker
    appears (a never-predicted speaker gets precision/recall/f1 = 0).
    """
    labels = np.unique(y_true)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        str(lbl): {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lbl in enumerate(labels)
    }

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "per_class": per_class,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate(
    train_csv: Path,
    test_csv: Path,
    emb_type: str,
    whisper_layer: str = "-1",
    classifiers: List[str] | None = None,
    knn_k: int = 5,
    svm_c: float = 1.0,
    per_class: bool = False,
) -> Dict:
    """Load embeddings and run all requested classifiers; return results dict.

    When ``per_class`` is True, each classifier's entry also carries per-speaker
    and macro/weighted precision/recall/F1 (computed from its predictions).  The
    raw ``predictions`` array is dropped from the returned dict either way so the
    result stays JSON-serialisable and compact.
    """
    if classifiers is None:
        classifiers = ["cosine", "knn", "svm"]

    X_train, y_train = load_embeddings(train_csv, emb_type, whisper_layer)
    X_test, y_test = load_embeddings(test_csv, emb_type, whisper_layer)

    results: Dict = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "emb_type": emb_type,
        "whisper_layer": whisper_layer,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_speakers": int(len(np.unique(y_train))),
        "embedding_dim": int(X_train.shape[1]),
        "classifiers": {},
    }

    runners = {
        "cosine": lambda: cosine_centroid(X_train, y_train, X_test, y_test),
        "knn": lambda: knn(X_train, y_train, X_test, y_test, k=knn_k),
        "svm": lambda: svm(X_train, y_train, X_test, y_test, C=svm_c),
    }

    for name in ("cosine", "knn", "svm"):
        if name not in classifiers:
            continue
        res = runners[name]()
        preds = res.pop("predictions")
        if per_class:
            res.update(per_class_metrics(y_test, preds))
        results["classifiers"][name] = res

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate speaker-ID classifiers on embedding manifests")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test",  type=Path, required=True)
    p.add_argument("--emb-type", choices=["xvector", "whisper"], required=True)
    p.add_argument("--whisper-layer", default="-1",
                   help="Layer index (0-based) or 'mean'. Default: -1 (last layer)")
    p.add_argument("--classifiers", nargs="+", default=["cosine", "knn", "svm"],
                   choices=["cosine", "knn", "svm"])
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--per-class", action="store_true",
                   help="Also compute per-speaker and macro/weighted P/R/F1")
    p.add_argument("--out", type=Path, default=None,
                   help="Write results JSON to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(
        train_csv=args.train,
        test_csv=args.test,
        emb_type=args.emb_type,
        whisper_layer=args.whisper_layer,
        classifiers=args.classifiers,
        knn_k=args.knn_k,
        svm_c=args.svm_c,
        per_class=args.per_class,
    )

    print(json.dumps(results, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
