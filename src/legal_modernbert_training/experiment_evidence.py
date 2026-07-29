from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


def _text_key(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_ner_rows(rows_by_split, *, seed, train_ratio=0.8, validation_ratio=0.1):
    grouped = defaultdict(list)
    total_rows = 0
    for rows in rows_by_split.values():
        for row in rows:
            total_rows += 1
            grouped[_text_key(row["text"])].append(row)

    unique_rows = []
    conflicting_texts = 0
    conflicting_rows = 0
    duplicate_rows = 0
    for text_hash, rows in grouped.items():
        annotations = {row["spans"] for row in rows}
        if len(annotations) != 1:
            conflicting_texts += 1
            conflicting_rows += len(rows)
            continue
        unique_rows.append((text_hash, rows[0]))
        duplicate_rows += len(rows) - 1

    result = {"train": [], "validation": [], "test": []}
    for text_hash, row in unique_rows:
        digest = hashlib.sha256(f"{seed}:{text_hash}".encode()).hexdigest()
        value = int(digest[:16], 16) / float(16**16)
        if value < train_ratio:
            split = "train"
        elif value < train_ratio + validation_ratio:
            split = "validation"
        else:
            split = "test"
        result[split].append(row)

    audit = {
        "input_rows": total_rows,
        "output_rows": sum(map(len, result.values())),
        "unique_texts": len(grouped),
        "conflicting_texts_removed": conflicting_texts,
        "conflicting_rows_removed": conflicting_rows,
        "duplicate_rows_removed": duplicate_rows,
    }
    return result, audit


def multilabel_scores(logits, labels, *, threshold=0.5):
    predictions = (1.0 / (1.0 + np.exp(-logits))) >= threshold
    truth = labels.astype(bool)
    tp = np.logical_and(predictions, truth).sum(axis=0)
    fp = np.logical_and(predictions, ~truth).sum(axis=0)
    fn = np.logical_and(~predictions, truth).sum(axis=0)
    denominators = 2 * tp + fp + fn
    per_label_f1 = np.divide(
        2 * tp,
        denominators,
        out=np.zeros_like(tp, dtype=float),
        where=denominators != 0,
    )
    micro_denominator = 2 * tp.sum() + fp.sum() + fn.sum()
    return {
        "micro_f1": float(2 * tp.sum() / micro_denominator if micro_denominator else 0),
        "macro_f1": float(per_label_f1.mean()),
        "exact_match": float(np.all(predictions == truth, axis=1).mean()),
    }

def classification_scores(logits, labels):
    predictions = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)
    classes = np.union1d(labels, predictions)
    f1_values = []
    for label in classes:
        predicted = predictions == label
        actual = labels == label
        tp = np.logical_and(predicted, actual).sum()
        fp = np.logical_and(predicted, ~actual).sum()
        fn = np.logical_and(~predicted, actual).sum()
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(np.mean(f1_values)),
    }
