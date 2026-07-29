import numpy as np

from legal_modernbert_training.experiment_evidence import (
    deduplicate_ner_rows,
    multilabel_scores,
)


def test_deduplicate_ner_rows_removes_cross_split_duplicates_and_conflicts():
    rows = {
        "train": [
            {"text": "alpha", "spans": "[(0, 1, 'a', 2)]"},
            {"text": "duplicate", "spans": "[(0, 1, 'd', 2)]"},
            {"text": "conflict", "spans": "[(0, 1, 'c', 2)]"},
        ],
        "validation": [
            {"text": "duplicate", "spans": "[(0, 1, 'd', 2)]"},
            {"text": "beta", "spans": "[]"},
        ],
        "test": [
            {"text": "conflict", "spans": "[(0, 2, 'co', 4)]"},
            {"text": "gamma", "spans": "[]"},
        ],
    }

    result, audit = deduplicate_ner_rows(rows, seed=42)

    all_texts = [row["text"] for split in result.values() for row in split]
    assert sorted(all_texts) == ["alpha", "beta", "duplicate", "gamma"]
    assert len(all_texts) == len(set(all_texts))
    assert audit["conflicting_texts_removed"] == 1
    assert audit["duplicate_rows_removed"] == 1


def test_multilabel_scores_use_threshold_and_report_micro_macro_f1():
    logits = np.array([[5.0, -5.0], [-5.0, 5.0], [5.0, 5.0]])
    labels = np.array([[1, 0], [0, 1], [1, 0]])

    scores = multilabel_scores(logits, labels, threshold=0.5)

    assert scores["micro_f1"] == 6 / 7
    assert scores["macro_f1"] == (1.0 + 2 / 3) / 2
    assert scores["exact_match"] == 2 / 3
