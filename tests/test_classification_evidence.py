import numpy as np

from legal_modernbert_training.experiment_evidence import classification_scores


def test_classification_scores_report_accuracy_and_macro_f1():
    logits = np.array([[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    labels = np.array([0, 1, 0])

    scores = classification_scores(logits, labels)

    assert scores["accuracy"] == 2 / 3
    assert scores["macro_f1"] == 2 / 3
