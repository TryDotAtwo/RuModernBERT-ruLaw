from legal_modernbert_training.train_ner_head import LABEL_TO_ID, label_for_offset, parse_spans


def test_parse_spans_keeps_known_entity_ids():
    value = "[[10, 15, 'истец', 2], [20, 25, 'noise', 99], [30, 33, 'ООО', 4]]"

    assert parse_spans(value) == [(10, 15, 2), (30, 33, 4)]


def test_label_for_offset_builds_bio_labels():
    spans = [(10, 20, 2)]

    first_label, entity = label_for_offset(10, 14, spans, None)
    second_label, entity = label_for_offset(14, 18, spans, entity)
    outside_label, entity = label_for_offset(25, 30, spans, entity)

    assert first_label == LABEL_TO_ID["B-2"]
    assert second_label == LABEL_TO_ID["I-2"]
    assert outside_label == LABEL_TO_ID["O"]
    assert entity is None
