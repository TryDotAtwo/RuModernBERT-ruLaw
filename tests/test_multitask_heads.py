from legal_modernbert_training.train_multitask_heads import build_input, parse_classifier, parse_keywords


def test_build_input_joins_heading_and_text():
    row = {"headingIPS": "О порядке учета", "textIPS": "Основной текст документа."}

    assert build_input(row) == "О порядке учета\n\nОсновной текст документа."


def test_parse_classifier_extracts_codes_before_dollar():
    value = "010.140$ Гражданское право 020.030$ Органы власти"

    assert parse_classifier(value) == ["010.140", "020.030"]


def test_parse_keywords_normalizes_comma_separated_values():
    value = "суд, Договор,  "

    assert parse_keywords(value) == ["СУД", "ДОГОВОР"]
