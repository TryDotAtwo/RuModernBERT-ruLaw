from legal_modernbert.pipeline import _append_span, _int_key_dict, _multi_label


def test_int_key_dict_converts_json_keys():
    assert _int_key_dict({"0": "O", "1": "B-2"}) == {0: "O", 1: "B-2"}


def test_append_span_deduplicates():
    spans = []
    seen = set()
    _append_span(spans, seen, "abcdef", {"start": 1, "end": 3, "label": "2"})
    _append_span(spans, seen, "abcdef", {"start": 1, "end": 3, "label": "2"})

    assert spans == [{"start": 1, "end": 3, "label": "2", "text": "bc"}]
