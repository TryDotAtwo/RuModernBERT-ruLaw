from legal_modernbert_training.text_pipeline import build_mlm_text, chunk_token_ids, is_usable_text


def test_build_mlm_text_uses_only_textips():
    row = {
        "headingIPS": "Ignored heading",
        "textIPS": "Main legal document text.",
    }

    assert build_mlm_text(row, "textIPS") == "Main legal document text."


def test_is_usable_text_rejects_empty_and_short_values():
    assert not is_usable_text(None, min_chars=20)
    assert not is_usable_text("   ", min_chars=20)
    assert not is_usable_text("too short", min_chars=20)
    assert is_usable_text("This text is long enough.", min_chars=20)


def test_chunk_token_ids_uses_overlap():
    chunks = chunk_token_ids(list(range(10)), max_length=6, overlap=2)

    assert chunks == [
        list(range(0, 6)),
        list(range(4, 10)),
    ]
