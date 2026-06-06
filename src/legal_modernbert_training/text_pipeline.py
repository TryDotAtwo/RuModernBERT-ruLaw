def build_mlm_text(row: dict, text_column: str = "textIPS") -> str:
    value = row.get(text_column)
    return value.strip() if isinstance(value, str) else ""


def is_usable_text(text: str | None, min_chars: int = 200) -> bool:
    return isinstance(text, str) and len(text.strip()) >= min_chars


def chunk_token_ids(token_ids: list[int], max_length: int, overlap: int) -> list[list[int]]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if overlap < 0 or overlap >= max_length:
        raise ValueError("overlap must be non-negative and smaller than max_length.")
    if len(token_ids) <= max_length:
        return [token_ids]

    step = max_length - overlap
    chunks = []
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start : start + max_length]
        if chunk:
            chunks.append(chunk)
        if start + max_length >= len(token_ids):
            break
    return chunks
