import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class LegalModernBertHeads(nn.Module):
    def __init__(self, model_dir: str | Path, attn_implementation: str = "flash_attention_2"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir)
        self.encoder = AutoModel.from_config(self.config, attn_implementation=attn_implementation)

        heads_config_path = _resolve_file(model_dir, "legal_heads_config.json")
        with heads_config_path.open("r", encoding="utf-8") as f:
            heads_config = json.load(f)
        self.heads_config = heads_config
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.config, "classifier_dropout", 0.1) or 0.1)
        self.doc_type_head = nn.Linear(hidden, len(heads_config["doc_type_id_to_label"]))
        self.classifier_head = nn.Linear(hidden, len(heads_config["classifier_id_to_label"]))
        self.keywords_head = nn.Linear(hidden, len(heads_config["keywords_id_to_label"]))
        self.ner_head = nn.Linear(hidden, len(heads_config["ner_id_to_label"]))

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = outputs.last_hidden_state
        pooled = self.dropout(sequence[:, 0])
        return {
            "doc_type_logits": self.doc_type_head(pooled),
            "classifier_logits": self.classifier_head(pooled),
            "keywords_logits": self.keywords_head(pooled),
            "ner_logits": self.ner_head(sequence),
        }


class LegalDocumentPipeline:
    def __init__(
        self,
        model: LegalModernBertHeads,
        tokenizer,
        max_length: int,
        ner_stride: int,
        device: str | torch.device | None = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ner_stride = ner_stride
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        cfg = model.heads_config
        self.doc_type_labels = _int_key_dict(cfg["doc_type_id_to_label"])
        self.classifier_labels = _int_key_dict(cfg["classifier_id_to_label"])
        self.keywords_labels = _int_key_dict(cfg["keywords_id_to_label"])
        self.ner_labels = _int_key_dict(cfg["ner_id_to_label"])

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str | torch.device | None = None,
        attn_implementation: str = "flash_attention_2",
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = LegalModernBertHeads(model_name_or_path, attn_implementation=attn_implementation)
        state = load_file(str(_resolve_file(model_name_or_path, "model.safetensors")))
        model.load_state_dict(state)
        heads_config = model.heads_config
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=heads_config.get("max_seq_length", 8192),
            ner_stride=heads_config.get("ner_stride", 1024),
            device=device,
        )

    @torch.inference_mode()
    def classify(self, text: str, top_k: int = 10, threshold: float = 0.5) -> dict:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)

        doc_probs = torch.softmax(outputs["doc_type_logits"][0].float(), dim=-1)
        doc_idx = int(torch.argmax(doc_probs).item())
        classifier_probs = torch.sigmoid(outputs["classifier_logits"][0].float())
        keyword_probs = torch.sigmoid(outputs["keywords_logits"][0].float())
        return {
            "doc_type": {"label": self.doc_type_labels[doc_idx], "score": float(doc_probs[doc_idx].item())},
            "classifier": _multi_label(classifier_probs, self.classifier_labels, top_k, threshold),
            "keywords": _multi_label(keyword_probs, self.keywords_labels, top_k, threshold),
        }

    @torch.inference_mode()
    def ner(self, text: str) -> list[dict]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.ner_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping", None)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded)["ner_logits"].float()
        predictions = torch.argmax(logits, dim=-1).cpu().tolist()
        spans = []
        seen = set()
        for chunk_preds, chunk_offsets in zip(predictions, offsets):
            active = None
            for pred_id, (start, end) in zip(chunk_preds, chunk_offsets):
                label = self.ner_labels[pred_id]
                if start == end or label == "O":
                    if active is not None:
                        _append_span(spans, seen, text, active)
                        active = None
                    continue
                prefix, entity = label.split("-", 1)
                if prefix == "B" or active is None or active["label"] != entity:
                    if active is not None:
                        _append_span(spans, seen, text, active)
                    active = {"start": start, "end": end, "label": entity}
                else:
                    active["end"] = end
            if active is not None:
                _append_span(spans, seen, text, active)
        return spans

    def __call__(self, text: str) -> dict:
        return {"classification": self.classify(text), "ner": self.ner(text)}


def _int_key_dict(values: dict) -> dict[int, str]:
    return {int(key): value for key, value in values.items()}


def _resolve_file(model_name_or_path: str | Path, filename: str) -> Path:
    local_path = Path(model_name_or_path) / filename
    if local_path.exists():
        return local_path
    return Path(hf_hub_download(repo_id=str(model_name_or_path), filename=filename))


def _multi_label(probs: torch.Tensor, labels: dict[int, str], top_k: int, threshold: float) -> list[dict]:
    values = [(idx, float(score.item())) for idx, score in enumerate(probs) if float(score.item()) >= threshold]
    values.sort(key=lambda item: item[1], reverse=True)
    if not values:
        values = [(idx, float(score.item())) for idx, score in enumerate(probs)]
        values.sort(key=lambda item: item[1], reverse=True)
    return [{"label": labels[idx], "score": score} for idx, score in values[:top_k]]


def _append_span(spans: list[dict], seen: set[tuple[int, int, str]], text: str, span: dict) -> None:
    key = (span["start"], span["end"], span["label"])
    if key in seen:
        return
    seen.add(key)
    spans.append({**span, "text": text[span["start"] : span["end"]]})
