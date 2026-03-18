"""
Custom data collator for Whisper Seq2Seq training.
Pads log-mel spectrograms and label token IDs in each batch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Collate a list of preprocessed examples into a padded batch.

    Each example must have:
        input_features : List[List[float]]  shape (80, <=3000)
        labels         : List[int]          token ids (with BOS prepended)
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── input_features ──────────────────────────────────────────────────────
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # ── labels ──────────────────────────────────────────────────────────────
        label_features = [{"input_ids": f["labels"]} for f in features]
        # Suppress the "use __call__ instead of pad" warning — we are padding
        # already-tokenized IDs, so __call__ is not applicable here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*WhisperTokenizerFast.*")
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If the BOS token was prepended by the tokenizer, strip it here —
        # Whisper's Seq2Seq model prepends it automatically during forward().
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
