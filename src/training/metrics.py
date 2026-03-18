"""
Evaluation metrics for Quranic ASR with tashkeel.

Three metrics are reported at every eval step:
  wer            – Word Error Rate  WITH full tashkeel (primary quality signal)
  cer            – Char Error Rate  WITH full tashkeel (primary for diacritization)
  wer_normalized – Word Error Rate  WITHOUT tashkeel   (baseline comparison)
"""

from __future__ import annotations

import re
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Unicode range for Arabic diacritics (tashkeel + tatweel)
_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritics and tatweel from text."""
    return _TASHKEEL_RE.sub("", text)


def make_compute_metrics(processor: Any) -> Callable:
    """
    Returns a compute_metrics function compatible with Seq2SeqTrainer.
    Loads evaluate metrics once at construction time.
    """
    import evaluate

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Greedy decode may return logits; take argmax if needed
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        # Replace -100 padding back to pad_token_id before decoding
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Strip leading/trailing whitespace
        pred_str = [p.strip() for p in pred_str]
        label_str = [l.strip() for l in label_str]

        # Log a few examples for inspection
        for i in range(min(3, len(pred_str))):
            logger.info("REF : %s", label_str[i])
            logger.info("PRED: %s", pred_str[i])

        # ── Metrics WITH tashkeel ──────────────────────────────────────────────
        wer_val = 100.0 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer_val = 100.0 * cer_metric.compute(predictions=pred_str, references=label_str)

        # ── Metrics WITHOUT tashkeel (normalized) ─────────────────────────────
        pred_norm = [strip_tashkeel(p) for p in pred_str]
        label_norm = [strip_tashkeel(l) for l in label_str]
        wer_norm_val = 100.0 * wer_metric.compute(predictions=pred_norm, references=label_norm)

        return {
            "wer": round(wer_val, 4),
            "cer": round(cer_val, 4),
            "wer_normalized": round(wer_norm_val, 4),
        }

    return compute_metrics


# ── Type hint alias ───────────────────────────────────────────────────────────
from typing import Any
