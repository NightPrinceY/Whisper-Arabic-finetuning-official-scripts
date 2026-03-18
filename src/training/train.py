#!/usr/bin/env python3
"""
Fine-tune openai/whisper-small on tarteel-ai/everyayah (Quranic Arabic ASR).

Features:
  - Full fine-tuning (not LoRA) for best tashkeel generation quality
  - fp16 mixed precision + gradient checkpointing
  - Multi-GPU via accelerate (4× RTX 2080 Ti)
  - Streaming dataset (117 GB) — no full download needed
  - TensorBoard logging + HuggingFace Hub push (weights, logs, model card)
  - Three evaluation metrics: WER/CER with tashkeel, WER normalized

Usage (single GPU):
    python src/training/train.py --config configs/config.yaml

Usage (multi-GPU, recommended):
    accelerate launch --num_processes=4 --mixed_precision=fp16 \\
        src/training/train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper Quran fine-tuning")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output_dir from config")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint dir to resume from")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Limit training samples (smoke test)")
    p.add_argument("--smoke_test", action="store_true",
                   help="Quick 100-step smoke test with tiny data subset")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessing_fn(processor, config: dict):
    """Returns a function that converts a raw dataset example to model inputs."""
    from src.data.audio_utils import load_audio_from_bytes

    target_sr = config["target_sampling_rate"]
    max_dur = config["max_audio_length_seconds"]
    # Whisper decoder max = 448; never truncate Quranic ayahs mid-verse
    max_target_length = config.get("generation_max_length", 448)

    def prepare(example: dict) -> dict:
        # ── Audio ──────────────────────────────────────────────────────────────
        raw = example["audio"]
        audio_array, _ = load_audio_from_bytes(
            audio_bytes=raw.get("bytes"),
            audio_path=raw.get("path"),
            target_sr=target_sr,
            max_duration_sec=max_dur,
        )

        # Log-mel spectrogram via WhisperFeatureExtractor
        features = processor.feature_extractor(
            audio_array,
            sampling_rate=target_sr,
            return_tensors="np",
        )
        example["input_features"] = features.input_features[0]  # (80, 3000)

        # ── Text ───────────────────────────────────────────────────────────────
        # Keep full tashkeel — do NOT normalize.
        # Data is pre-filtered by scripts/prefilter_dataset.py:
        #   - audio > 30s removed (corrupt signal: truncated audio, full-text label)
        #   - tokens > 448 removed (Whisper decoder hard limit)
        text = example["text"].strip()
        example["labels"] = processor.tokenizer(text).input_ids

        return example

    return prepare


def load_train_dataset(config: dict, hf_token: str, world_size: int = 1, rank: int = 0):
    """
    Load training split.
    - If data_dir is set in config: load from local parquet files (fast, no streaming).
    - Otherwise: stream from HuggingFace Hub (slow on bad networks).
    """
    from datasets import load_dataset, Audio
    import glob

    data_dir = config.get("data_dir")
    if data_dir:
        train_files = sorted(glob.glob(os.path.join(data_dir, "train", "*.parquet")))
        if not train_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}/train/")
        logger.info("Loading train from local files: %s", train_files)
        ds = load_dataset("parquet", data_files={"train": train_files}, split="train")
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.shuffle(seed=config["seed"])
        return ds

    # Streaming fallback
    ds = load_dataset(
        config["dataset_name"],
        split="train",
        streaming=True,
        token=hf_token,
    ).cast_column("audio", Audio(decode=False))

    shuffle_buffer = config.get("shuffle_buffer_size", 5000)
    ds = ds.shuffle(seed=config["seed"], buffer_size=shuffle_buffer)

    if world_size > 1:
        from datasets.distributed import split_dataset_by_node
        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

    return ds


def load_eval_dataset(config: dict, hf_token: str, split: str = "validation"):
    """
    Load evaluation split.
    - If data_dir is set: load from local parquet, cap at max_eval/test_samples.
    - Otherwise: stream from Hub.
    """
    from datasets import load_dataset, Audio, Dataset
    import glob

    data_dir = config.get("data_dir")
    n = config["max_eval_samples"] if split == "validation" else config["max_test_samples"]

    if data_dir:
        split_dir = "validation" if split == "validation" else "test"
        files = sorted(glob.glob(os.path.join(data_dir, split_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}/{split_dir}/")
        logger.info("Loading %s from local files: %s", split, files)
        ds = load_dataset("parquet", data_files={split: files}, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        if len(ds) > n:
            ds = ds.select(range(n))
        return ds

    # Streaming fallback
    ds = load_dataset(
        config["dataset_name"],
        split=split,
        streaming=True,
        token=hf_token,
    ).cast_column("audio", Audio(decode=False))

    samples = list(ds.take(n))
    return Dataset.from_list(samples)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    # Always use absolute path for output_dir to avoid CWD ambiguity across DDP ranks
    config["output_dir"] = os.path.abspath(config["output_dir"])
    os.makedirs(config["output_dir"], exist_ok=True)

    # Smoke-test overrides
    if args.smoke_test:
        config["max_steps"] = 100
        config["eval_steps"] = 50
        config["save_steps"] = 50
        config["logging_steps"] = 10
        config["max_eval_samples"] = 32
        config["per_device_train_batch_size"] = 2
        config["gradient_accumulation_steps"] = 1
        config["shuffle_buffer_size"] = 200   # small buffer → fast startup
        logger.warning("⚡ SMOKE TEST MODE — 100 steps, tiny batch")

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ── Distributed context (set by accelerate) ───────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0
    logger.info("Distributed: world_size=%d  rank=%d  local_rank=%d", world_size, rank, local_rank)

    # ── HuggingFace Hub auth ──────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", config.get("hf_token", ""))
    if not hf_token:
        raise ValueError("HF_TOKEN not set. Pass via env or configs/config.yaml.")

    # Ensure HF_TOKEN is set as env var for all HF libraries (datasets, hub, transformers)
    os.environ["HF_TOKEN"] = hf_token

    from huggingface_hub import whoami
    hub_username = whoami(token=hf_token)["name"]
    hub_model_id = config["hub_model_id"]
    if is_main:
        logger.info("HF auth OK — user: %s — pushing to: %s", hub_username, hub_model_id)

    # ── Processor ─────────────────────────────────────────────────────────────
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        config["model_name_or_path"],
        language=config["language"],
        task=config["task"],
    )
    processor.tokenizer.set_prefix_tokens(language=config["language"], task=config["task"])

    # ── Model ─────────────────────────────────────────────────────────────────
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(config["model_name_or_path"])

    # Do not suppress any tokens — we want ALL Arabic diacritics to be generable
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required with gradient_checkpointing=True

    # Set generation config for inference calls inside the trainer
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config["language"], task=config["task"]
    )
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []
    model.generation_config.language = config["language"]
    model.generation_config.task = config["task"]

    # Apply dropout for regularization (reduces overfitting on multi-reciter data)
    dropout = config.get("dropout", 0.0)
    if dropout > 0.0:
        model.config.dropout = dropout
        model.config.attention_dropout = dropout
        model.config.activation_dropout = dropout
        if is_main:
            logger.info("Dropout set to %.2f on encoder/decoder/attention layers", dropout)

    # Enable gradient checkpointing — reduces activation memory ~4×
    model.gradient_checkpointing_enable()
    # Required to compute gradients through checkpointed layers
    model.enable_input_require_grads()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(
            "Model: %s | total params: %s | trainable: %s (%.1f%%)",
            config["model_name_or_path"],
            f"{total_params:,}",
            f"{trainable_params:,}",
            100.0 * trainable_params / total_params,
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    import torch.distributed as dist

    if is_main:
        logger.info("Loading datasets…")

    preprocess = build_preprocessing_fn(processor, config)

    # Training — rank 0 prepares the HF datasets cache first, then all ranks load.
    # Without this barrier, all ranks hit load_dataset() simultaneously and race to
    # write the same cache files → DatasetGenerationError.
    if world_size > 1 and dist.is_initialized():
        if is_main:
            train_ds = load_train_dataset(config, hf_token, world_size=world_size, rank=rank)
        dist.barrier()   # non-rank-0 wait for cache to be ready
        if not is_main:
            train_ds = load_train_dataset(config, hf_token, world_size=world_size, rank=rank)
    else:
        train_ds = load_train_dataset(config, hf_token, world_size=world_size, rank=rank)

    if args.max_train_samples:
        train_ds = train_ds.take(args.max_train_samples)
    train_ds = train_ds.map(
        preprocess,
        remove_columns=["audio", "duration", "reciter", "text"],
    )

    # Validation — same pattern: rank 0 loads + preprocesses first, then barrier.
    if world_size > 1 and dist.is_initialized():
        if is_main:
            eval_ds = load_eval_dataset(config, hf_token, split="validation")
            logger.info("Preprocessing %d eval samples…", len(eval_ds))
            eval_ds = eval_ds.map(
                preprocess,
                remove_columns=["audio", "duration", "reciter", "text"],
                batched=False,
                num_proc=1,
                desc="Preprocessing validation set",
            )
        dist.barrier()   # non-rank-0 wait for cache to be ready
        if not is_main:
            eval_ds = load_eval_dataset(config, hf_token, split="validation")
            eval_ds = eval_ds.map(
                preprocess,
                remove_columns=["audio", "duration", "reciter", "text"],
                batched=False,
                num_proc=1,
                desc="Preprocessing validation set",
            )
    else:
        eval_ds = load_eval_dataset(config, hf_token, split="validation")
        if is_main:
            logger.info("Preprocessing %d eval samples…", len(eval_ds))
        eval_ds = eval_ds.map(
            preprocess,
            remove_columns=["audio", "duration", "reciter", "text"],
            batched=False,
            num_proc=1,
            desc="Preprocessing validation set",
        )
    # Final barrier — all ranks in sync before training starts
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    # ── Data collator ─────────────────────────────────────────────────────────
    from src.training.data_collator import DataCollatorSpeechSeq2SeqWithPadding

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    from src.training.metrics import make_compute_metrics

    compute_metrics = make_compute_metrics(processor)

    # ── Training arguments ────────────────────────────────────────────────────
    from transformers import Seq2SeqTrainingArguments

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],

        # Batch / gradient
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],

        # Optimizer
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        max_grad_norm=config["max_grad_norm"],
        label_smoothing_factor=config["label_smoothing_factor"],

        # Precision
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Evaluation
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        predict_with_generate=True,
        generation_max_length=config["generation_max_length"],
        eval_accumulation_steps=8,   # accumulate eval outputs in small chunks to save VRAM

        # Saving
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        # Logging
        logging_steps=config["logging_steps"],
        logging_dir=os.path.join(config["output_dir"], "tensorboard"),
        report_to=["tensorboard"],

        # HuggingFace Hub
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_token=hf_token,
        hub_strategy="every_save",         # push checkpoint at every save_steps

        # DataLoader
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_prefetch_factor=config["dataloader_prefetch_factor"],
        remove_unused_columns=False,       # we handle columns manually

        # DDP
        ddp_find_unused_parameters=False,

        seed=config["seed"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    from transformers import Seq2SeqTrainer, EarlyStoppingCallback

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,          # full processor = feature extractor + tokenizer
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # ── Log model card to hub before training ────────────────────────────────
    if is_main:
        _push_model_card(trainer, config, hub_model_id, hf_token)

    # ── Train ─────────────────────────────────────────────────────────────────
    if is_main:
        logger.info("Starting training — effective batch size: %d",
                    config["per_device_train_batch_size"]
                    * config["gradient_accumulation_steps"]
                    * world_size)

    checkpoint = args.resume_from_checkpoint
    if checkpoint is None:
        # Auto-detect last checkpoint in output_dir
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(config["output_dir"])
        if last_ckpt:
            checkpoint = last_ckpt
            if is_main:
                logger.info("Resuming from checkpoint: %s", checkpoint)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ── Save final model ──────────────────────────────────────────────────────
    if is_main:
        trainer.save_model()
        processor.save_pretrained(config["output_dir"])

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Training complete. Metrics: %s", metrics)

    # ── Evaluate on test split ────────────────────────────────────────────────
    if is_main:
        logger.info("Evaluating on test split (%d samples)…", config["max_test_samples"])
        test_ds = load_eval_dataset(config, hf_token, split="test")
        test_ds = test_ds.map(
            preprocess,
            remove_columns=["audio", "duration", "reciter", "text"],
            batched=False,
            num_proc=1,
            desc="Preprocessing test set",
        )
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        logger.info("Test metrics: %s", test_metrics)

    # ── Final hub push (model + processor + tensorboard + metrics) ────────────
    if is_main:
        logger.info("Pushing final model + everything to Hub: %s", hub_model_id)
        trainer.push_to_hub(commit_message="Training complete — final model + metrics")

    if is_main:
        logger.info("All done. Model at: https://huggingface.co/%s", hub_model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Model card
# ─────────────────────────────────────────────────────────────────────────────

def _push_model_card(trainer, config: dict, hub_model_id: str, hf_token: str) -> None:
    """Create and push a professional model card to the Hub."""
    from huggingface_hub import ModelCard, ModelCardData

    card_data = ModelCardData(
        language=["ar"],
        license="apache-2.0",
        base_model=config["model_name_or_path"],
        datasets=[config["dataset_name"]],
        tags=["whisper", "automatic-speech-recognition", "arabic", "quran", "tashkeel"],
        metrics=["wer", "cer"],
        model_name=hub_model_id,
    )

    card_content = f"""---
{card_data.to_yaml()}---

# {hub_model_id}

Fine-tuned [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) on
[`tarteel-ai/everyayah`](https://huggingface.co/datasets/tarteel-ai/everyayah) for
**Quranic Arabic Automatic Speech Recognition with full tashkeel (diacritics)**.

## Model Description

| Property | Value |
|---|---|
| Base model | openai/whisper-small (244 M params) |
| Language | Arabic (ar) |
| Task | Automatic Speech Recognition |
| Dataset | tarteel-ai/everyayah |
| Output | Arabic text with **full tashkeel preserved** |
| Fine-tuning type | Full fine-tuning (not LoRA) |
| Precision | fp16 mixed precision |
| Hardware | 4× NVIDIA RTX 2080 Ti (44 GB VRAM total) |

## Why Full Fine-Tuning?

- **Domain gap**: Quranic Tajweed recitation differs substantially from conversational Arabic
- **Tashkeel precision**: All 12 decoder layers need to adapt for reliable diacritic generation
- **5 diverse reciters**: Broad acoustic variety prevents reciter-specific overfitting

## Usage

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="{hub_model_id}",
    generate_kwargs={{"language": "arabic", "task": "transcribe"}},
)

result = pipe("your_quran_audio.mp3")
print(result["text"])
# → بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ
```

## Training Details

| Setting | Value |
|---|---|
| Learning rate | {config['learning_rate']} |
| LR scheduler | cosine |
| Effective batch size | {config['per_device_train_batch_size']} × {config['gradient_accumulation_steps']} × 4 GPUs = {config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] * 4} |
| Max steps | {config['max_steps']} |
| Warmup steps | {config['warmup_steps']} |
| Weight decay | {config['weight_decay']} |
| Dropout | {config.get('dropout', 0.0)} |
| Early stopping | patience=5 (eval every {config['eval_steps']} steps) |
| Best model criterion | CER (Character Error Rate with tashkeel) |

## Evaluation Metrics

| Metric | Description |
|---|---|
| `cer` | Char Error Rate — **with** full tashkeel (primary metric) |
| `wer` | Word Error Rate — **with** full tashkeel |
| `wer_normalized` | Word Error Rate — **without** tashkeel (normalized comparison) |

## Intended Use

Transcribing Quranic recitation audio to text with complete harakat (tashkeel).
Suitable for Quran learning apps, recitation evaluation, and Islamic education tools.

## Training Data

[`tarteel-ai/everyayah`](https://huggingface.co/datasets/tarteel-ai/everyayah) contains
verse-level (ayah-level) recordings from multiple Quranic reciters.
Text labels contain complete tashkeel from the Uthmani script.

Training uses **6 reciters (~24,191 samples total)**:

| Reciter | Samples |
|---|---|
| abdulsamad | ~4,269 |
| abdul_basit | ~4,269 |
| abdullah_basfar | ~4,269 |
| husary | ~4,269 |
| menshawi | ~2,846 |
| minshawi | ~4,269 |
"""

    card = ModelCard(card_content)
    card.push_to_hub(hub_model_id, token=hf_token)
    logger.info("Model card pushed to %s", hub_model_id)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
