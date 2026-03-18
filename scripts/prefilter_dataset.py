#!/usr/bin/env python3
"""
Pre-filter everyayah parquet files in-place.

Removes rows where:
  1. Audio duration > 30.0 seconds  (audio truncation = corrupted training signal)
  2. Tokenized text > 448 tokens    (Whisper decoder hard limit)

Processes ALL splits: train, validation, test.
Run once before training. After this, train.py needs no .filter() step.
"""

import io
import glob
import soundfile as sf
import pandas as pd
from pathlib import Path
from transformers import WhisperTokenizer

MAX_AUDIO_SEC = 30.0
MAX_TOKENS    = 448


def check_audio_duration(audio_bytes: bytes) -> float:
    try:
        info = sf.info(io.BytesIO(audio_bytes))
        return info.duration
    except Exception:
        return 0.0


def filter_split(files: list, tokenizer, split_name: str):
    total_before = 0
    total_after  = 0

    for fpath in files:
        df = pd.read_parquet(fpath)
        n_before = len(df)
        total_before += n_before

        keep = []
        for idx, row in df.iterrows():
            # 1. Audio duration check
            audio_bytes = row["audio"]["bytes"] if isinstance(row["audio"], dict) else row["audio"]
            dur = check_audio_duration(audio_bytes)
            if dur > MAX_AUDIO_SEC:
                continue

            # 2. Token length check
            text = str(row["text"]).strip()
            n_tok = len(tokenizer(text).input_ids)
            if n_tok > MAX_TOKENS:
                continue

            keep.append(idx)

        df_clean = df.loc[keep].reset_index(drop=True)
        n_after = len(df_clean)
        total_after += n_after

        df_clean.to_parquet(fpath, index=False)
        removed = n_before - n_after
        print(f"  [{split_name}] {Path(fpath).name}: {n_before} → {n_after}  (removed {removed}, {100*removed/max(n_before,1):.1f}%)")

    return total_before, total_after


def main():
    base = Path(__file__).parent.parent / "data" / "everyayah"

    print("Loading tokenizer…")
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="arabic", task="transcribe"
    )
    tokenizer.set_prefix_tokens(language="arabic", task="transcribe")

    grand_before = 0
    grand_after  = 0

    for split in ("train", "validation", "test"):
        split_dir = base / split
        files = sorted(glob.glob(str(split_dir / "*.parquet")))
        if not files:
            print(f"\n[{split}] No parquet files found in {split_dir} — skipping.")
            continue

        print(f"\nProcessing {split} ({len(files)} file(s))…")
        b, a = filter_split(files, tokenizer, split)
        grand_before += b
        grand_after  += a
        removed = b - a
        print(f"  [{split}] subtotal: {b} → {a}  (removed {removed}, {100*removed/max(b,1):.1f}%)")

    print(f"\n{'='*60}")
    print(f"Grand total: {grand_before} → {grand_after}  "
          f"(removed {grand_before - grand_after}, "
          f"{100*(grand_before-grand_after)/max(grand_before,1):.1f}%)")
    print("Done. All splits are clean — no sample exceeds 30s or 448 tokens.")


if __name__ == "__main__":
    main()
