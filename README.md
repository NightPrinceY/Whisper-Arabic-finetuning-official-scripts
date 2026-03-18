# Whisper Quran Fine-Tuning

Full fine-tuning pipeline for [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) on Quranic Arabic recitation with complete tashkeel (diacritics) preservation.

## Links

| Resource | URL |
|---|---|
| Fine-tuned model on HuggingFace | [NightPrince/stt-arabic-whisper-finetuned-diactires](https://huggingface.co/NightPrince/stt-arabic-whisper-finetuned-diactires) |
| Model inference repository | [NightPrinceY/stt-arabic-whisper-finetuned-diactires](https://github.com/NightPrinceY/stt-arabic-whisper-finetuned-diactires) |
| Training dataset | [tarteel-ai/everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah) |
| Author portfolio | [yahya-portfoli-app.netlify.app](https://yahya-portfoli-app.netlify.app/) |

## Results

| Metric | Baseline whisper-small | This model (step 1500) |
|---|---|---|
| CER (with tashkeel) | 61.97% | **0.6922%** |
| WER (with tashkeel) | 102.22% | **3.2801%** |
| WER (normalized) | — | **3.0519%** |

**Generalization to unseen reciters** (voices never present in training):
- CER: 3.9918% | WER: 15.9292%

## What This Repository Contains

```
whisper-quran-finetune/
├── src/
│   ├── training/
│   │   ├── train.py              # Main training script (Seq2SeqTrainer + DDP)
│   │   ├── data_collator.py      # Custom collator for Whisper seq2seq batching
│   │   └── metrics.py            # CER / WER / WER-normalized computation
│   └── data/
│       └── audio_utils.py        # Audio loading via soundfile (no torchcodec)
├── scripts/
│   ├── download_dataset.sh       # Download specific everyayah shards from HF Hub
│   ├── prefilter_dataset.py      # Pre-filter all splits (removes >30s audio, >448 tokens)
│   ├── run_training.sh           # Launch multi-GPU training via Accelerate
│   ├── run_smoke_test.sh         # Quick 100-step smoke test
│   └── setup_accelerate.sh       # Configure accelerate for your machine
├── configs/
│   └── config.yaml               # All hyperparameters in one place
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Accelerate for your GPUs

```bash
bash scripts/setup_accelerate.sh
```

### 3. Download the dataset

```bash
export HF_TOKEN=your_token_here
bash scripts/download_dataset.sh
```

This downloads the 6 reciter shards used in training (~47 hours of audio, ~18 GB).

### 4. Pre-filter the dataset

```bash
python scripts/prefilter_dataset.py
```

Removes samples with audio > 30 seconds or text > 448 tokens from **all splits** (train, validation, test). This step is mandatory — skipping it will cause a crash at the first evaluation.

### 5. Launch training

```bash
bash scripts/run_training.sh
```

Training auto-detects the number of available GPUs and launches accordingly.

To run a quick smoke test (100 steps, tiny data) before full training:

```bash
bash scripts/run_smoke_test.sh
```

## Training Configuration

All hyperparameters are in `configs/config.yaml`. Key settings:

| Parameter | Value | Reason |
|---|---|---|
| Learning rate | 1e-5 | Standard for full fine-tuning of Whisper |
| LR scheduler | Cosine | Smoother decay, better generalization |
| Effective batch size | 128 (8 × 4 accum × 4 GPUs) | Stable gradient estimates |
| Warmup steps | 500 | ~3 epochs warmup |
| Weight decay | 0.05 | Regularization across 6 diverse reciters |
| Dropout | 0.1 | Encoder / decoder / attention |
| Eval metric | CER with tashkeel | Primary quality signal |
| Early stopping | patience = 5 | Stops after 2500 steps without improvement |

## Dataset

Training used **19,284 verse-level (ayah-level) recordings** from 6 reciters in the `tarteel-ai/everyayah` dataset:

| Reciter | Samples | Shards |
|---|---|---|
| Abdulsamad | 4,269 | 0, 1, 2 |
| Abdul Basit | 4,269 | 4, 5, 6 |
| Abdullah Basfar | 4,269 | 10, 11, 12 |
| Husary | 4,269 | 55, 56, 57 |
| Menshawi | 2,846 | 71, 72 |
| Minshawi | 4,269 | 74, 75, 76 |

Validation: 500 samples (mixed reciters). Test: 1,000 samples (mixed reciters).

After pre-filtering: 522 validation samples removed (26.7%), 550 test samples removed (30.5%) — all exceeding the 448-token Whisper decoder limit.

## Hardware Requirements

| Spec | Minimum | Used in this project |
|---|---|---|
| GPU VRAM | 11 GB per GPU | 4 × RTX 2080 Ti (11 GB each) |
| GPU count | 1 (slower) | 4 |
| RAM | 32 GB | 220 GB (WSL2) |
| Storage | 25 GB free | ~70 GB (dataset + outputs) |

Training time to best checkpoint (step 1500): approximately **1.5 hours on 4 × RTX 2080 Ti**.

## Training Progression

| Step | Epoch | Train Loss | CER | WER |
|------|-------|------------|-----|-----|
| 50 | 0.3 | 0.7672 | — | — |
| 500 | 3.3 | 0.0149 | 1.3581% | 6.6743% |
| 1000 | 6.6 | 0.0009 | 0.7120% | 3.6794% |
| **1500** | **9.9** | **0.0003** | **0.6922%** | **3.2801%** |
| 2000 | 13.3 | 0.0028 | 0.8241% | 4.1928% |

Overfitting begins at step 2000 — best checkpoint is step 1500.

## Architecture Decisions

**Why full fine-tuning instead of LoRA or adapters:**

1. The Quranic Tajweed acoustic domain differs substantially from the conversational Arabic in Whisper's training data. All 12 encoder and decoder layers need to adapt.
2. Reliable tashkeel generation requires reshaping the full decoder vocabulary distribution, not steering it with adapter weights.
3. With 19,284 training samples, the dataset is large enough to support full fine-tuning without severe overfitting.

**Why no truncation for long ayahs:**

Truncating Quranic text mid-verse corrupts the label and teaches the model wrong outputs. All samples exceeding 448 tokens are filtered out entirely in the pre-filtering step.

## Citation

```bibtex
@misc{elnawasany2026whisperquran,
  author       = {Yahya Mohamed Elnawasany},
  title        = {Whisper Small Fine-Tuned for Quranic Arabic ASR with Full Tashkeel},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/NightPrince/stt-arabic-whisper-finetuned-diactires}},
  note         = {Training code: https://github.com/NightPrinceY/Whisper-Arabic-finetuning-official-scripts}
}
```

**Author**: Yahya Mohamed Elnawasany
**Email**: yahyaalnwsany39@gmail.com
**Portfolio**: https://yahya-portfoli-app.netlify.app/

## License

Apache 2.0 — see [LICENSE](LICENSE).
