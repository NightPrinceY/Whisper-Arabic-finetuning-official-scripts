#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Download three reciters' shards from tarteel-ai/everyayah to local disk.
#
#  Reciters (6 reciters = ~24,191 training samples ≈ 47 hours):
#    - abdulsamad      shards 0-2   (4,269 samples)
#    - abdul_basit     shards 4-6   (4,269 samples)
#    - abdullah_basfar shards 10-12 (4,269 samples)
#    - husary          shards 55-57 (4,269 samples)
#    - menshawi        shards 71-72 (2,846 samples — only 2 pure shards exist)
#    - minshawi        shards 74-76 (4,269 samples)
#
#  Output: data/everyayah/{train,validation,test}/*.parquet
#
#  Run once:  bash scripts/download_dataset.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source ~/CollegeX/bin/activate
cd "$(dirname "$0")/.."

export HF_TOKEN="${}"
export HF_HUB_ENABLE_HF_TRANSFER=1   # Rust-based fast downloader

echo "Downloading tarteel-ai/everyayah (6 reciters — 17 train + 1 val + 1 test shards)"

python - <<'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

token = os.environ["HF_TOKEN"]

# 5 reciters = ~19,922 training samples
files = {
    "train": [
        # abdulsamad — shards 0-2 (4,269 samples)
        "data/train-00000-of-00132-5f576ecc6b9cc4e3.parquet",
        "data/train-00001-of-00132-1efd575d27eae362.parquet",
        "data/train-00002-of-00132-1fe19dfaeff1e438.parquet",
        # abdul_basit — shards 4-6 (4,269 samples, pure)
        "data/train-00004-of-00132-b12fae9d9ff0e666.parquet",
        "data/train-00005-of-00132-a9ed73fbe9a97f09.parquet",
        "data/train-00006-of-00132-b1c2d74f4ab01280.parquet",
        # abdullah_basfar — shards 10-12 (4,269 samples, pure)
        "data/train-00010-of-00132-67a2ee44f4b0bac2.parquet",
        "data/train-00011-of-00132-fa3800cc7ff1872f.parquet",
        "data/train-00012-of-00132-ff3b3a718fd352fc.parquet",
        # husary — shards 55-57 (4,269 samples, pure)
        "data/train-00055-of-00132-0c89448d76c24507.parquet",
        "data/train-00056-of-00132-e4c40b8684a91681.parquet",
        "data/train-00057-of-00132-da32d74a96d2e081.parquet",
        # menshawi — shards 71-72 (2,846 samples — only 2 pure shards exist)
        "data/train-00071-of-00132-73715944b05aa73c.parquet",
        "data/train-00072-of-00132-026616327775ced6.parquet",
        # minshawi — shards 74-76 (4,269 samples, pure)
        "data/train-00074-of-00132-563389682cdc2d1d.parquet",
        "data/train-00075-of-00132-7620c9855de421de.parquet",
        "data/train-00076-of-00132-2a0ea21635fd6b17.parquet",
    ],
    # Validation shard 0 mixes multiple reciters — gives diverse evaluation
    "validation": [
        "data/validation-00000-of-00012-91e6174ad3aace19.parquet",
    ],
    # Test shard 0 similarly mixed
    "test": [
        "data/test-00000-of-00013-18d88ded65aa7b53.parquet",
    ],
}

local_dir = "data/everyayah"
os.makedirs(local_dir, exist_ok=True)

for split, shards in files.items():
    os.makedirs(f"{local_dir}/{split}", exist_ok=True)
    for shard in shards:
        fname = shard.split("/")[-1]
        dest = f"{local_dir}/{split}/{fname}"
        if os.path.exists(dest):
            print(f"  [SKIP] {dest} already exists")
            continue
        print(f"  Downloading {shard} → {dest}")
        sys.stdout.flush()
        path = hf_hub_download(
            repo_id="tarteel-ai/everyayah",
            filename=shard,
            repo_type="dataset",
            token=token,
            local_dir=f"{local_dir}",   # saves to local_dir/data/<filename>
        )
        # hf_hub_download preserves the repo structure, move to flat location
        import shutil
        shutil.move(path, dest)
        print(f"  Done: {dest}")
        sys.stdout.flush()

print("\nAll shards downloaded.")
PYEOF

echo "Dataset ready in data/everyayah/"
ls -lh data/everyayah/train/ data/everyayah/validation/ data/everyayah/test/ 2>/dev/null
