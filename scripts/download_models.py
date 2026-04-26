"""Download HuggingFace model checkpoints for local LLM inference.

Downloads model weights and tokenizer files to the local ``models/``
directory so that inference can run fully offline.  Supports any
HuggingFace model ID.

Recommended models for this project (4x A6000, 192GB VRAM total):
    - meta-llama/Llama-3.1-8B-Instruct    (~16GB, fits on 1 GPU)
    - Qwen/Qwen2.5-7B-Instruct            (~14GB, fits on 1 GPU)
    - meta-llama/Llama-3.1-70B-Instruct    (~140GB in 4-bit, needs 4 GPUs)
    - Qwen/Qwen2.5-72B-Instruct           (~140GB in 4-bit, needs 4 GPUs)

Usage:
    python scripts/download_models.py meta-llama/Llama-3.1-8B-Instruct
    python scripts/download_models.py Qwen/Qwen2.5-72B-Instruct
    python scripts/download_models.py --list   (show recommended models)

Note:
    Some models (e.g. Llama) require a HuggingFace access token.
    Set the HF_TOKEN environment variable or run ``huggingface-cli login``
    before downloading gated models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

RECOMMENDED_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "size": "~16GB",
        "vram": "~16GB (fp16) / ~8GB (4-bit)",
        "gpus": "1x A6000",
        "notes": "Strong 8B instruct model, good baseline for comparison with Qwen-7B",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "size": "~14GB",
        "vram": "~14GB (fp16) / ~7GB (4-bit)",
        "gpus": "1x A6000",
        "notes": "Updated Qwen with improved instruction following",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "size": "~140GB",
        "vram": "~140GB (fp16) / ~40GB (4-bit)",
        "gpus": "4x A6000 (fp16) or 1x A6000 (4-bit)",
        "notes": "Large model, expected to significantly outperform 7-8B models",
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "size": "~145GB",
        "vram": "~145GB (fp16) / ~42GB (4-bit)",
        "gpus": "4x A6000 (fp16) or 1x A6000 (4-bit)",
        "notes": "Alternative large model, strong multilingual capabilities",
    },
}


def list_models() -> None:
    """Print the table of recommended models."""
    print("\nRecommended models for this project:")
    print("=" * 80)
    for model_id, info in RECOMMENDED_MODELS.items():
        print(f"\n  {model_id}")
        print(f"    Size:  {info['size']}")
        print(f"    VRAM:  {info['vram']}")
        print(f"    GPUs:  {info['gpus']}")
        print(f"    Notes: {info['notes']}")
    print("\n" + "=" * 80)


def download_model(model_id: str) -> Path:
    """Download a model from HuggingFace Hub to the local models directory.

    Returns the local path where the model was saved.
    """
    from huggingface_hub import snapshot_download

    # Create a clean directory name from the model ID.
    local_name = model_id.replace("/", "--")
    local_path = MODELS_DIR / local_name
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} → {local_path}")
    print("This may take a while for large models...")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_path),
        resume_download=True,
    )

    print(f"\nDownload complete: {local_path}")
    return local_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HuggingFace models for local inference")
    parser.add_argument("model_id", nargs="?", help="HuggingFace model ID to download")
    parser.add_argument("--list", action="store_true", help="List recommended models")
    args = parser.parse_args()

    if args.list or not args.model_id:
        list_models()
        if not args.model_id:
            print("\nUsage: python scripts/download_models.py <model_id>")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_model(args.model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
