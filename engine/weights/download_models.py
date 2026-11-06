"""
Script to download open-source Virtual Try-On checkpoints from Hugging Face & GitHub releases.
Run:
    python engine/weights/download_models.py --model leffa
"""
import os
import argparse
from huggingface_hub import snapshot_download

MODEL_REGISTRY = {
    "leffa": {
        "repo_id": "franciszzj/Leffa",
        "description": "Leffa: CVPR 2025 SOTA flow-field attention virtual try-on model"
    },
    "idm_vton": {
        "repo_id": "yisol/IDM-VTON",
        "description": "IDM-VTON: ECCV 2024 high-fidelity dual-path try-on model"
    },
    "cat_vton": {
        "repo_id": "Zheng-Chong/CatVTON",
        "description": "CatVTON: ICLR 2025 lightweight concatenation try-on model (<8GB VRAM)"
    },
    "densepose": {
        "repo_id": "yisol/IDM-VTON",
        "subfolder": "densepose",
        "description": "DensePose 3D human body surface models"
    },
    "sam2": {
        "repo_id": "facebook/sam2-hiera-tiny",
        "description": "Meta SAM 2: Segment Anything Model 2 (Hiera-Tiny for fast segmentation)"
    },
    "florence2": {
        "repo_id": "microsoft/Florence-2-base",
        "description": "Microsoft Florence-2: Open-vocabulary clothing detection"
    }
}

def download_checkpoint(model_key: str, dest_dir: str):
    if model_key not in MODEL_REGISTRY:
        print(f"Unknown model key '{model_key}'. Choose from: {list(MODEL_REGISTRY.keys())}")
        return

    info = MODEL_REGISTRY[model_key]
    print(f"[*] Downloading {info['description']} from Hugging Face Hub ({info['repo_id']})...")
    
    target_path = os.path.join(dest_dir, model_key)
    os.makedirs(target_path, exist_ok=True)
    
    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=target_path,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"[+] Download complete: saved to {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Virtual Try-On Model Checkpoints")
    parser.add_argument("--model", type=str, default="leffa", choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--dest", type=str, default=os.path.dirname(__file__))
    args = parser.parse_args()

    if args.model == "all":
        for k in MODEL_REGISTRY:
            download_checkpoint(k, args.dest)
    else:
        download_checkpoint(args.model, args.dest)

