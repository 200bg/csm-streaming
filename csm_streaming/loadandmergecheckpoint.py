import os
import re
import torch
from models import Model
from moshi.models import loaders
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from lora import (
    remove_lora_modules,
    merge_lora_weights,
    strip_bias_keys,
    DEVICE,
    OUTPUT_DIR,
)
MODEL_NAME = "sesame/csm-1b"
R=32
APLHA=64

def find_latest_checkpoint(dir_path):
    checkpoints = [
        (int(re.search(r"checkpoint-epoch-(\d+)", d).group(1)), os.path.join(dir_path, d))
        for d in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, d)) and "checkpoint-epoch" in d
    ]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found.")
    latest_epoch, latest_path = max(checkpoints, key=lambda x: x[0])
    print(f"Latest checkpoint: epoch {latest_epoch} -> {latest_path}")
    return latest_path

def load_checkpoint_and_merge():
    print("Loading model...")
    model = Model.from_pretrained(MODEL_NAME).to(DEVICE)

    print("Applying LoRA structure...")
    from lora import replace_linear_with_lora
    model = replace_linear_with_lora(model, r=R, alpha=APLHA, dropout=0.0)

    checkpoint_path = find_latest_checkpoint(OUTPUT_DIR)
    state = torch.load(os.path.join(checkpoint_path, "model.safetensors"), map_location=DEVICE)

    model.load_state_dict(state["model_state_dict"], strict=False)

    print("Merging LoRA weights into base model...")
    merge_lora_weights(model)

    print("Replacing LoRALinear modules with nn.Linear...")
    model = remove_lora_modules(model)

    print("Stripping bias keys...")
    merged_state = strip_bias_keys(model.state_dict())

    final_path = os.path.join(OUTPUT_DIR, "model_clean.safetensors")
    save_file(merged_state, final_path)
    print(f"Merged and cleaned model saved to: {final_path}")

if __name__ == "__main__":
    load_checkpoint_and_merge()
