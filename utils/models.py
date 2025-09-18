import os
from typing import List

SD_EXTS   = {".safetensors", ".ckpt"}
LORA_EXTS = {".safetensors", ".ckpt"}

def list_files(root: str, allowed_exts) -> List[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in allowed_exts:
            out.append(name)
    return out

def list_sd_models(root: str) -> List[str]:
    return list_files(root, SD_EXTS)

def list_loras(root: str) -> List[str]:
    return list_files(root, LORA_EXTS)
