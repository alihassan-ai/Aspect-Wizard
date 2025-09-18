import os, glob, math, base64
from typing import List, Tuple

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

def list_images(folder: str) -> List[str]:
    files: List[str] = []
    for e in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    return files

def b64img(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_b64_image(b64s: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = b64s.split(",", 1)[-1]
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(payload))

def ratio_str(w: int, h: int) -> str:
    g = math.gcd(int(w), int(h)) or 1
    return f"{int(w)//g}x{int(h)//g}"
