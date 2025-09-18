import os, shutil, pathlib, re
from typing import Dict, List, Tuple
from PIL import Image
from math import gcd

from utils.images import list_images
from utils.flux_resolutions import ROUNDED_RESOLUTIONS
from config import CLASSIFY_OUT_ROOT


def _ratio_from_size(w: int, h: int) -> str:
    """Return simplified aspect ratio as 'W:H'."""
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def safe_folder_name(ratio: str) -> str:
    """
    Convert ratio strings like '3:2' or '16/9' into OS-safe folder names.
    Works on both Windows and Linux.
    """
    return re.sub(r'[^0-9a-zA-Z]+', 'x', str(ratio)).strip("x")


def classify_images(input_dir: str) -> Tuple[str, Dict[str, str]]:
    """
    Classify images into folders based on aspect ratio.
    - Try exact aspect ratio from resolution.
    - If not exact, round to nearest Flux-supported ratio using ROUNDED_RESOLUTIONS.
    - Folder names are ratio only (e.g. '3:4', '9:16'), sanitized for OS safety.
    """

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(input_dir)

    images = list_images(input_dir)
    if not images:
        raise RuntimeError("No images found")

    # Build a lookup: map resolution -> ratio
    supported_lookup: Dict[Tuple[int, int], str] = {}
    for group in ROUNDED_RESOLUTIONS.values():
        for ratio, (w, h) in group.items():
            supported_lookup[(w, h)] = ratio

    out_map: Dict[str, str] = {}
    unsupported: List[str] = []

    for p in images:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue

        # Step 1: Exact ratio detection
        ratio = _ratio_from_size(w, h)

        # Step 2: If ratio not in Flux, try nearest rounded resolution
        if ratio not in {r for g in ROUNDED_RESOLUTIONS.values() for r in g.keys()}:
            matched_ratio = None
            for (sw, sh), r in supported_lookup.items():
                if abs(w - sw) <= 32 and abs(h - sh) <= 32:  # tolerance
                    matched_ratio = r
                    break
            ratio = matched_ratio if matched_ratio else "Not_supported_with_flux"

        # Step 3: Assign folder with safe name
        safe_ratio = safe_folder_name(ratio)
        out_folder = os.path.join(CLASSIFY_OUT_ROOT, safe_ratio)
        os.makedirs(out_folder, exist_ok=True)
        out_map[ratio] = out_folder

        dst = os.path.join(out_folder, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)

    return CLASSIFY_OUT_ROOT, out_map
