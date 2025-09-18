import os, pathlib, re
from typing import List, Tuple, Dict
from PIL import Image
from utils.images import list_images, ratio_str
from config import SCALE_OUT_ROOT


def safe_folder_name(ratio: str) -> str:
    """
    Convert ratio strings like '3:2' or '16/9' into OS-safe folder names.
    Works on both Windows and Linux.
    """
    return re.sub(r'[^0-9a-zA-Z]+', 'x', str(ratio)).strip("x")


# Flux standard resolutions (landscape + portrait)
FLUX_RESOLUTIONS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "2MP": {
        "1:1":  (1408, 1408),
        "3:2":  (1728, 1152),
        "2:3":  (1152, 1728),
        "4:3":  (1664, 1216),
        "3:4":  (1216, 1664),
        "16:9": (1920, 1088),
        "9:16": (1088, 1920),
        "21:9": (2176, 960),
        "9:21": (960, 2176),
    },
    "1MP": {
        "1:1":  (1024, 1024),
        "3:2":  (1216, 832),
        "2:3":  (832, 1216),
        "4:3":  (1152, 896),
        "3:4":  (896, 1152),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "21:9": (1536, 640),
        "9:21": (640, 1536),
    },
}


def detect_ratio(w: int, h: int) -> str:
    """Return simplified aspect ratio like '3:4' or '16:9'."""
    from math import gcd
    g = gcd(w, h)
    rw, rh = w // g, h // g
    return f"{rw}:{rh}"


def available_flux_resolutions_for_folder(folder_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Use the folder's aspect ratio (from its name) to find all matching Flux resolutions.
    """
    raw_name = pathlib.Path(folder_path).name  # e.g. "3x4", "16x9" (safe form)
    ratio = raw_name.replace("x", ":")        # convert back for lookup
    matches: Dict[str, Tuple[int, int]] = {}

    for group, entries in FLUX_RESOLUTIONS.items():
        for r, (fw, fh) in entries.items():
            if r == ratio:
                matches[f"{group}_{r} → {fw}x{fh}"] = (fw, fh)

    return matches


def scale_folder_to_flux(folder_path: str, target_label: str) -> Tuple[str, List[str]]:
    os.makedirs(SCALE_OUT_ROOT, exist_ok=True)
    name = pathlib.Path(folder_path).name

    # Parse target_label like "1MP_3:4 → 896x1152"
    mp_part, rest = target_label.split("_", 1)
    ratio = rest.split("→")[0].strip()

    out_w, out_h = None, None
    for group, entries in FLUX_RESOLUTIONS.items():
        for r, (w, h) in entries.items():
            if f"{group}_{r}" == f"{mp_part}_{ratio}":
                out_w, out_h = w, h
                break
        if out_w and out_h:
            break

    if not out_w:
        raise ValueError(f"❌ {target_label} not in Flux chart")

    safe_ratio = safe_folder_name(ratio)
    out_dir = os.path.join(SCALE_OUT_ROOT, f"{name}_{mp_part}_{safe_ratio}_{out_w}x{out_h}")
    os.makedirs(out_dir, exist_ok=True)

    saved: List[str] = []
    for p in list_images(folder_path):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB") if im.mode in ("P", "RGBA", "LA") else im
                if (im.width, im.height) != (out_w, out_h):
                    im = im.resize((out_w, out_h), Image.LANCZOS)
                outp = os.path.join(out_dir, os.path.basename(p))
                im.save(outp)
                saved.append(outp)
        except Exception as e:
            print("scale error:", p, e)

    return out_dir, saved
