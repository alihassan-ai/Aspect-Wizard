import os, shutil, pathlib
from typing import Dict, List, Tuple
from PIL import Image

from utils.images import list_images, ratio_str
from config import CLASSIFY_OUT_ROOT

# utils/flux_resolutions.py
STANDARD_RESOLUTIONS = {
    "2.0MP": {
        "1:1":   (1448, 1448),
        "3:2":   (1773, 1182),
        "4:3":   (1672, 1254),
        "16:9":  (1936, 1089),
        "21:9":  (2212, 948),
    },
    "1.0MP": {
        "1:1":   (1024, 1024),
        "3:2":   (1254, 836),
        "4:3":   (1182, 887),
        "16:9":  (1365, 768),
        "21:9":  (1564, 670),
    }
}

# Round to nearest standard value for classification
ROUNDED_RESOLUTIONS = {
    "2.0MP": {
        "1:1":   (1408, 1408),
        "3:2":   (1728, 1152),
        "4:3":   (1664, 1216),
        "16:9":  (1920, 1088),
        "21:9":  (2176, 960),
    },
    "1.0MP": {
        "1:1":   (1024, 1024),
        "3:2":   (1216, 832),
        "4:3":   (1152, 896),
        "16:9":  (1344, 768),
        "21:9":  (1536, 640),
    }
}

def classify_images(input_dir: str) -> Tuple[str, Dict[str, str]]:
    """
    Classify images in `input_dir` into folders based on supported Flux resolutions (1MP + 2MP).
    Images not matching the chart are placed in 'Not_supported_with_flux'.
    """

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(input_dir)

    images = list_images(input_dir)
    if not images:
        raise RuntimeError("No images found")

    # Build set of all supported (w, h) pairs
    supported = {
        f"{ratio}_{w}x{h}": (w, h)
        for group in ROUNDED_RESOLUTIONS.values()
        for ratio, (w, h) in group.items()
    }

    out_map: Dict[str, str] = {}
    unsupported: List[str] = []

    for p in images:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue

        matched_label = None
        for label, (sw, sh) in supported.items():
            if (w, h) == (sw, sh):
                matched_label = label
                break

        if matched_label:
            out_folder = os.path.join(CLASSIFY_OUT_ROOT, matched_label)
            os.makedirs(out_folder, exist_ok=True)
            out_map[matched_label] = out_folder
            dst = os.path.join(out_folder, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
        else:
            unsupported.append(p)

    # Handle unsupported
    if unsupported:
        unsup_folder = os.path.join(CLASSIFY_OUT_ROOT, "Not_supported_with_flux")
        os.makedirs(unsup_folder, exist_ok=True)
        out_map["Not_supported_with_flux"] = unsup_folder
        for src in unsupported:
            dst = os.path.join(unsup_folder, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    return CLASSIFY_OUT_ROOT, out_map