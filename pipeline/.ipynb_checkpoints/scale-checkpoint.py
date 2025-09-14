import os, pathlib
from typing import List, Tuple
from PIL import Image
from utils.images import list_images
from config import SCALE_OUT_ROOT
FLUX_RESOLUTIONS = {
    "2MP": {
        "1:1": (1408, 1408),
        "3:2": (1728, 1152),
        "4:3": (1664, 1216),
        "16:9": (1920, 1088),
        "21:9": (2176, 960),
    },
    "1MP": {
        "1:1": (1024, 1024),
        "3:2": (1216, 832),
        "4:3": (1152, 896),
        "16:9": (1344, 768),
        "21:9": (1536, 640),
    },
}


def find_flux_size(mode: str, target_px: int) -> Tuple[int, int]:
    """
    Given mode (Long side/Width/Height) and requested dimension,
    return the valid Flux (w,h) or raise Error.
    """
    for group, entries in FLUX_RESOLUTIONS.items():
        for ratio, (w, h) in entries.items():
            if mode == "Long side" and max(w, h) == target_px:
                return (w, h)
            if mode == "Width" and w == target_px:
                return (w, h)
            if mode == "Height" and h == target_px:
                return (w, h)
    raise ValueError(f"Target {target_px}px with {mode} not in Flux chart")

def scale_folder_by(folder_path: str, mode: str, target_px: int) -> Tuple[str, List[str]]:
    os.makedirs(SCALE_OUT_ROOT, exist_ok=True)
    name = pathlib.Path(folder_path).name
    out_w, out_h = find_flux_size(mode, target_px)
    out_dir = os.path.join(SCALE_OUT_ROOT, f"{name}_{out_w}x{out_h}")
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
