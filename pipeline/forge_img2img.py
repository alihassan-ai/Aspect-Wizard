# pipeline/forge_img2img.py
import os, pathlib, gc
from typing import List, Tuple, Generator
from services.forge import ForgeClient, ForgeSettings
from utils.images import list_images, save_b64_image
from config import FORGE_BASE, FORGE_IMG2IMG_OUT

# Optional torch import for VRAM cleanup and reporting
try:
    import torch
except ImportError:
    torch = None


def free_vram(tag: str = ""):
    """
    Force release of GPU memory (if torch is available) and log status.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        free_mb = torch.cuda.mem_get_info()[0] / (1024**2)  # free memory in MB
        total_mb = torch.cuda.mem_get_info()[1] / (1024**2) # total memory in MB
        print(f"[VRAM] Freed at {tag} → {free_mb:.0f} MB free / {total_mb:.0f} MB total")
    else:
        print(f"[VRAM] Freed at {tag} (torch not available)")


def run_forge_img2img(folder_path: str, s: ForgeSettings) -> Generator[Tuple[str, List[str], str], None, None]:
    """Generator that yields (out_dir, saved_paths, status) as images are processed."""
    client = ForgeClient(FORGE_BASE)  # ✅ client defined here
    client.check_api()
    client.ensure_model(s.model_checkpoint)

    os.makedirs(FORGE_IMG2IMG_OUT, exist_ok=True)
    saved: List[str] = []

    all_imgs = list_images(folder_path)
    total = len(all_imgs)

    for idx, p in enumerate(all_imgs, start=1):
        try:
            outs = client.img2img_single(p, s)   # ✅ client is in scope
            stem = pathlib.Path(p).stem
            for i, b64 in enumerate(outs or [], 1):
                outp = os.path.join(FORGE_IMG2IMG_OUT, f"{stem}_img2img_{i}.png")
                save_b64_image(b64, outp)
                saved.append(outp)
        except Exception as e:
            print("img2img error:", p, e)

        # 🔧 Free VRAM after each image
        free_vram(tag=f"after {os.path.basename(p)}")

        # yield progress after each image
        yield FORGE_IMG2IMG_OUT, saved.copy(), f"Processed {idx}/{total}"

    # 🔧 Final cleanup after all images
    free_vram(tag="end of img2img batch")


def run_forge_img2img_single(img: str, s: ForgeSettings):
    """Run Img2Img for one image only (no folder loop)."""
    client = ForgeClient(FORGE_BASE)
    client.check_api()
    client.ensure_model(s.model_checkpoint)

    os.makedirs(FORGE_IMG2IMG_OUT, exist_ok=True)
    saved = []

    try:
        outs = client.img2img_single(img, s)
        stem = pathlib.Path(img).stem
        for i, b64 in enumerate(outs or [], 1):
            outp = os.path.join(FORGE_IMG2IMG_OUT, f"{stem}_img2img_{i}.png")
            save_b64_image(b64, outp)
            saved.append(outp)
        yield FORGE_IMG2IMG_OUT, saved, f"✅ Processed {os.path.basename(img)}"
    except Exception as e:
        yield FORGE_IMG2IMG_OUT, saved, f"❌ Error on {os.path.basename(img)}: {e}"
    finally:
        # 🔧 Always free VRAM after single run
        free_vram(tag=f"after {os.path.basename(img)}")
