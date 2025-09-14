# pipeline/forge_inpaint.py
import os, pathlib
from typing import Optional, List, Tuple, Generator
from services.forge import ForgeClient, ForgeSettings
from utils.images import list_images, save_b64_image
from config import FORGE_BASE, FORGE_INPAINT_OUT


def _find_mask(mask_dir: str, stem: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def run_forge_inpaint(orig_dir: str, mask_dir: str, s: ForgeSettings) -> Generator[Tuple[str, List[str], str], None, None]:
    """
    Generator version of inpaint runner.
    Yields (output_dir, all_saved_paths, progress_message).
    """
    client = ForgeClient(FORGE_BASE)
    client.check_api()
    client.ensure_model(s.model_checkpoint)

    os.makedirs(FORGE_INPAINT_OUT, exist_ok=True)
    saved: List[str] = []
    all_images = list_images(orig_dir)
    total = len(all_images)

    for idx, p in enumerate(all_images, 1):
        stem = pathlib.Path(p).stem
        m = _find_mask(mask_dir, stem)
        if not m:
            msg = f"⚠️ mask missing for {stem} ({idx}/{total})"
            print(msg)
            yield FORGE_INPAINT_OUT, saved, msg
            continue

        try:
            outs = client.inpaint_single(p, m, s)
            for i, b64 in enumerate(outs or [], 1):
                outp = os.path.join(FORGE_INPAINT_OUT, f"{stem}_inpaint_{i}.png")
                save_b64_image(b64, outp)
                saved.append(outp)
            msg = f"✅ Processed {idx}/{total}: {os.path.basename(p)}"
            print(msg)
            yield FORGE_INPAINT_OUT, saved, msg
        except Exception as e:
            msg = f"❌ inpaint error: {os.path.basename(p)} ({idx}/{total}) → {e}"
            print(msg)
            yield FORGE_INPAINT_OUT, saved, msg

    yield FORGE_INPAINT_OUT, saved, f"🎉 Done. {len(saved)} results saved."


def run_forge_inpaint_single(img: str, mask_dir: str, s: ForgeSettings):
    """
    Run Inpaint for a single image with its mask only.
    """
    client = ForgeClient(FORGE_BASE)
    client.check_api()
    client.ensure_model(s.model_checkpoint)

    os.makedirs(FORGE_INPAINT_OUT, exist_ok=True)
    saved: List[str] = []

    stem = pathlib.Path(img).stem
    m = _find_mask(mask_dir, stem)
    if not m:
        yield FORGE_INPAINT_OUT, [], f"⚠️ mask missing for {stem}"
        return

    try:
        outs = client.inpaint_single(img, m, s)
        for i, b64 in enumerate(outs or [], 1):
            outp = os.path.join(FORGE_INPAINT_OUT, f"{stem}_inpaint_{i}.png")
            save_b64_image(b64, outp)
            saved.append(outp)
        yield FORGE_INPAINT_OUT, saved, f"✅ Inpainted {os.path.basename(img)}"
    except Exception as e:
        yield FORGE_INPAINT_OUT, [], f"❌ Inpaint error: {os.path.basename(img)} → {e}"

