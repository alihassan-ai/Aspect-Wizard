# pipeline/comfy_mask.py
import os, time, glob, shutil, pathlib, json, gc
from typing import List, Tuple, Generator, Union
from services.comfy import Comfy
from utils.images import list_images
from config import COMFY_BASE, CLIENT_ID, COMFY_MASK_WORKFLOW, COMFY_OUTPUT_MASK_DIR, FINAL_MASK_DIR

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


def run_comfy_masking_single(
    input_path: Union[str, os.PathLike],
    workflow_json: str = COMFY_MASK_WORKFLOW
) -> Generator[Tuple[str, str, List[str], List[str], str], None, None]:
    """
    Runs ComfyUI masking pipeline.

    Args:
        input_path: Either a folder containing images or a single image file.
        workflow_json: Path to a workflow JSON file.

    Yields:
        (input_dir, FINAL_MASK_DIR, originals, masks_so_far, status_message)
    """
    c = Comfy(COMFY_BASE, CLIENT_ID)
    c.check()

    # Load workflow template
    with open(workflow_json, "r") as f:
        base_wf = json.load(f)

    os.makedirs(COMFY_OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(FINAL_MASK_DIR, exist_ok=True)

    # Determine files to process
    if os.path.isdir(input_path):
        files = list_images(input_path)
        input_dir = input_path
    elif os.path.isfile(input_path):
        files = [str(input_path)]
        input_dir = os.path.dirname(input_path)
    else:
        yield str(input_path), FINAL_MASK_DIR, [], [], f"❌ Invalid input: {input_path}"
        return

    if not files:
        yield input_dir, FINAL_MASK_DIR, [], [], f"❌ No images found in {input_path}"
        return

    moved: List[str] = []
    total = len(files)

    for idx, img in enumerate(files, 1):
        try:
            wf = json.loads(json.dumps(base_wf))  # deep copy for each image
            patched = False

            # Patch workflow with current image
            for _, node in wf.items():
                if isinstance(node, dict) and node.get("class_type") == "Image Load":
                    node.setdefault("inputs", {})["image_path"] = img
                    patched = True
                    break

            if not patched:
                available_classes = [node.get("class_type") for node in wf.values() if isinstance(node, dict)]
                yield input_dir, FINAL_MASK_DIR, files, moved, f"⚠️ Workflow missing 'Image Load'. Found: {available_classes}"
                continue

            # Submit prompt
            out = c.post("/prompt", {"prompt": wf, "client_id": CLIENT_ID})
            pid = out["prompt_id"]

            # Poll until results ready
            while True:
                hist = c.get(f"/history/{pid}")
                if pid in hist and hist[pid].get("outputs"):
                    break
                time.sleep(1.0)

            # Move mask results
            stem = pathlib.Path(img).stem
            produced = sorted(glob.glob(os.path.join(COMFY_OUTPUT_MASK_DIR, f"{stem}*")))
            for src in produced:
                dst = os.path.join(FINAL_MASK_DIR, os.path.basename(src))
                try:
                    shutil.move(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
                moved.append(dst)

            yield (
                input_dir,
                FINAL_MASK_DIR,
                files,
                moved,
                f"✅ Processed {idx}/{total}: {os.path.basename(img)}"
            )

            # 🔧 Free VRAM after each image, log
            free_vram(tag=f"after {os.path.basename(img)}")

        except Exception as e:
            yield input_dir, FINAL_MASK_DIR, files, moved, f"❌ Error on {img}: {e}"

    # 🔧 Final cleanup after all images
    free_vram(tag="end of masking run")

    yield input_dir, FINAL_MASK_DIR, files, moved, f"✅ Masked {img}"
