# services/forge.py
# Minimal Forge/A1111-compatible client.
# Assumes the UI builds the full prompt (including any <lora:NAME:WEIGHT> tags).

from dataclasses import dataclass
from typing import Optional, List, Dict
import requests
from PIL import Image
from utils.images import b64img


@dataclass
class ForgeSettings:
    # Core generation params (UI fills these; prompt may include LoRA tags)
    model_checkpoint: str = ""          # e.g. "flux1-dev.safetensors"
    prompt: str = ""                    # may already contain <lora:NAME:WEIGHT>
    negative_prompt: str = ""
    sampler_name: str = "DPM2"
    schedule: Optional[str] = "Simple"  # sent lower-case to API
    steps: int = 30
    width: Optional[int] = 1216
    height: Optional[int] = 1664
    cfg_scale: float = 1.0
    distilled_cfg_scale: float = 0.0
    denoising_strength: float = 0.45    # used by img2img/inpaint
    batch_size: int = 1
    batch_count: int = 1                # n_iter
    seed: int = -1
    inpaint_full_res: bool = False
    inpaint_full_res_padding: int = 32

    # Free-form passthrough to the payload if you need extras later
    extra: Optional[Dict] = None


class ForgeClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    # ---------------- Health / model selection ----------------

    def check_api(self) -> None:
        r = requests.get(self.base, timeout=5)
        r.raise_for_status()

    def ensure_model(self, name: str) -> None:
        """Switch the active checkpoint if present; otherwise keep current."""
        if not name:
            return
        r = requests.get(f"{self.base}/sdapi/v1/sd-models", timeout=15)
        r.raise_for_status()
        models = r.json()
        names = [m.get("title") or m.get("model_name") or m.get("filename") for m in models]
        if any(name in str(x) for x in names):
            requests.post(
                f"{self.base}/sdapi/v1/options",
                json={"sd_model_checkpoint": name},
                timeout=30
            )

    # ---------------- Payload helpers ----------------

    def _payload_common(self, s: ForgeSettings, W: int, H: int) -> Dict:
        payload: Dict = {
            "prompt": s.prompt,                         # already includes any LoRA tags from UI
            "negative_prompt": s.negative_prompt,
            "sampler_name": s.sampler_name,
            "steps": int(s.steps),
            "width": int(W),
            "height": int(H),
            "cfg_scale": float(s.cfg_scale),
            "distilled_cfg_scale": float(s.distilled_cfg_scale),
            "denoising_strength": float(s.denoising_strength),
            "seed": int(s.seed),
            "batch_size": int(s.batch_size),
            "n_iter": int(s.batch_count),
        }
        if s.schedule:
            payload["scheduler"] = str(s.schedule).lower()
        if s.extra:
            payload.update(s.extra)
        return payload

    # ---------------- Operations ----------------

    def img2img_single(self, src_path: str, s: ForgeSettings) -> List[str]:
        """Run img2img on a single image and return base64 images from Forge."""
        with Image.open(src_path) as im:
            iw, ih = im.size
        W, H = s.width or iw, s.height or ih
        payload = self._payload_common(s, W, H)
        payload["init_images"] = [b64img(src_path)]
        #print("[forge] payload prompt:", payload["prompt"])


        r = requests.post(f"{self.base}/sdapi/v1/img2img", json=payload, timeout=600)
        r.raise_for_status()
        return r.json().get("images", [])

    def inpaint_single(self, src_path: str, mask_path: str, s: ForgeSettings) -> List[str]:
        """Run inpaint on a single (image, mask) pair and return base64 images."""
        with Image.open(src_path) as im:
            iw, ih = im.size
        W, H = s.width or iw, s.height or ih
        payload = self._payload_common(s, W, H)
        payload.update({
            "init_images": [b64img(src_path)],
            "mask": b64img(mask_path),
            "inpainting_fill": 1,               # keep masked content (adjust as needed)
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32,
        })

        r = requests.post(f"{self.base}/sdapi/v1/img2img", json=payload, timeout=600)
        r.raise_for_status()
        return r.json().get("images", [])
