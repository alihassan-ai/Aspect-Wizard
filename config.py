# config.py
import os, json

CONFIG_JSON_PATH = os.path.join(os.getcwd(), "config_runtime.json")

# Hardcoded defaults for first run
_defaults = {
    "PROJECTS_ROOT": os.path.join(os.getcwd(), "Projects"),
    "DEFAULT_INPUT_DIR": os.path.join(os.getcwd(), "Projects/default/i"),
    "CLASSIFY_OUT_ROOT": os.path.join(os.getcwd(), "Projects/default/aw_out"),
    "SCALE_OUT_ROOT": os.path.join(os.getcwd(), "Projects/default/aw_scaled"),
    "FORGE_IMG2IMG_OUT": os.path.join(os.getcwd(), "Projects/default/forge_batch_img2img"),
    "FORGE_INPAINT_OUT": os.path.join(os.getcwd(), "Projects/default/forge_batch_inpaint"),
    "FINAL_MASK_DIR": os.path.join(os.getcwd(), "Projects/default/Masked"),
    "MODEL_DIR_SD": "/workspace/ForgeUI/models/Stable-diffusion",
    "MODEL_DIR_LORA": "/workspace/ForgeUI/models/Lora",
    "COMFY_MASK_WORKFLOW": "/workspace/ComfyUI/user/default/workflows/mw.json",
    "COMFY_OUTPUT_MASK_DIR": "/workspace/ComfyUI/output/Masked",
    "FORGE_BASE": "http://127.0.0.1:7801",
    "COMFY_BASE": "http://127.0.0.1:7802",
    "CLIENT_ID": "aspect-wizard-ui",
}

def load_config():
    """Reload latest values from JSON every time."""
    data = _defaults.copy()
    if os.path.exists(CONFIG_JSON_PATH):
        try:
            with open(CONFIG_JSON_PATH, "r") as f:
                data.update(json.load(f))
        except Exception as e:
            print("⚠️ Failed to load runtime config:", e)
    return data

# Initial load (for backward compatibility)
_cfg = load_config()

PROJECTS_ROOT       = _cfg["PROJECTS_ROOT"]
DEFAULT_INPUT_DIR   = _cfg["DEFAULT_INPUT_DIR"]
CLASSIFY_OUT_ROOT   = _cfg["CLASSIFY_OUT_ROOT"]
SCALE_OUT_ROOT      = _cfg["SCALE_OUT_ROOT"]
FORGE_IMG2IMG_OUT   = _cfg["FORGE_IMG2IMG_OUT"]
FORGE_INPAINT_OUT   = _cfg["FORGE_INPAINT_OUT"]
FINAL_MASK_DIR      = _cfg["FINAL_MASK_DIR"]

MODEL_DIR_SD        = _cfg["MODEL_DIR_SD"]
MODEL_DIR_LORA      = _cfg["MODEL_DIR_LORA"]

COMFY_MASK_WORKFLOW   = _cfg["COMFY_MASK_WORKFLOW"]
COMFY_OUTPUT_MASK_DIR = _cfg["COMFY_OUTPUT_MASK_DIR"]

FORGE_BASE = _cfg["FORGE_BASE"]
COMFY_BASE = _cfg["COMFY_BASE"]
CLIENT_ID  = _cfg["CLIENT_ID"]

# Ensure directories exist
os.makedirs(PROJECTS_ROOT, exist_ok=True)
for d in [CLASSIFY_OUT_ROOT, SCALE_OUT_ROOT, FORGE_IMG2IMG_OUT, FORGE_INPAINT_OUT, FINAL_MASK_DIR]:
    os.makedirs(d, exist_ok=True)
