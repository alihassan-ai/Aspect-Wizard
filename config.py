# config.py
import os
import json

CONFIG_JSON_PATH = os.path.join(os.getcwd(), "config_runtime.json")

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
    data = _defaults.copy()
    if os.path.exists(CONFIG_JSON_PATH):
        try:
            with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
                data.update(json.load(f))
        except Exception as e:
            print("⚠️ Failed to load runtime config:", e)
    return data


def save_config(data: dict):
    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return CONFIG_JSON_PATH


# -------------------------------
# Dynamic path object
# -------------------------------
class DynamicPath:
    def __init__(self, key):
        self.key = key

    def __str__(self):
        return load_config()[self.key]

    def __fspath__(self):  # so os.path works
        return str(self)

    def __repr__(self):
        return f"<DynamicPath {self.key}={str(self)}>"

# -------------------------------
# Module-level dynamic variables
# -------------------------------
PROJECTS_ROOT        = DynamicPath("PROJECTS_ROOT")
DEFAULT_INPUT_DIR    = DynamicPath("DEFAULT_INPUT_DIR")
CLASSIFY_OUT_ROOT    = DynamicPath("CLASSIFY_OUT_ROOT")
SCALE_OUT_ROOT       = DynamicPath("SCALE_OUT_ROOT")
FORGE_IMG2IMG_OUT    = DynamicPath("FORGE_IMG2IMG_OUT")
FORGE_INPAINT_OUT    = DynamicPath("FORGE_INPAINT_OUT")
FINAL_MASK_DIR       = DynamicPath("FINAL_MASK_DIR")
MODEL_DIR_SD         = DynamicPath("MODEL_DIR_SD")
MODEL_DIR_LORA       = DynamicPath("MODEL_DIR_LORA")
COMFY_MASK_WORKFLOW  = DynamicPath("COMFY_MASK_WORKFLOW")
COMFY_OUTPUT_MASK_DIR= DynamicPath("COMFY_OUTPUT_MASK_DIR")
FORGE_BASE           = DynamicPath("FORGE_BASE")
COMFY_BASE           = DynamicPath("COMFY_BASE")
CLIENT_ID            = DynamicPath("CLIENT_ID")

# -------------------------------
# Ensure directories exist
# -------------------------------
os.makedirs(str(PROJECTS_ROOT), exist_ok=True)
for d in [CLASSIFY_OUT_ROOT, SCALE_OUT_ROOT, FORGE_IMG2IMG_OUT, FORGE_INPAINT_OUT, FINAL_MASK_DIR]:
    os.makedirs(str(d), exist_ok=True)
