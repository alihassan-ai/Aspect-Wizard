# pipeline/project_manager.py
import os
import json
import pathlib
import importlib
from typing import Dict

import config  # keep using config.py

# Paths
PROJECTS_ROOT = os.path.join(os.getcwd(), "Projects")
os.makedirs(PROJECTS_ROOT, exist_ok=True)

CONFIG_JSON_PATH = os.path.join(os.getcwd(), "config_runtime.json")


# ───────────────────────────────────────────────────────────────
# Config Helpers
# ───────────────────────────────────────────────────────────────
def _load_runtime_config() -> dict:
    """Load runtime config JSON if it exists."""
    if not os.path.exists(CONFIG_JSON_PATH):
        return {}
    try:
        with open(CONFIG_JSON_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_runtime_config(values: dict) -> None:
    """Save runtime config JSON."""
    with open(CONFIG_JSON_PATH, "w") as f:
        json.dump(values, f, indent=2)


def read_config() -> dict:
    """Read current config values from JSON (runtime)."""
    return _load_runtime_config()


from config import CONFIG_JSON_PATH, load_config

def write_config(values: dict):
    """Write the config_runtime.json file."""
    with open(CONFIG_JSON_PATH, "w") as f:
        json.dump(values, f, indent=2)
    return CONFIG_JSON_PATH


# ───────────────────────────────────────────────────────────────
# Project Creation
# ───────────────────────────────────────────────────────────────
def create_project(project_name: str,
                   project_details: str,
                   source_images: str,
                   model_dir_sd: str,
                   model_dir_lora: str,
                   comfy_workflow: str,
                   comfy_out: str,
                   forge_base: str,
                   comfy_base: str) -> Dict[str, str]:
    """
    Create a new project inside Projects/ and return updated paths.
    """
    if not project_name:
        raise ValueError("Project name is required.")

    proj_dir = os.path.join(PROJECTS_ROOT, project_name)
    os.makedirs(proj_dir, exist_ok=True)

    # Create subfolders for pipeline outputs
    paths = {
        "DEFAULT_INPUT_DIR":   os.path.join(proj_dir, "Input"),
        "CLASSIFY_OUT_ROOT":   os.path.join(proj_dir, "aw_out"),
        "SCALE_OUT_ROOT":      os.path.join(proj_dir, "aw_scaled"),
        "FORGE_IMG2IMG_OUT":   os.path.join(proj_dir, "forge_batch_img2img"),
        "FORGE_INPAINT_OUT":   os.path.join(proj_dir, "forge_batch_inpaint"),
        "FINAL_MASK_DIR":      os.path.join(proj_dir, "Masked"),
    }
    for d in paths.values():
        os.makedirs(d, exist_ok=True)

    # Add model dirs + comfy settings + endpoints
    paths.update({
        "MODEL_DIR_SD": model_dir_sd,
        "MODEL_DIR_LORA": model_dir_lora,
        "COMFY_MASK_WORKFLOW": comfy_workflow,
        "COMFY_OUTPUT_MASK_DIR": comfy_out,
        "FORGE_BASE": forge_base,
        "COMFY_BASE": comfy_base,
    })

    # Save project details note (optional)
    if project_details:
        with open(os.path.join(proj_dir, "project_notes.txt"), "w") as f:
            f.write(project_details)

    # Write runtime config + reload
    write_config(paths)

    return {"project_dir": proj_dir, **paths}
