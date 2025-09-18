# aspect-wizard/ui/ui.py
# Full UI with:
# - Dynamic dropdowns for Base Models (SD) and LoRA files
# - Single-LoRA selection with an adjacent weight input (default 1.0)
# - Static dropdowns for Samplers and Schedulers
# - Same controls and behavior in both Img2Img and Inpaint tabs
#
# NOTE: We append the LoRA tag directly into the prompt as: <lora:NAME:WEIGHT>
#       before sending to Forge (single-LoRA per run as requested).

import os
from dataclasses import asdict
from typing import List
import importlib.util
import torch

import gradio as gr
from PIL import Image

from config import (
    SCALE_OUT_ROOT,
    DEFAULT_INPUT_DIR,
    FORGE_IMG2IMG_OUT,
    FORGE_INPAINT_OUT,
    FINAL_MASK_DIR,
    MODEL_DIR_SD,
    MODEL_DIR_LORA,
)
from utils.images import list_images
from utils.models import list_sd_models, list_loras
from pipeline.classify import classify_images
from pipeline.forge_img2img import run_forge_img2img
from pipeline.forge_inpaint import run_forge_inpaint
from pipeline.comfy_mask import run_comfy_masking_single
from services.forge import ForgeSettings
from pipeline.project_manager import write_config
import importlib.util, shutil, datetime
from pipeline.scale import FLUX_RESOLUTIONS
from pipeline.scale import (
    available_flux_resolutions_for_folder,
    scale_folder_to_flux
)



import json

from config import load_config

cfg = load_config()
TEMPLATES_ROOT = os.path.join(os.getcwd(), "Templates")
os.makedirs(TEMPLATES_ROOT, exist_ok=True)





# ────────────────────────────────────────────────────────────────────────────
# Static choice lists (from Forge UI screenshots)
# ────────────────────────────────────────────────────────────────────────────
SAMPLERS: List[str] = [
    "DPM++ 2M", "DPM++ SDE", "DPM++ 2M SDE", "DPM++ 2M SDE Heun",
    "DPM++ 2S a", "DPM++ 3M SDE",
    "Euler a", "Euler", "LMS", "Heun",
    "DPM2", "DPM2 a", "DPM fast", "DPM adaptive",
    "Restart", "HeunPP2", "IPNDM", "IPNDM_V",
    "DEIS", "DDIM", "DDIM CFG++", "PLMS", "UniPC", "LCM", "DDPM"
]

SCHEDULERS: List[str] = [
    "Automatic", "Uniform", "Karras", "Exponential", "Polyexponential",
    "SGM Uniform", "KL Optimal", "Align Your Steps",
    "Simple", "Normal", "DDIM", "Beta", "Turbo",
    "Align Your Steps GITS", "Align Your Steps 11", "Align Your Steps 32"
]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def gallery_preview(paths, limit=24):
    return [(p, os.path.basename(p)) for p in paths[:limit]]


# ────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    # Scan models once at build; users can refresh in the tab too.
    sd_models_initial = list_sd_models(MODEL_DIR_SD)
    loras_initial = list_loras(MODEL_DIR_LORA)

    with gr.Blocks(title="Aspect Wizard Orchestrator") as demo:
        gr.Markdown(
            "# Aspect Wizard Orchestrator\n"
            "A modular UI combining Classification → Scaling → Forge Img2Img → Comfy Masking → Forge Inpaint."
        )

        # Shared state across tabs
        st_classify_map = gr.State({})
        st_scaled_path  = gr.State("")
        st_img2img_path = gr.State("")
        st_mask_orig    = gr.State("")
        st_mask_dir     = gr.State(FINAL_MASK_DIR)
        st_settings     = gr.State(asdict(ForgeSettings()))


        # ────────────────────────────────────────────────────────────────────
        # Tab 0: System Manager
        # ────────────────────────────────────────────────────────────────────        
        with gr.Tab("0) System Manager"):
            gr.Markdown("### System Manager — Check Dependencies & Services")
        
            sys_output = gr.Textbox(label="System Status", lines=10, interactive=False)
        
            btn_check_deps   = gr.Button("Check Dependencies")
            btn_check_hw     = gr.Button("Check Hardware (GPU)")
            btn_check_forge  = gr.Button("Check Forge WebUI")
            btn_check_comfy  = gr.Button("Check ComfyUI")
            btn_check_all    = gr.Button("Run All Checks")
        
            # Functions reused from old app.py
            import requests, torch, importlib
        
            REQUIRED_PACKAGES = ["torch", "gradio", "PIL", "requests"]
        
            def check_dependencies_ui():
                missing = []
                for pkg in REQUIRED_PACKAGES:
                    try:
                        if pkg == "PIL":
                            importlib.import_module("PIL.Image")
                        else:
                            importlib.import_module(pkg)
                    except ImportError:
                        missing.append(pkg)
                if missing:
                    return f"❌ Missing dependencies: {', '.join(missing)}"
                return "✅ All required dependencies are installed."
        


            def check_hardware_ui():
                # CUDA (NVIDIA)
                if torch.cuda.is_available():
                    return f"✅ NVIDIA GPU detected: {torch.cuda.get_device_name(0)}"

                # ROCm (AMD on Linux)
                if hasattr(torch, "version") and hasattr(torch.version, "hip") and torch.version.hip is not None:
                    if torch.backends.mps.is_available():
                        # macOS MPS (Apple Silicon + AMD fallback)
                        return "✅ GPU detected: Apple Metal (MPS backend)"
                    else:
                        return "✅ AMD GPU detected (ROCm backend)"

                # macOS MPS (Apple Silicon)
                if torch.backends.mps.is_available():
                    return "✅ GPU detected: Apple Metal (MPS backend)"

                return "❌ No supported GPU found (CPU only)."

        
            from config import FORGE_BASE, COMFY_BASE
        
            def check_forge_ui():
                try:
                    r = requests.get(f"{FORGE_BASE}/sdapi/v1/sd-models", timeout=5)
                    r.raise_for_status()
                    return f"✅ Forge WebUI is running at {FORGE_BASE}"
                except Exception as e:
                    return f"❌ Forge WebUI not available at {FORGE_BASE}: {e}"
        
            def check_comfy_ui():
                try:
                    r = requests.get(COMFY_BASE, timeout=5)
                    r.raise_for_status()
                    return f"✅ ComfyUI is running at {COMFY_BASE}"
                except Exception as e:
                    return f"❌ ComfyUI not available at {COMFY_BASE}: {e}"
        
            def run_all_checks():
                return "\n".join([
                    check_dependencies_ui(),
                    check_hardware_ui(),
                    check_forge_ui(),
                    check_comfy_ui()
                ])
        
            # Wire buttons
            btn_check_deps.click(fn=check_dependencies_ui, outputs=sys_output)
            btn_check_hw.click(fn=check_hardware_ui, outputs=sys_output)
            btn_check_forge.click(fn=check_forge_ui, outputs=sys_output)
            btn_check_comfy.click(fn=check_comfy_ui, outputs=sys_output)
            btn_check_all.click(fn=run_all_checks, outputs=sys_output)


        # ────────────────────────────────────────────────────────────────────
        # Tab 1: Project Manager
        # ────────────────────────────────────────────────────────────────────
        import importlib.util, os, shutil, datetime
        from config import PROJECTS_ROOT

        def load_current_config():
            spec = importlib.util.spec_from_file_location("config", "config.py")
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)
            return cfg

        with gr.Tab("1) Project Manager"):
            gr.Markdown("### Create or Configure Project")

            # --- Create New Project Section ---
            gr.Markdown("#### Create a New Project")

            new_proj_name   = gr.Textbox(label="Project Name")
            new_proj_notes  = gr.Textbox(label="Project Notes / Details", lines=3)
            new_proj_images = gr.Textbox(label="Source Images Folder")

            btn_create_proj = gr.Button("📂 Create Project")
            proj_out_dir    = gr.Textbox(label="New Project Path", interactive=False)
            proj_status     = gr.Textbox(label="Status", interactive=False)

            # --- Edit Existing Config Section (pre-load) ---
            cfg = load_current_config()

            inp_dir     = gr.Textbox(value=cfg.DEFAULT_INPUT_DIR, label="Default Input Dir")
            classify    = gr.Textbox(value=cfg.CLASSIFY_OUT_ROOT, label="Classification Output")
            scale       = gr.Textbox(value=cfg.SCALE_OUT_ROOT, label="Scale Output")
            forge_i2i   = gr.Textbox(value=cfg.FORGE_IMG2IMG_OUT, label="Forge Img2Img Output")
            forge_inp   = gr.Textbox(value=cfg.FORGE_INPAINT_OUT, label="Forge Inpaint Output")
            model_sd    = gr.Textbox(value=cfg.MODEL_DIR_SD, label="Stable Diffusion Models")
            model_lora  = gr.Textbox(value=cfg.MODEL_DIR_LORA, label="LoRA Models")
            comfy_wf    = gr.Textbox(value=cfg.COMFY_MASK_WORKFLOW, label="Comfy Workflow JSON")
            comfy_out   = gr.Textbox(value=cfg.COMFY_OUTPUT_MASK_DIR, label="Comfy Mask Output")
            final_mask  = gr.Textbox(value=cfg.FINAL_MASK_DIR, label="Final Mask Output")
            forge_base  = gr.Textbox(value=cfg.FORGE_BASE, label="Forge Base URL")
            comfy_base  = gr.Textbox(value=cfg.COMFY_BASE, label="Comfy Base URL")

            save_btn = gr.Button("💾 Save Config")
            out_msg  = gr.Textbox(label="Status", interactive=False)

            # --- Logic: Create project + update config + refresh fields ---
            def create_project_fn(name, notes, src_folder):
                if not name:
                    return "", "❌ Project name required", *[gr.update() for _ in range(12)]

                proj_dir = os.path.join(PROJECTS_ROOT, name)
                if os.path.exists(proj_dir):
                    return proj_dir, f"⚠️ Project {name} already exists", *[gr.update() for _ in range(12)]

                os.makedirs(proj_dir, exist_ok=True)

                # Subfolders
                input_dir    = os.path.join(proj_dir, "i")
                classify_dir = os.path.join(proj_dir, "aw_out")
                scale_dir    = os.path.join(proj_dir, "aw_scaled")
                i2i_dir      = os.path.join(proj_dir, "forge_batch_img2img")
                inpaint_dir  = os.path.join(proj_dir, "forge_batch_inpaint")
                masks_dir    = os.path.join(proj_dir, "Masked")

                for d in [input_dir, classify_dir, scale_dir, i2i_dir, inpaint_dir, masks_dir]:
                    os.makedirs(d, exist_ok=True)

                # Copy source images if provided
                if src_folder and os.path.isdir(src_folder):
                    for f in os.listdir(src_folder):
                        src = os.path.join(src_folder, f)
                        if os.path.isfile(src):
                            shutil.copy2(src, os.path.join(input_dir, os.path.basename(src)))

                # Save notes
                if notes:
                    with open(os.path.join(proj_dir, "project_info.txt"), "w") as f:
                        f.write(f"Project: {name}\n")
                        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
                        f.write(f"Notes:\n{notes}\n")

                # Build values to update config
                data = {
                    "DEFAULT_INPUT_DIR": input_dir,
                    "CLASSIFY_OUT_ROOT": classify_dir,
                    "SCALE_OUT_ROOT": scale_dir,
                    "FORGE_IMG2IMG_OUT": i2i_dir,
                    "FORGE_INPAINT_OUT": inpaint_dir,
                    "MODEL_DIR_SD": model_sd.value,
                    "MODEL_DIR_LORA": model_lora.value,
                    "COMFY_MASK_WORKFLOW": comfy_wf.value,
                    "COMFY_OUTPUT_MASK_DIR": comfy_out.value,
                    "FINAL_MASK_DIR": masks_dir,
                    "FORGE_BASE": forge_base.value,
                    "COMFY_BASE": comfy_base.value,
                }
                path = write_config(data)

                # Return new values for all fields so UI refreshes
                return (
                    proj_dir,
                    f"✅ Project created at {proj_dir}. Config updated: {path}",
                    data["DEFAULT_INPUT_DIR"], data["CLASSIFY_OUT_ROOT"], data["SCALE_OUT_ROOT"],
                    data["FORGE_IMG2IMG_OUT"], data["FORGE_INPAINT_OUT"],
                    data["MODEL_DIR_SD"], data["MODEL_DIR_LORA"],
                    data["COMFY_MASK_WORKFLOW"], data["COMFY_OUTPUT_MASK_DIR"], data["FINAL_MASK_DIR"],
                    data["FORGE_BASE"], data["COMFY_BASE"],
                )

            btn_create_proj.click(
                create_project_fn,
                [new_proj_name, new_proj_notes, new_proj_images],
                [proj_out_dir, proj_status,
                 inp_dir, classify, scale, forge_i2i, forge_inp,
                 model_sd, model_lora, comfy_wf, comfy_out, final_mask,
                 forge_base, comfy_base],
            )

            # --- Logic: Save edited config ---
            def save_config_fn(*vals):
                keys = [
                    "DEFAULT_INPUT_DIR","CLASSIFY_OUT_ROOT","SCALE_OUT_ROOT","FORGE_IMG2IMG_OUT","FORGE_INPAINT_OUT",
                    "MODEL_DIR_SD","MODEL_DIR_LORA","COMFY_MASK_WORKFLOW","COMFY_OUTPUT_MASK_DIR","FINAL_MASK_DIR",
                    "FORGE_BASE","COMFY_BASE"
                ]
                data = dict(zip(keys, vals))
                path = write_config(data)
                return f"✅ Config updated at {path}"

            save_btn.click(
                save_config_fn,
                [inp_dir, classify, scale, forge_i2i, forge_inp,
                 model_sd, model_lora, comfy_wf, comfy_out, final_mask,
                 forge_base, comfy_base],
                out_msg
            )

                
        # ────────────────────────────────────────────────────────────────────
        # Tab 2: Classification
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("2) Classification"):
            in_dir = gr.Textbox(value=DEFAULT_INPUT_DIR, label="Input folder")
            btn_classify = gr.Button("Run Classification")
            out_root = gr.Textbox(label="Output root", interactive=False)
            ratio_json = gr.JSON(label="Aspect folders {ratio: path}")
        
            # Shared state across tabs
            st_classify_map = gr.State({})
        
            # up to 6 groups: title directly above its gallery
            titles, galleries = [], []
            for _ in range(6):
                with gr.Column():
                    t = gr.Markdown(visible=False)
                    g = gr.Gallery(columns=6, height=220, visible=False, show_label=False, label=None)
                titles.append(t)
                galleries.append(g)
        
            def on_classify(inp):
                root, mapping = classify_images(inp)
                groups = []
                for ratio, folder in mapping.items():
                    paths = list_images(folder)
                    if not paths:
                        continue
                    res = set()
                    for p in paths[:64]:
                        try:
                            with Image.open(p) as im:
                                res.add(f"{im.width}×{im.height}")
                        except Exception:
                            pass
                    groups.append((ratio, paths, sorted(res)))
                groups.sort(key=lambda x: len(x[1]), reverse=True)
        
                t_updates, g_updates = [], []
                for i in range(6):
                    if i < len(groups):
                        ratio, paths, res_list = groups[i]
                        md = f"**{ratio}**\n{', '.join(res_list) if res_list else '—'}"
                        t_updates.append(gr.update(value=md, visible=True))
                        g_updates.append(gr.update(value=gallery_preview(paths), visible=True))
                    else:
                        t_updates.append(gr.update(visible=False))
                        g_updates.append(gr.update(visible=False))
        
                return (root, mapping, *t_updates, *g_updates)
        
            btn_classify.click(on_classify, [in_dir], [out_root, ratio_json, *titles, *galleries])
            ratio_json.change(lambda m: m, [ratio_json], [st_classify_map])
        
        
        # ────────────────────────────────────────────────────────────────────
        # Tab 3: Resolution Scaling (Flux Standard Only)
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("3) Resolution Scaling"):
            with gr.Row():
                src_folder = gr.Dropdown(label="Choose a classified folder", choices=[])
                load_btn   = gr.Button("Load folders from Tab 2")
        
            # Dynamically populated Flux resolution choices
            target_res = gr.Dropdown(
                label="Available Flux Resolutions",
                choices=[],
                value=None
            )
        
            run_btn = gr.Button("Run Scaling")
            scaled_path = gr.Textbox(label="Scaled folder", interactive=False)
            gal2 = gr.Gallery(label="Scaled preview", columns=6, height=280)
        
            # Load choices from Tab 2 classification
            load_btn.click(
                lambda m: gr.update(choices=list(m.values()) if isinstance(m, dict) else []),
                [st_classify_map],
                [src_folder]
            )
        
            # When folder is selected, update dropdown with valid resolutions
            def update_res_choices(folder):
                if not folder:
                    return gr.update(choices=[], value=None)
                try:
                    opts = available_flux_resolutions_for_folder(folder)
                    return gr.update(choices=list(opts.keys()), value=(list(opts.keys())[0] if opts else None))
                except Exception as e:
                    print("error in update_res_choices:", e)
                    return gr.update(choices=[], value=None)


        
            src_folder.change(
                fn=update_res_choices,
                inputs=[src_folder],
                outputs=[target_res]
            )
        
            # Scaling runner
            def on_scale(folder, target_label):
                if not folder:
                    raise gr.Error("Pick a folder first.")
                if not target_label:
                    raise gr.Error("Pick a target resolution.")
                try:
                    out_dir, saved = scale_folder_to_flux(folder, target_label)
                except ValueError as e:
                    raise gr.Error(str(e))
                return out_dir, gallery_preview(saved), out_dir
        
            run_btn.click(
                on_scale,
                [src_folder, target_res],
                [scaled_path, gal2, st_scaled_path]
            )


        # ────────────────────────────────────────────────────────────────────
        # Model list refresher (shared)
        # ────────────────────────────────────────────────────────────────────
        def _scan_models():
            sd = list_sd_models(MODEL_DIR_SD)
            lr = list_loras(MODEL_DIR_LORA)
            # return two updates for dropdowns (caller wires outputs)
            return (
                gr.update(choices=sd, value=(sd[0] if sd else None)),
                gr.update(choices=lr, value=(lr[0] if lr else None)),
            )

        # ────────────────────────────────────────────────────────────────────
        # Tab 3: Forge Img2Img
        # ────────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────
        # Tab 3: Forge Img2Img (with Prompt + LoRA tag injection)
        # Requires in scope:
        #   - SAMPLERS, SCHEDULERS
        #   - list_sd_models, list_loras
        #   - MODEL_DIR_SD, MODEL_DIR_LORA
        #   - DEFAULT_INPUT_DIR, st_scaled_path, gallery_preview
        #   - run_forge_img2img, ForgeSettings
        # ────────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────
        # Tab 4: Forge Img2Img (streaming generator)
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("4) Forge Img2Img"):
            # Initial scan
            sd_models_initial = list_sd_models(MODEL_DIR_SD)
            loras_initial     = list_loras(MODEL_DIR_LORA)
        
            with gr.Row():
                forge_input = gr.Textbox(
                    value="",                      # leave empty by default
                    placeholder=SCALE_OUT_ROOT,    # hint
                    label="Images folder (leave blank to use Tab 3 scaled)"
                )
                use_scaled  = gr.Button("Use Scaled From Tab 3")
        
            fs_prompt_i2i = gr.Textbox(
                label="Prompt",
                placeholder="Describe your image... (LoRA tag will be appended automatically if selected)",
                lines=3,
            )
        
            with gr.Accordion("Settings", open=False):
                dd_model_i2i = gr.Dropdown(
                    label="Base model (Stable-diffusion)",
                    choices=sd_models_initial,
                    value=(sd_models_initial[0] if sd_models_initial else None),
                    interactive=True,
                    allow_custom_value=False,
                )
        
                with gr.Row():
                    dd_lora_i2i = gr.Dropdown(
                        label="LoRA (single select)",
                        choices=loras_initial,
                        value=(loras_initial[0] if loras_initial else None),
                        interactive=True,
                        multiselect=False,
                        allow_custom_value=False,
                    )
                    lora_strength_i2i = gr.Number(value=1.0, precision=2, label="LoRA weight")
                    btn_refresh_models_i2i = gr.Button("↻ Refresh model/LoRA lists")
        
                with gr.Row():
                    dd_sampler_i2i   = gr.Dropdown(label="Sampler",   choices=SAMPLERS,  value="DPM2",    interactive=True)
                    dd_scheduler_i2i = gr.Dropdown(label="Schedule",  choices=SCHEDULERS, value="Simple", interactive=True)
        
                with gr.Row():
                    fs_steps_i2i = gr.Number(value=30, label="Steps")
                    fs_cfg_i2i   = gr.Number(value=1.0, label="CFG Scale")
        
                with gr.Row():
                    fs_width_i2i  = gr.Number(value=1216, label="Width (None = auto)")
                    fs_height_i2i = gr.Number(value=1664, label="Height (None = auto)")
        
                with gr.Row():
                    fs_denoise_i2i = gr.Number(value=0.45, label="Denoising strength")
                    fs_bsize_i2i   = gr.Number(value=1,    label="Batch size")
                    fs_bcnt_i2i    = gr.Number(value=1,    label="Batch count (n_iter)")
        
                fs_seed_i2i = gr.Number(value=-1, label="Seed (-1=random)")
        
            run_i2i   = gr.Button("Run Forge Img2Img")
            out_i2i   = gr.Textbox(label="Img2Img output folder", interactive=False)
            gal3      = gr.Gallery(label="Img2Img preview", columns=6, height=280)
            status_i2i = gr.Textbox(label="Progress", interactive=False)
        
            # Use scaled folder from Tab 3
            use_scaled.click(lambda p: gr.update(value=p or ""), inputs=[st_scaled_path], outputs=[forge_input])
        
            # Refresh model/LoRA lists
            def _scan_models():
                sd = list_sd_models(MODEL_DIR_SD)
                lr = list_loras(MODEL_DIR_LORA)
                return (
                    gr.update(choices=sd, value=(sd[0] if sd else None)),
                    gr.update(choices=lr, value=(lr[0] if lr else None)),
                )
            btn_refresh_models_i2i.click(fn=_scan_models, inputs=[], outputs=[dd_model_i2i, dd_lora_i2i])
        
            # Main runner (generator)
            def do_i2i(path, model_choice, lora_choice, lora_weight, sampler, scheduler,
                       stp, cfg, w, h, den, bs, bc, sd, scaled_fallback, prompt_text):
        
                # Resolve source before printing/logging
                src = path or scaled_fallback or DEFAULT_INPUT_DIR
                print("🚀 Forge Img2Img started with src:", src)
        
                # Build prompt
                prompt = (str(prompt_text or "").strip())
                if lora_choice:
                    try:
                        lw = float(lora_weight) if lora_weight is not None else 1.0
                    except Exception:
                        lw = 1.0
                    lora_name = os.path.splitext(str(lora_choice))[0]
                    prompt = (prompt + f" <lora:{lora_name}:{lw:.2f}>").strip()
        
                settings = ForgeSettings(
                    model_checkpoint=str(model_choice or "flux1-dev.safetensors"),
                    prompt=prompt,
                    negative_prompt="",
                    sampler_name=str(sampler),
                    schedule=str(scheduler),
                    distilled_cfg_scale=0.0,
                    steps=int(stp), cfg_scale=float(cfg),
                    width=None if w in (None, "", "None") else int(w),
                    height=None if h in (None, "", "None") else int(h),
                    denoising_strength=float(den),
                    batch_size=int(bs), batch_count=int(bc), seed=int(sd),
                )
        
                # Stream updates coming from the backend generator
                for out_dir, saved_paths, status in run_forge_img2img(src, settings):
                    yield out_dir, gallery_preview(saved_paths), settings.__dict__, status
        
            # ✅ IMPORTANT: bind the button OUTSIDE the function
            run_i2i.click(
                fn=do_i2i,
                inputs=[
                    forge_input, dd_model_i2i, dd_lora_i2i, lora_strength_i2i,
                    dd_sampler_i2i, dd_scheduler_i2i,
                    fs_steps_i2i, fs_cfg_i2i, fs_width_i2i, fs_height_i2i,
                    fs_denoise_i2i, fs_bsize_i2i, fs_bcnt_i2i, fs_seed_i2i,
                    st_scaled_path, fs_prompt_i2i
                ],
                outputs=[out_i2i, gal3, st_settings, status_i2i]
            )
        
            # Pass result path to Masking tab
            out_i2i.change(lambda p: p, inputs=[out_i2i], outputs=[st_img2img_path])

        # ────────────────────────────────────────────────────────────────────
        # Tab 5: Comfy Masking (streaming with progress)
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("5) Comfy Masking"):
            with gr.Row():
                cm_input = gr.Textbox(value="", label="Images folder (defaults to Img2Img output)")
                use_prev = gr.Button("Use Img2Img From Tab 4")
        
            mask_btn = gr.Button("Run Comfy Masking")
            orig_box = gr.Textbox(label="Originals path", interactive=False)
            mask_box = gr.Textbox(label="Masks path", value=FINAL_MASK_DIR, interactive=False)
            status_mask = gr.Textbox(label="Progress", interactive=False)
        
            with gr.Row():
                gal4a = gr.Gallery(label="Originals", columns=6, height=240)
                gal4b = gr.Gallery(label="Masks", columns=6, height=240)
        
            # Fill input from Tab 4 state
            use_prev.click(
                lambda p: gr.update(value=p or ""),
                [st_img2img_path],
                [cm_input]
            )
        
            # Generator-based masking runner
            def do_mask(path, i2i_state):
                src = path or i2i_state or FORGE_IMG2IMG_OUT
                print("🚀 Comfy Masking started with src:", src)
        
                for in_dir, m_dir, originals, masks, status in run_comfy_masking_single(src):
                    yield (
                        in_dir,                # originals folder
                        m_dir,                 # masks folder
                        gallery_preview(originals),  # originals gallery
                        gallery_preview(masks),      # masks gallery
                        in_dir,                # update st_mask_orig
                        m_dir,                 # update st_mask_dir
                        status                 # progress text
                    )
        
            mask_btn.click(
                fn=do_mask,
                inputs=[cm_input, st_img2img_path],
                outputs=[
                    orig_box, mask_box, gal4a, gal4b,
                    st_mask_orig, st_mask_dir, status_mask
                ]
            )


        # ────────────────────────────────────────────────────────────────────
        # Tab 6: Forge Inpaint (Batch) — streaming generator
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("6) Forge Inpaint (Batch)"):
            with gr.Row():
                inpaint_img_dir = gr.Textbox(value="", label="Original images (defaults to Tab 5)")
                inpaint_mask_dir = gr.Textbox(value=FINAL_MASK_DIR, label="Masks folder")
                use_prev2 = gr.Button("Use Paths From Tab 5")

            with gr.Accordion("Settings (can reuse Img2Img selections)", open=False):
                with gr.Row():
                    dd_model_inp = gr.Dropdown(
                        label="Base model (Stable-diffusion)",
                        choices=sd_models_initial,
                        value=(sd_models_initial[0] if sd_models_initial else None),
                        interactive=True,
                        allow_custom_value=False,
                    )
                with gr.Row():
                    dd_lora_inp = gr.Dropdown(
                        label="LoRA (single select)",
                        choices=loras_initial,
                        value=(loras_initial[0] if loras_initial else None),
                        interactive=True,
                        multiselect=False,
                        allow_custom_value=False,
                    )
                    lora_strength_inp = gr.Number(value=1.0, precision=2, label="LoRA weight")
                    inpaint_area = gr.Radio(
                        choices=["Whole picture", "Only masked"],
                        value="Only masked",
                        label="Inpaint area"
                    )
                    inpaint_padding = gr.Slider(minimum=0, maximum=128, step=1, value=32, label="Padding around mask (px)")
                    btn_refresh_models_inp = gr.Button("↻ Refresh model/LoRA lists")

                with gr.Row():
                    dd_sampler_inp   = gr.Dropdown(label="Sampler",   choices=SAMPLERS,  value="DPM2",    interactive=True)
                    dd_scheduler_inp = gr.Dropdown(label="Schedule",  choices=SCHEDULERS, value="Simple", interactive=True)

                with gr.Row():
                    fs_steps_inp = gr.Number(value=30, label="Steps")
                    fs_cfg_inp   = gr.Number(value=1.0, label="CFG Scale")
                with gr.Row():
                    fs_width_inp  = gr.Number(value=1216, label="Width (None = auto)")
                    fs_height_inp = gr.Number(value=1664, label="Height (None = auto)")
                with gr.Row():
                    fs_denoise_inp = gr.Number(value=0.45, label="Denoising strength")
                    fs_bsize_inp   = gr.Number(value=1,    label="Batch size")
                    fs_bcnt_inp    = gr.Number(value=1,    label="Batch count (n_iter)")
                fs_seed_inp = gr.Number(value=-1, label="Seed (-1=random)")

            btn_inpaint   = gr.Button("Run Forge Inpaint (Batch)")
            out_inpaint   = gr.Textbox(label="Inpaint output", interactive=False)
            gal5          = gr.Gallery(label="Inpaint preview", columns=6, height=280)
            status_inpaint = gr.Textbox(label="Progress", interactive=False)

            # Bring paths from Masking tab
            use_prev2.click(
                lambda a, b: (gr.update(value=a or ""), gr.update(value=b or FINAL_MASK_DIR)),
                [st_mask_orig, st_mask_dir], [inpaint_img_dir, inpaint_mask_dir]
            )

            # Refresh models/loras for Inpaint
            btn_refresh_models_inp.click(fn=_scan_models, inputs=[], outputs=[dd_model_inp, dd_lora_inp])

            # Main runner (generator)
            def do_inpaint(img_dir, m_dir, model_choice, lora_choice, lora_weight,
                           sampler, scheduler, stp, cfg, w, h, den, bs, bc, sd,
                           mask_orig_state, mask_dir_state, base_settings,
                           inpaint_area_choice, inpaint_padding_px):

                src = img_dir or mask_orig_state or FORGE_IMG2IMG_OUT
                m2  = m_dir  or mask_dir_state  or FINAL_MASK_DIR
                print("🚀 Forge Inpaint started with src:", src, "mask:", m2)

                base = base_settings or ForgeSettings().__dict__
                fs = ForgeSettings(**base)

                # Model/sampler/scheduler overrides
                fs.model_checkpoint = str(model_choice or fs.model_checkpoint or "flux1-dev.safetensors")
                fs.sampler_name = str(sampler or fs.sampler_name or "DPM2")
                fs.schedule     = str(scheduler or fs.schedule or "Simple")
                fs.inpaint_full_res = (inpaint_area_choice == "Whole picture")
                fs.inpaint_full_res_padding = int(inpaint_padding_px)

                # Numeric overrides
                if stp is not None: fs.steps = int(stp)
                if cfg is not None: fs.cfg_scale = float(cfg)
                if w not in (None, "", "None"): fs.width = int(w)
                if h not in (None, "", "None"): fs.height = int(h)
                if den is not None: fs.denoising_strength = float(den)
                if bs is not None: fs.batch_size = int(bs)
                if bc is not None: fs.batch_count = int(bc)
                if sd is not None: fs.seed = int(sd)

                # Append LoRA tag into prompt
                if lora_choice:
                    try:
                        lw = float(lora_weight) if lora_weight is not None else 1.0
                    except Exception:
                        lw = 1.0
                    lora_name = os.path.splitext(str(lora_choice))[0]
                    fs.prompt = (fs.prompt + f" <lora:{lora_name}:{lw:.2f}>").strip()

                # Stream updates from backend generator
                for out_dir, saved, status in run_forge_inpaint(src, m2, fs):
                    yield out_dir, gallery_preview(saved), status

            # ✅ IMPORTANT: bind button outside the function
            btn_inpaint.click(
                do_inpaint,
                inputs=[
                    inpaint_img_dir, inpaint_mask_dir,
                    dd_model_inp, dd_lora_inp, lora_strength_inp,
                    dd_sampler_inp, dd_scheduler_inp,
                    fs_steps_inp, fs_cfg_inp, fs_width_inp, fs_height_inp,
                    fs_denoise_inp, fs_bsize_inp, fs_bcnt_inp, fs_seed_inp,
                    st_mask_orig, st_mask_dir, st_settings,
                    inpaint_area, inpaint_padding
                ],
                outputs=[out_inpaint, gal5, status_inpaint]
            )
        # ────────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────
        # Tab 7: Full Automation (Img2Img → Masking → Inpaint)
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("7) Full Automation"):
            gr.Markdown("### Run the entire pipeline (Forge Img2Img → Comfy Masking → Forge Inpaint) in one go")
        
            with gr.Row():
                auto_input = gr.Textbox(
                    value="",
                    placeholder=SCALE_OUT_ROOT,
                    label="Images folder (leave blank to use Tab 3 scaled)"
                )
                btn_use_scaled = gr.Button("Use Scaled From Tab 3")
        
            fs_prompt_auto = gr.Textbox(
                label="Prompt",
                placeholder="Describe your image... (LoRA tag will be appended automatically if selected)",
                lines=3,
            )
        
            status_auto = gr.Textbox(label="Progress", interactive=False)
        
            # Img2Img settings
            with gr.Accordion("Img2Img Settings", open=False):
                with gr.Row():
                    dd_model_auto = gr.Dropdown(
                        label="Base model (Stable-diffusion)",
                        choices=sd_models_initial,
                        value=(sd_models_initial[0] if sd_models_initial else None),
                        interactive=True,
                        allow_custom_value=False,
                    )
                with gr.Row():
                    dd_lora_auto = gr.Dropdown(
                        label="LoRA (single select)",
                        choices=loras_initial,
                        value=(loras_initial[0] if loras_initial else None),
                        interactive=True,
                        multiselect=False,
                        allow_custom_value=False,
                    )
                    lora_strength_auto = gr.Number(value=1.0, precision=2, label="LoRA weight")
                    btn_refresh_models_auto = gr.Button("↻ Refresh model/LoRA lists")
        
                with gr.Row():
                    dd_sampler_i2i   = gr.Dropdown(label="Sampler", choices=SAMPLERS, value="DPM2", interactive=True)
                    dd_scheduler_i2i = gr.Dropdown(label="Schedule", choices=SCHEDULERS, value="Simple", interactive=True)
        
                with gr.Row():
                    fs_steps_i2i = gr.Number(value=30, label="Steps")
                    fs_cfg_i2i   = gr.Number(value=1.0, label="CFG Scale")
        
                with gr.Row():
                    fs_width_i2i  = gr.Number(value=1216, label="Width (None = auto)")
                    fs_height_i2i = gr.Number(value=1664, label="Height (None = auto)")
        
                with gr.Row():
                    fs_denoise_i2i = gr.Number(value=0.45, label="Denoising strength")
                    fs_bsize_i2i   = gr.Number(value=1, label="Batch size")
                    fs_bcnt_i2i    = gr.Number(value=1, label="Batch count (n_iter)")
        
                fs_seed_i2i = gr.Number(value=-1, label="Seed (-1=random)")
        
            # Inpaint settings
            with gr.Accordion("Inpaint Settings", open=False):
                with gr.Row():
                    dd_sampler_inp   = gr.Dropdown(label="Sampler", choices=SAMPLERS, value="DPM2", interactive=True)
                    dd_scheduler_inp = gr.Dropdown(label="Schedule", choices=SCHEDULERS, value="Simple", interactive=True)
        
                with gr.Row():
                    fs_steps_inp = gr.Number(value=30, label="Steps")
                    fs_cfg_inp   = gr.Number(value=1.0, label="CFG Scale")
        
                with gr.Row():
                    fs_width_inp  = gr.Number(value=1216, label="Width (None = auto)")
                    fs_height_inp = gr.Number(value=1664, label="Height (None = auto)")
        
                with gr.Row():
                    fs_denoise_inp = gr.Number(value=0.45, label="Denoising strength")
                    fs_bsize_inp   = gr.Number(value=1, label="Batch size")
                    fs_bcnt_inp    = gr.Number(value=1, label="Batch count (n_iter)")
        
                fs_seed_inp = gr.Number(value=-1, label="Seed (-1=random)")
        
                with gr.Row():
                    inpaint_area_auto = gr.Radio(
                        choices=["Whole picture", "Only masked"],
                        value="Only masked",
                        label="Inpaint area"
                    )
                    inpaint_padding_auto = gr.Slider(
                        minimum=0, maximum=128, step=1,
                        value=32, label="Padding around mask (px)"
                    )
        
            with gr.Accordion("Template Manager", open=False):
                template_name = gr.Textbox(label="Template Name")
                btn_save_tpl = gr.Button("💾 Save Template")
                template_dropdown = gr.Dropdown(
                    label="Load Template",
                    choices=[os.path.splitext(f)[0] for f in os.listdir(TEMPLATES_ROOT) if f.endswith(".json")]
                )
                btn_refresh_tpl = gr.Button("↻ Refresh Templates")
                btn_load_tpl = gr.Button("📂 Load Template")
                status_tpl = gr.Textbox(label="Template Status", interactive=False)
        
            btn_auto = gr.Button("🚀 Run Full Pipeline")
        
            # 3 galleries (Img2Img, Masks, Inpaint)
            gal_auto_i2i   = gr.Gallery(label="Step 1: Img2Img Preview", columns=6, height=200)
            gal_auto_masks = gr.Gallery(label="Step 2: Masks Preview", columns=6, height=200)
            gal_auto_inp   = gr.Gallery(label="Step 3: Inpaint Preview", columns=6, height=200)
        
            # Use scaled folder from Tab 3
            btn_use_scaled.click(lambda p: gr.update(value=p or ""), [st_scaled_path], [auto_input])
        
            # Refresh model/LoRA lists
            def _scan_models_auto():
                sd = list_sd_models(MODEL_DIR_SD)
                lr = list_loras(MODEL_DIR_LORA)
                return (
                    gr.update(choices=sd, value=(sd[0] if sd else None)),
                    gr.update(choices=lr, value=(lr[0] if lr else None)),
                )
            btn_refresh_models_auto.click(fn=_scan_models_auto, inputs=[], outputs=[dd_model_auto, dd_lora_auto])
        
            # Main pipeline runner
            def run_full_pipeline(
                folder, model_choice, lora_choice, lora_weight,
                sampler_i2i, scheduler_i2i, stp_i2i, cfg_i2i, w_i2i, h_i2i, den_i2i, bs_i2i, bc_i2i, sd_i2i,
                sampler_inp, scheduler_inp, stp_inp, cfg_inp, w_inp, h_inp, den_inp, bs_inp, bc_inp, sd_inp,
                scaled_path, prompt_text,
                inpaint_area_choice, inpaint_padding_px
            ):
                from pipeline.forge_img2img import run_forge_img2img_single
                from pipeline.forge_inpaint import run_forge_inpaint_single
                from pipeline.comfy_mask import run_comfy_masking_single
        
                src_dir = folder or scaled_path or DEFAULT_INPUT_DIR
                images = list_images(src_dir)
        
                if not images:
                    yield [], [], [], "❌ No images found."
                    return
        
                all_i2i, all_masks, all_inp = [], [], []
        
                for idx, img in enumerate(images, 1):
                    yield all_i2i, all_masks, all_inp, f"🚀 [{idx}/{len(images)}] Starting pipeline for {os.path.basename(img)}"
        
                    # Build prompt
                    prompt = (str(prompt_text or "").strip())
                    if lora_choice:
                        try:
                            lw = float(lora_weight) if lora_weight is not None else 1.0
                        except Exception:
                            lw = 1.0
                        lora_name = os.path.splitext(str(lora_choice))[0]
                        prompt = (prompt + f" <lora:{lora_name}:{lw:.2f}>").strip()
        
                    # Img2Img settings
                    i2i_settings = ForgeSettings(
                        model_checkpoint=str(model_choice or "flux1-dev.safetensors"),
                        prompt=prompt,
                        negative_prompt="",
                        sampler_name=str(sampler_i2i),
                        schedule=str(scheduler_i2i),
                        distilled_cfg_scale=0.0,
                        steps=int(stp_i2i), cfg_scale=float(cfg_i2i),
                        width=None if w_i2i in (None, "", "None") else int(w_i2i),
                        height=None if h_i2i in (None, "", "None") else int(h_i2i),
                        denoising_strength=float(den_i2i),
                        batch_size=int(bs_i2i), batch_count=int(bc_i2i), seed=int(sd_i2i),
                    )
        
                    # Step 1: Img2Img
                    img_outs = []
                    for out_dir, saved_paths, status in run_forge_img2img_single(img, i2i_settings):
                        img_outs.extend(saved_paths)
                        yield gallery_preview(all_i2i + img_outs), all_masks, all_inp, status
                    all_i2i.extend(img_outs)
        
                    if not img_outs:
                        continue
        
                    # Step 2: Masking
                    masks = []
                    for in_dir, m_dir, originals, masks_so_far, status in run_comfy_masking_single(img_outs[0]):
                        masks = masks_so_far
                        yield gallery_preview(all_i2i), gallery_preview(all_masks + masks), all_inp, status
                    all_masks.extend(masks)
        
                    # Step 3: Inpainting
                    inp_settings = ForgeSettings(
                        model_checkpoint=str(model_choice or "flux1-dev.safetensors"),
                        prompt=prompt,
                        negative_prompt="",
                        sampler_name=str(sampler_inp),
                        schedule=str(scheduler_inp),
                        distilled_cfg_scale=0.0,
                        steps=int(stp_inp), cfg_scale=float(cfg_inp),
                        width=None if w_inp in (None, "", "None") else int(w_inp),
                        height=None if h_inp in (None, "", "None") else int(h_inp),
                        denoising_strength=float(den_inp),
                        batch_size=int(bs_inp), batch_count=int(bc_inp), seed=int(sd_inp),
                        inpaint_full_res=(inpaint_area_choice == "Whole picture"),
                        inpaint_full_res_padding=int(inpaint_padding_px),
                    )
        
                    img_inp = []
                    for out_inp, saved_inp, status in run_forge_inpaint_single(img_outs[0], m_dir, inp_settings):
                        img_inp.extend(saved_inp)
                        yield gallery_preview(all_i2i), gallery_preview(all_masks), gallery_preview(all_inp + img_inp), status
                    all_inp.extend(img_inp)
        
                    yield gallery_preview(all_i2i), gallery_preview(all_masks), gallery_preview(all_inp), f"✅ Finished {os.path.basename(img)}"
        
                yield gallery_preview(all_i2i), gallery_preview(all_masks), gallery_preview(all_inp), "🎉 Pipeline finished!"
        
            def list_templates():
                if not os.path.exists(TEMPLATES_ROOT):
                    return []
                return [os.path.splitext(f)[0] for f in os.listdir(TEMPLATES_ROOT) if f.endswith(".json")]
        
            def save_template_fn(name, *vals):
                if not name:
                    return "❌ Template name required"
                path = os.path.join(TEMPLATES_ROOT, f"{name}.json")
                keys = [
                    "model_choice","lora_choice","lora_weight",
                    "sampler_i2i","scheduler_i2i","steps_i2i","cfg_i2i","width_i2i","height_i2i",
                    "denoise_i2i","batch_size_i2i","batch_count_i2i","seed_i2i",
                    "sampler_inp","scheduler_inp","steps_inp","cfg_inp","width_inp","height_inp",
                    "denoise_inp","batch_size_inp","batch_count_inp","seed_inp",
                    "prompt","inpaint_area","inpaint_padding"
                ]
                data = dict(zip(keys, vals))
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                return f"✅ Template saved: {path}"
        
            def load_template_fn(name):
                path = os.path.join(TEMPLATES_ROOT, f"{name}.json")
                if not os.path.exists(path):
                    raise gr.Error(f"Template {name} not found")
                with open(path) as f:
                    data = json.load(f)
                return [
                    gr.update(value=data.get("model_choice")),
                    gr.update(value=data.get("lora_choice")),
                    gr.update(value=data.get("lora_weight", 1.0)),
        
                    gr.update(value=data.get("sampler_i2i")),
                    gr.update(value=data.get("scheduler_i2i")),
                    gr.update(value=data.get("steps_i2i")),
                    gr.update(value=data.get("cfg_i2i")),
                    gr.update(value=data.get("width_i2i")),
                    gr.update(value=data.get("height_i2i")),
                    gr.update(value=data.get("denoise_i2i", 0.45)),
                    gr.update(value=data.get("batch_size_i2i")),
                    gr.update(value=data.get("batch_count_i2i")),
                    gr.update(value=data.get("seed_i2i")),
        
                    gr.update(value=data.get("sampler_inp")),
                    gr.update(value=data.get("scheduler_inp")),
                    gr.update(value=data.get("steps_inp")),
                    gr.update(value=data.get("cfg_inp")),
                    gr.update(value=data.get("width_inp")),
                    gr.update(value=data.get("height_inp")),
                    gr.update(value=data.get("denoise_inp", 0.45)),
                    gr.update(value=data.get("batch_size_inp")),
                    gr.update(value=data.get("batch_count_inp")),
                    gr.update(value=data.get("seed_inp")),
        
                    gr.update(value=data.get("prompt", "")),
                    gr.update(value=data.get("inpaint_area")),
                    gr.update(value=data.get("inpaint_padding", 32)),
                ]
        
            # Bind button
            btn_auto.click(
                fn=run_full_pipeline,
                inputs=[
                    auto_input, dd_model_auto, dd_lora_auto, lora_strength_auto,
                    dd_sampler_i2i, dd_scheduler_i2i, fs_steps_i2i, fs_cfg_i2i, fs_width_i2i, fs_height_i2i,
                    fs_denoise_i2i, fs_bsize_i2i, fs_bcnt_i2i, fs_seed_i2i,
                    dd_sampler_inp, dd_scheduler_inp, fs_steps_inp, fs_cfg_inp, fs_width_inp, fs_height_inp,
                    fs_denoise_inp, fs_bsize_inp, fs_bcnt_inp, fs_seed_inp,
                    st_scaled_path, fs_prompt_auto, inpaint_area_auto, inpaint_padding_auto
                ],
                outputs=[gal_auto_i2i, gal_auto_masks, gal_auto_inp, status_auto]
            )
        
            # Template save/load
            btn_save_tpl.click(
                save_template_fn,
                [
                    template_name,
                    dd_model_auto, dd_lora_auto, lora_strength_auto,
                    dd_sampler_i2i, dd_scheduler_i2i, fs_steps_i2i, fs_cfg_i2i, fs_width_i2i, fs_height_i2i,
                    fs_denoise_i2i, fs_bsize_i2i, fs_bcnt_i2i, fs_seed_i2i,
                    dd_sampler_inp, dd_scheduler_inp, fs_steps_inp, fs_cfg_inp, fs_width_inp, fs_height_inp,
                    fs_denoise_inp, fs_bsize_inp, fs_bcnt_inp, fs_seed_inp,
                    fs_prompt_auto, inpaint_area_auto, inpaint_padding_auto
                ],
                [status_tpl]
            )
        
            btn_load_tpl.click(
                load_template_fn,
                [template_dropdown],
                [
                    dd_model_auto, dd_lora_auto, lora_strength_auto,
                    dd_sampler_i2i, dd_scheduler_i2i, fs_steps_i2i, fs_cfg_i2i, fs_width_i2i, fs_height_i2i,
                    fs_denoise_i2i, fs_bsize_i2i, fs_bcnt_i2i, fs_seed_i2i,
                    dd_sampler_inp, dd_scheduler_inp, fs_steps_inp, fs_cfg_inp, fs_width_inp, fs_height_inp,
                    fs_denoise_inp, fs_bsize_inp, fs_bcnt_inp, fs_seed_inp,
                    fs_prompt_auto, inpaint_area_auto, inpaint_padding_auto
                ]
            )
        
            btn_refresh_tpl.click(
                lambda: gr.update(choices=list_templates()),
                [],
                [template_dropdown]
            )
        


    return demo
