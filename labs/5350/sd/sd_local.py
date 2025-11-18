import os
import time
import io
import random

import streamlit as st
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


# -------------------------------------------------------------------
# CONFIG: where your models live
# -------------------------------------------------------------------
SD15_PATH = "./sd15"   # runwayml/stable-diffusion-v1-5 downloaded here
SDXL_PATH = "./sdxl"   # stabilityai/stable-diffusion-xl-base-1.0 downloaded here


# -------------------------------------------------------------------
# DEVICE SELECTION
# -------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


DEVICE = get_device()


# -------------------------------------------------------------------
# MODEL LOADING (CACHED)
# -------------------------------------------------------------------
@st.cache_resource
def load_model(model_name: str):
    """
    Load SD 1.5 or SDXL into memory with safe defaults for Apple Silicon.
    """
    device = DEVICE

    if model_name == "Stable Diffusion 1.5":
        if not os.path.isdir(SD15_PATH):
            raise FileNotFoundError(
                f"Model folder '{SD15_PATH}' not found. "
                f"Download runwayml/stable-diffusion-v1-5 there."
            )

        pipe = StableDiffusionPipeline.from_pretrained(
            SD15_PATH,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )

    else:  # SDXL
        if not os.path.isdir(SDXL_PATH):
            raise FileNotFoundError(
                f"Model folder '{SDXL_PATH}' not found. "
                f"Download stabilityai/stable-diffusion-xl-base-1.0 there."
            )

        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_PATH,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )

    pipe.to(device)

    # Memory-saving configs (especially for MPS / SDXL)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        # Not all configs support this; safe to ignore
        pass

    return pipe


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def make_generator(seed: int, device: str):
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    gen = torch.Generator(device if device != "mps" else "cpu")
    gen.manual_seed(seed)
    return gen, seed


# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Local Stable Diffusion", layout="centered")
    st.title("🎨 Local Stable Diffusion Image Generator")

    st.markdown(
        "Generate images **completely offline** using Stable Diffusion running on your Mac."
    )

    st.caption(f"Detected device: **{DEVICE.upper()}**")

    # Sidebar
    with st.sidebar:
        st.header("Model & Settings")

        model_name = st.selectbox(
            "Model",
            ["Stable Diffusion 1.5", "Stable Diffusion XL"],
        )

        steps = st.slider("Inference steps", 10, 60, 30)
        guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5)

        if model_name == "Stable Diffusion 1.5":
            default_w, default_h = 512, 512
            max_res = 768
        else:
            default_w, default_h = 768, 768
            max_res = 896

        width = st.slider("Width", 384, max_res, default_w, step=64)
        height = st.slider("Height", 384, max_res, default_h, step=64)

        seed = st.number_input(
            "Seed (negative for random)", value=-1, step=1
        )

    # Main prompt area
    prompt = st.text_area(
        "Prompt",
        "A friendly robot painting in a bright art studio, ultra detailed, digital art",
        height=80,
    )

    negative_prompt = st.text_area(
        "Negative prompt",
        "low quality, blurry, distorted, deformed, watermark, text",
        height=60,
    )

    generate_button = st.button("🚀 Generate Image")

    if generate_button:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        try:
            with st.spinner("Loading model (first time may take a bit)…"):
                pipe = load_model(model_name)

            generator, actual_seed = make_generator(seed, DEVICE)

            st.info(f"Using seed: **{actual_seed}**")

            start_time = time.time()
            with st.spinner("Generating image…"):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt.strip() else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                )
                image = result.images[0]
            end_time = time.time()

            st.subheader("🖼️ Generated Image")
            st.image(image, use_container_width=True)

            st.caption(f"Generation time: {end_time - start_time:.1f} seconds")

            st.download_button(
                "💾 Download PNG",
                data=image_to_bytes(image),
                file_name="sd_output.png",
                mime="image/png",
            )

        except FileNotFoundError as e:
            st.error(str(e))
            st.info(
                "Make sure you ran:\n\n"
                "```bash\n"
                "huggingface-cli download runwayml/stable-diffusion-v1-5 "
                "--local-dir sd15 --local-dir-use-symlinks False\n\n"
                "huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 "
                "--local-dir sdxl --local-dir-use-symlinks False\n"
                "```"
            )
        except RuntimeError as e:
            # Typical MPS OOM message
            if "MPS backend out of memory" in str(e):
                st.error("⚠️ MPS out of memory.")
                st.write(
                    "Try:\n"
                    "- Lowering width/height\n"
                    "- Reducing inference steps\n"
                    "- Using **Stable Diffusion 1.5** instead of SDXL\n"
                    "- Closing other GPU-hungry apps"
                )
            else:
                st.error(f"Runtime error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

        # Try to free some MPS memory between runs
        if DEVICE == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
