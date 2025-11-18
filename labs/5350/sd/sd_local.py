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
SD15_PATH = "./sd15"
SDXL_PATH = "./sdxl"


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
    Load SD1.5 (fp32 on MPS) or SDXL (fp16) with memory-optimized configs.
    """

    device = DEVICE

    # Stable Diffusion 1.5 (must use fp32 on MPS → fp16 = black images)
    if model_name == "Stable Diffusion 1.5":
        if not os.path.isdir(SD15_PATH):
            raise FileNotFoundError(
                f"Model directory '{SD15_PATH}' not found.\n"
                f"Download SD1.5 using:\n"
                f"huggingface-cli download runwayml/stable-diffusion-v1-5 "
                f"--local-dir sd15 --local-dir-use-symlinks False"
            )
        pipe = StableDiffusionPipeline.from_pretrained(
            SD15_PATH,
            torch_dtype=torch.float32,  # FIX FOR BLACK IMAGES
        )

    # Stable Diffusion XL (fp16 fine)
    else:
        if not os.path.isdir(SDXL_PATH):
            raise FileNotFoundError(
                f"Model directory '{SDXL_PATH}' not found.\n"
                f"Download SDXL using:\n"
                f"huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 "
                f"--local-dir sdxl --local-dir-use-symlinks False"
            )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_PATH,
            torch_dtype=torch.float16,
        )

    pipe.to(device)

    # Memory optimizations (especially for MPS/smaller VRAM)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pass  # Some pipelines don't support it

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
    st.set_page_config(page_title="Local Stable Diffusion", layout="wide")
    st.title("🎨 Local Stable Diffusion (Offline Image Generator)")

    st.caption(f"Device detected: **{DEVICE.upper()}**")

    # Sidebar Settings
    with st.sidebar:
        st.header("Model Settings")

        model_name = st.selectbox(
            "Choose Model",
            ["Stable Diffusion 1.5", "Stable Diffusion XL"],
            index=0,
        )

        steps = st.slider("Inference Steps", 10, 60, 30)
        guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

        # SD1.5 defaults to 512×512, SDXL to 768×768
        if model_name == "Stable Diffusion 1.5":
            default_w, default_h = 512, 512
            max_res = 768
        else:
            default_w, default_h = 768, 768
            max_res = 1024

        width = st.slider("Width", 384, max_res, default_w, step=64)
        height = st.slider("Height", 384, max_res, default_h, step=64)

        seed = st.number_input("Seed (negative = random)", value=-1, step=1)

    # Prompt input
    prompt = st.text_area(
        "Prompt",
        "A friendly robot painting in a bright art studio, ultra detailed, digital art",
        height=80,
    )

    negative_prompt = st.text_area(
        "Negative Prompt",
        "low quality, blurry, distorted, watermark, text",
        height=60,
    )

    run_button = st.button("🚀 Generate Image")

    if run_button:
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
            return

        try:
            with st.spinner("Loading model…"):
                pipe = load_model(model_name)

            generator, final_seed = make_generator(seed, DEVICE)
            st.info(f"Using seed: **{final_seed}**")

            start_time = time.time()

            with st.spinner("Generating image…"):
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                )
                image = output.images[0]

            end_time = time.time()

            st.subheader("🖼️ Generated Image")
            st.image(image, width="stretch")

            st.caption(f"Generation time: {end_time - start_time:.2f} seconds")

            st.download_button(
                "💾 Download PNG",
                data=image_to_bytes(image),
                file_name="sd_output.png",
                mime="image/png",
            )

        except RuntimeError as e:
            if "out of memory" in str(e):
                st.error("⚠️ MPS ran out of memory.")
                st.write(
                    "- Lower image resolution\n"
                    "- Reduce inference steps\n"
                    "- Close other GPU-heavy apps\n"
                    "- Use SD 1.5 instead of SDXL"
                )
            else:
                st.error(f"Runtime error: {e}")

        except FileNotFoundError as e:
            st.error(str(e))

        except Exception as e:
            st.error(f"Unexpected error: {e}")

        # Free VRAM (important for MPS)
        if DEVICE == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
