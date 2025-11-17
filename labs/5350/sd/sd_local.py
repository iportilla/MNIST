import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
import time

# ---------------------------------------
# Load Model (Cached)
# ---------------------------------------
@st.cache_resource
def load_model(model_name):
    if model_name == "Stable Diffusion 1.5":
        pipe = StableDiffusionPipeline.from_pretrained(
            "./sd15",
            torch_dtype=torch.float32
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "./sdxl",
            torch_dtype=torch.float32
        )

    # Use CPU on Mac – diffusers will use Metal automatically
    pipe.to("mps")
    return pipe


# ---------------------------------------
# Streamlit UI
# ---------------------------------------
def main():
    st.title("🎨 Local Stable Diffusion Image Generator (Offline)")

    st.write("Generate images locally with Stable Diffusion running entirely on your Mac.")

    model_name = st.selectbox(
        "Choose Model",
        ["Stable Diffusion 1.5", "Stable Diffusion XL"]
    )

    prompt = st.text_area(
        "Enter your image prompt:",
        "A friendly robot painting on a canvas in a sunny art studio, digital art"
    )

    negative_prompt = st.text_area(
        "Negative Prompt (optional)",
        "low quality, blurry, distorted, deformed, watermark"
    )

    steps = st.slider("Inference Steps", 10, 60, 30)
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
    width = st.slider("Width", 512, 1024, 768, step=64)
    height = st.slider("Height", 512, 1024, 768, step=64)

    seed = st.number_input("Seed (set -1 for random)", value=-1)

    if st.button("Generate Image"):
        with st.spinner("Generating image locally…"):
            pipe = load_model(model_name)

            generator = torch.Generator("cpu")
            if seed >= 0:
                generator = generator.manual_seed(seed)

            start = time.time()
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator
            ).images[0]
            end = time.time()

        st.subheader("🖼️ Generated Image")
        st.image(image)

        st.write(f"⏱️ Generation time: {end-start:.2f} sec")

        # Save/download
        st.download_button(
            "Download Image",
            data=image_to_bytes(image),
            file_name="output.png",
            mime="image/png"
        )


# ---------------------------------------
# Helper
# ---------------------------------------
def image_to_bytes(image: Image.Image):
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    main()