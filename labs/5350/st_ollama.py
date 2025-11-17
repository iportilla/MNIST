import streamlit as st
import httpx
import base64
import asyncio
from PIL import Image
import io

# -------------------------------
# Convert PIL image -> base64 JPEG
# -------------------------------
def pil_to_base64(pil_image):
    buf = io.BytesIO()
    # JPEG is what most examples use; quality 90 is usually plenty
    pil_image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------
# Call Ollama vision model correctly
# -------------------------------
async def analyze_with_ollama(pil_image, model: str, prompt: str):
    # Resize to something reasonable (helps some models)
    pil_image = pil_image.convert("RGB")
    pil_image.thumbnail((1024, 1024))

    img_b64 = pil_to_base64(pil_image)

    # NOTE: images MUST be inside the message object, not top level
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "http://localhost:11434/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Standard Ollama chat response: { ..., "message": {"role": "assistant", "content": "..."} }
    message = data.get("message", {})
    return message.get("content", "⚠️ No content returned from model."), data


# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title("🦙 Local Vision LLM (Ollama) – What’s in this Image?")

    st.write("This uses an Ollama **vision model** running locally to describe the image.")

    model = st.selectbox(
        "Choose a vision-capable Ollama model",
        [
            "llama3.2-vision",
            "llava:7b",
            "llava:13b",
            "moondream",
            "phi3-vision",
        ],
    )

    default_prompt = "Describe exactly what you see in this image. Be factual and avoid guessing."
    user_prompt = st.text_area("Prompt sent to the model:", value=default_prompt, height=80)

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded:
        # Load image with PIL
        pil_image = Image.open(uploaded)
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image with Vision LLM"):
            with st.spinner(f"Calling {model} via Ollama…"):
                description, raw = asyncio.run(
                    analyze_with_ollama(pil_image, model=model, prompt=user_prompt)
                )

            st.subheader("🧠 Vision LLM Description")
            st.write(description)

            with st.expander("Debug: raw JSON from Ollama"):
                st.code(raw, language="json")


if __name__ == "__main__":
    main()