import io
import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from PIL import Image, ImageDraw

load_dotenv(find_dotenv())

st.set_page_config(page_title="Azure Face Detection", page_icon="🧑", layout="wide")
st.title("🧑 Azure Face Detection")
st.caption("Powered by Azure AI Vision Face API")

# Sidebar: credentials
with st.sidebar:
    st.header("Configuration")
    endpoint = st.text_input(
        "Face API Endpoint",
        value=os.getenv("AZURE_FACE_API_ENDPOINT", ""),
        placeholder="https://<resource>.cognitiveservices.azure.com/",
    )
    api_key = st.text_input(
        "Face API Key",
        value=os.getenv("AZURE_FACE_API_ACCOUNT_KEY", ""),
        type="password",
    )
    st.divider()
    st.header("Input")
    input_mode = st.radio("Source", ["Upload image", "Image URL"])

# Input area
image_bytes = None
image_url = None

if input_mode == "Upload image":
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "gif", "webp"])
    if uploaded:
        image_bytes = uploaded.read()
else:
    image_url = st.text_input(
        "Image URL",
        value="https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png",
    )

run = st.button("Detect Faces", type="primary", disabled=not (endpoint and api_key))

if run:
    if not endpoint or not api_key:
        st.error("Please provide both Endpoint and API Key in the sidebar.")
        st.stop()

    if input_mode == "Upload image" and not image_bytes:
        st.error("Please upload an image.")
        st.stop()

    if input_mode == "Image URL" and not image_url:
        st.error("Please enter an image URL.")
        st.stop()

    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    from azure.ai.vision.face import FaceClient
    from azure.ai.vision.face.models import (
        FaceDetectionModel,
        FaceRecognitionModel,
        FaceAttributeTypeDetection01,
        FaceAttributeTypeDetection03,
        FaceAttributeTypeRecognition04,
    )

    with st.spinner("Calling Azure Face API..."):
        try:
            with FaceClient(endpoint=endpoint, credential=AzureKeyCredential(api_key)) as face_client:
                if input_mode == "Upload image":
                    faces = face_client.detect(
                        image_bytes,
                        detection_model=FaceDetectionModel.DETECTION03,
                        recognition_model=FaceRecognitionModel.RECOGNITION04,
                        return_face_id=False,
                        return_face_attributes=[
                            FaceAttributeTypeDetection03.BLUR,
                            FaceAttributeTypeDetection03.HEAD_POSE,
                            FaceAttributeTypeDetection03.MASK,
                            FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION,
                        ],
                        return_face_landmarks=True,
                    )
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                else:
                    faces = face_client.detect_from_url(
                        url=image_url,
                        detection_model=FaceDetectionModel.DETECTION01,
                        recognition_model=FaceRecognitionModel.RECOGNITION04,
                        return_face_id=False,
                        return_face_attributes=[
                            FaceAttributeTypeDetection01.ACCESSORIES,
                            FaceAttributeTypeDetection01.EXPOSURE,
                            FaceAttributeTypeDetection01.GLASSES,
                            FaceAttributeTypeDetection01.NOISE,
                        ],
                    )
                    import urllib.request
                    with urllib.request.urlopen(image_url) as resp:
                        pil_image = Image.open(io.BytesIO(resp.read())).convert("RGB")

        except HttpResponseError as e:
            st.error(f"Azure API error: {e.message}")
            st.stop()

    # Draw bounding boxes
    draw = ImageDraw.Draw(pil_image)
    colors = ["#FF4B4B", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"]
    for idx, face in enumerate(faces):
        rect = face.face_rectangle
        color = colors[idx % len(colors)]
        draw.rectangle(
            [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height],
            outline=color,
            width=3,
        )
        draw.text((rect.left + 4, rect.top + 2), f"#{idx + 1}", fill=color)

    # Layout: image + results side by side
    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.subheader(f"Detected {len(faces)} face(s)")
        st.image(pil_image, use_container_width=True)

    with col_results:
        st.subheader("Face Attributes")
        if not faces:
            st.info("No faces detected.")
        for idx, face in enumerate(faces):
            color = colors[idx % len(colors)]
            with st.expander(f"Face #{idx + 1}", expanded=True):
                rect = face.face_rectangle
                st.markdown(f"**Bounding Box** — top: {rect.top}, left: {rect.left}, "
                            f"width: {rect.width}, height: {rect.height}")

                attrs = face.face_attributes
                if attrs:
                    cols = st.columns(2)
                    if attrs.head_pose:
                        cols[0].metric("Pitch", f"{attrs.head_pose.pitch:.1f}°")
                        cols[0].metric("Yaw", f"{attrs.head_pose.yaw:.1f}°")
                        cols[1].metric("Roll", f"{attrs.head_pose.roll:.1f}°")
                    if attrs.blur:
                        st.markdown(f"**Blur** — level: `{attrs.blur.blur_level}`, value: `{attrs.blur.value:.2f}`")
                    if attrs.mask:
                        st.markdown(f"**Mask** — type: `{attrs.mask.type}`, "
                                    f"nose/mouth covered: `{attrs.mask.nose_and_mouth_covered}`")
                    if attrs.quality_for_recognition:
                        st.markdown(f"**Quality for Recognition** — `{attrs.quality_for_recognition}`")
                    if attrs.glasses:
                        st.markdown(f"**Glasses** — `{attrs.glasses}`")
                    if attrs.accessories:
                        st.markdown(f"**Accessories** — {', '.join(str(a) for a in attrs.accessories) or 'none'}")
                    if attrs.exposure:
                        st.markdown(f"**Exposure** — level: `{attrs.exposure.exposure_level}`, "
                                    f"value: `{attrs.exposure.value:.2f}`")
                    if attrs.noise:
                        st.markdown(f"**Noise** — level: `{attrs.noise.noise_level}`, "
                                    f"value: `{attrs.noise.value:.2f}`")

                if face.face_landmarks:
                    with st.expander("Landmarks", expanded=False):
                        landmarks = face.face_landmarks.as_dict()
                        lm_cols = st.columns(2)
                        for i, (name, coords) in enumerate(landmarks.items()):
                            lm_cols[i % 2].markdown(
                                f"**{name}** — x: {coords['x']:.1f}, y: {coords['y']:.1f}"
                            )
