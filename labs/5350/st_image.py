import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")   # small + fast model

model = load_model()

def main():
    st.title("🖼️ Image Recognition + Object Detection")
    st.write("Upload an image to see detected objects using YOLOv8.")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                st.image(img, channels="BGR", caption="Uploaded Image")

                # Run YOLOv8 object detection
                results = model(img)

                # Extract detections
                annotated_img = results[0].plot()  # YOLO draws boxes automatically

                # Show annotated image
                st.subheader("🟦 Object Detection Results")
                st.image(annotated_img, channels="BGR", caption="Objects Detected")

                # Summary list
                st.subheader("📋 Detected Objects:")
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = results[0].names[cls]
                    conf = float(box.conf[0])
                    st.write(f"- **{label}** ({conf:.2f})")

            else:
                st.warning("Image could not be read.")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

##pip install ultralytics opencv-python-headless