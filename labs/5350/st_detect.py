import streamlit as st
import cv2
import numpy as np

def main():
    st.title("🖼️ Image Recognition - What's in the Image?")
    st.write("Please upload an image.")

    # FIXED: correct function name
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # FIXED: cv2 cannot read directly; decode manually
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                st.image(img, channels="BGR", caption="Uploaded Image")

                # Resize for simple preprocessing
                resized_img = cv2.resize(img, (224, 224))
                img_np = np.array(resized_img)

                # VERY simple “feature extraction”
                features = {
                    "Red channel sum": int(np.sum(img_np[:,:,2])),
                    "Green channel sum": int(np.sum(img_np[:,:,1])),
                    "Blue channel sum": int(np.sum(img_np[:,:,0])),
                }

                st.subheader("🧠 Simple Feature Summary")
                st.json(features)

                st.subheader("Detected Objects (placeholder)")
                st.write("Object detection model not implemented. Add your model here!")

            else:
                st.warning("Image could not be read.")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()