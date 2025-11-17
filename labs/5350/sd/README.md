# 🎨 Local Stable Diffusion Image Generator (Offline, Mac Optimized)

This repository provides a complete, fully offline **Stable Diffusion image generation app** using:

- **Hugging Face Diffusers**
- **Apple Silicon (M1/M2/M3) optimized pipelines**
- **Streamlit as a friendly UI**
- **Stable Diffusion 1.5** or **Stable Diffusion XL**

Everything runs entirely **on your local machine**.  
No cloud. No external API calls. Maximum privacy and zero cost.

---

## 🚀 Features

✔ Runs **100% offline** on Mac (MPS acceleration via Metal)  
✔ Supports **SD 1.5** and **SDXL**  
✔ Adjustable parameters:
- Prompt  
- Negative prompt  
- Steps  
- Guidance scale  
- Width/height  
- Seed (for reproducibility)

✔ Download generated images  
✔ Clean and easy Streamlit UI  
✔ Fast generation using Apple Silicon GPU  

---

## 🛠️ Installation

### 1️⃣ Create a clean environment (recommended)

```bash
conda create -n sd python=3.10
conda activate sd
```

### 2️⃣ Install dependencies

```bash
pip install diffusers transformers accelerate scipy safetensors
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install streamlit
pip install invisible-watermark
```

### 3️⃣ Download Stable Diffusion models

```bash
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./sd15
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ./sdxl
```

---

## ▶️ Running the App

```bash
streamlit run sd_local_app.py
```

---

## 📁 Project Structure

```
.
├── sd_local_app.py
├── sd15/
├── sdxl/
└── README.md
```

---

## 🎨 Example Prompt

> “A friendly robot painting a portrait in a cozy art studio, ultra detailed, digital art, 8k”

---

## ❓ Troubleshooting

See README for common issues.

---

## 📜 License

MIT License.

