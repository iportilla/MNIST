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
