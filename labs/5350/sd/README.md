# 🎨 Local Stable Diffusion Image Generator (Offline, Apple Silicon Optimized)

This repository contains a fully offline Stable Diffusion image generator using **Hugging Face Diffusers**, **Streamlit**, and **Apple Silicon (M1/M2/M3/M4) Metal (MPS)** acceleration.  
It supports **Stable Diffusion 1.5 (FP32 fix for MPS)** and **Stable Diffusion XL (FP16)**.

Everything runs **100% locally**, ensuring privacy and zero API costs.

---

## 🚀 Features

✔ Runs 100% offline  
✔ SD 1.5 (FP32 – fixes black image issue on MPS)  
✔ SDXL (FP16 for performance)  
✔ Adjustable: prompt, negative prompt, steps, guidance, width, height, seed  
✔ Optimized for Apple Silicon (attention slicing, VAE tiling, CPU offload)  
✔ Modern Streamlit UI using `width="stretch"`  
✔ Safe against out‑of‑memory errors  
✔ Download generated images  
✔ Fully private and secure  

---

## 📁 Project Structure

```
.
├── sd_local_app.py      # Main Streamlit app
├── sd15/                # Stable Diffusion 1.5 model folder (Diffusers format)
├── sdxl/                # Stable Diffusion XL model folder
└── README.md
```

---

## 🛠 Installation

### 1️⃣ Create Environment

```bash
conda create -n sd python=3.10
conda activate sd
```

or

```bash
python3 -m venv sd
source sd/bin/activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install diffusers transformers accelerate scipy safetensors
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install streamlit
pip install invisible-watermark   # optional
```

Torch CPU wheels automatically use **Metal (MPS)** acceleration on macOS.

---

## 📥 Download Required Models

### 🔹 Stable Diffusion 1.5 (Diffusers format)

```bash
huggingface-cli login

huggingface-cli download runwayml/stable-diffusion-v1-5   --local-dir sd15   --local-dir-use-symlinks False
```

### 🔹 Stable Diffusion XL

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0   --local-dir sdxl   --local-dir-use-symlinks False
```

### ✔ Token‑Free Alternative Models (optional)

```bash
huggingface-cli download dreamlike-art/dreamlike-photoreal-2.0   --local-dir sd15 --local-dir-use-symlinks False

huggingface-cli download SG161222/RealVisXL_V4.0   --local-dir sdxl --local-dir-use-symlinks False
```

These include full Diffusers folders and generate high‑quality images.

---

## ▶️ Running the App

```bash
streamlit run sd_local_app.py
```

---

## 🧠 Model Behavior

### Stable Diffusion 1.5
- Runs in **float32** on MPS  
- Prevents black/dark images  
- Works best at **512×512 resolution**

### Stable Diffusion XL
- Runs in **float16**  
- Works well at **768×768**  
- Higher detail and realism  

---

## 🔧 Memory Optimization (Apple Silicon)

The app automatically enables:

- `enable_attention_slicing()`
- `enable_vae_tiling()`
- `enable_sequential_cpu_offload()`
- VRAM cleanup via `torch.mps.empty_cache()`

This greatly reduces out‑of‑memory errors.

---

## ❗ Troubleshooting

### 🟥 Black Image Output
SD1.5 must run in **FP32** on MPS → already fixed in app.

### 🟥 MPS Out of Memory
Reduce:
- resolution  
- inference steps  
- guidance scale  

Or switch to SD 1.5.

### 🟥 Missing model files
Ensure you downloaded **full diffusers repositories**, not `.ckpt` files.

---

## 🌟 Future Feature Ideas

This project is ready for extensions:

- Img2Img  
- Inpainting  
- LoRA loading  
- Prompt enhancement with local LLM (Ollama)  
- Batch generation  
- Multi-image grids  
- SDXL Refiner  

Ask and they can be added!

---

## 📜 License

This project is under the MIT License.  
Model licenses vary depending on provider (Hugging Face datasets/models).

Enjoy creating offline AI art! 🎨✨
