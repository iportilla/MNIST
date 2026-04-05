# Demo Lab — Azure Face Detection

This lab demonstrates real-time face detection using the **Azure AI Vision Face API**, exposed through two interfaces: a command-line script and an interactive Streamlit web UI.

---

## Files

| File | Description |
|------|-------------|
| `sample_face_detection.py` | CLI script — detects faces from a local image and a URL |
| `app.py` | Streamlit web UI — upload an image or enter a URL and visualise results |
| `.env` | API credentials (not committed) |
| `tomc1.jpeg` | Sample image used by the CLI script |

---

## Architecture

### System Overview

```mermaid
graph TD
    A[User] -->|Uploads image or enters URL| B[Streamlit UI - app.py]
    A -->|Runs script| C[CLI Script - sample_face_detection.py]
    B -->|Image bytes / URL| D[Azure AI Vision - Face API]
    C -->|Image bytes / URL| D
    D -->|Face rectangles - Attributes - Landmarks| B
    D -->|Face rectangles - Attributes - Landmarks| C
    B -->|Annotated image - + attribute cards| A
    C -->|Logged JSON output| A
```

### Streamlit App Flow

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant ENV as .env File
    participant API as Azure Face API

    U->>UI: Open http://localhost:8501
    UI->>ENV: Load AZURE_FACE_API_ENDPOINT\nAZURE_FACE_API_ACCOUNT_KEY
    ENV-->>UI: Credentials pre-filled in sidebar
    U->>UI: Upload image (or enter URL)
    U->>UI: Click "Detect Faces"
    UI->>API: POST /detect (image bytes)\nor detect_from_url (URL)
    API-->>UI: Face rectangles + attributes + landmarks
    UI->>UI: Draw bounding boxes on image (Pillow)
    UI-->>U: Annotated image + attribute cards
```

### Detection Models

```mermaid
graph LR
    subgraph Upload Mode
        A1[Image file] --> B1[Detection03 - Recognition04]
        B1 --> C1[Blur]
        B1 --> C2[Head Pose]
        B1 --> C3[Mask]
        B1 --> C4[Quality for Recognition]
        B1 --> C5[Landmarks]
    end

    subgraph URL Mode
        A2[Image URL] --> B2[Detection01 - Recognition04]
        B2 --> D1[Accessories]
        B2 --> D2[Exposure]
        B2 --> D3[Glasses]
        B2 --> D4[Noise]
    end
```

---

## Prerequisites

- Python 3.9+
- An **Azure AI Services** (or Face) resource with key and endpoint
- The `.venv` virtual environment in the repo root (already set up)

---

## Setup

1. **Add credentials to `.env`**

   ```
   AZURE_FACE_API_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
   AZURE_FACE_API_ACCOUNT_KEY=<your-key>
   ```

2. **Activate the virtual environment**

   ```bash
   source ../../.venv/bin/activate
   ```

3. **Install dependencies** (first time only)

   ```bash
   pip install azure-ai-vision-face python-dotenv streamlit pillow
   ```

---

## Running

### CLI Script

Detects faces in `tomc1.jpeg` and then from a sample URL, printing full JSON results to the console.

```bash
python sample_face_detection.py
```

### Streamlit UI

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Attribute Reference

| Attribute | Available in | Description |
|-----------|-------------|-------------|
| `blur` | Upload (Detection03) | Blur level and score |
| `headPose` | Upload (Detection03) | Pitch, yaw, roll in degrees |
| `mask` | Upload (Detection03) | Mask type and nose/mouth coverage |
| `qualityForRecognition` | Upload (Recognition04) | `low` / `medium` / `high` |
| `accessories` | URL (Detection01) | Hat, glasses, mask, etc. |
| `exposure` | URL (Detection01) | Exposure level and score |
| `glasses` | URL (Detection01) | `NoGlasses`, `ReadingGlasses`, `Sunglasses`, `SwimmingGoggles` |
| `noise` | URL (Detection01) | Noise level and score |

---

## Notes

- `return_face_id=False` is set because face identification/verification requires [additional Microsoft approval](https://aka.ms/facerecognition).
- The `.env` file is loaded automatically; values entered in the sidebar override environment variables at runtime.
