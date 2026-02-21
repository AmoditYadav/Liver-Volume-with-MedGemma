# LiverLens

**A Neuro-Symbolic Agent for Metabolic Health Profiling**

LiverLens is a research-grade, 3-stage pipeline that combines **deep learning segmentation**, **deterministic metric extraction**, and **LLM-powered clinical reasoning** to analyse liver CT scans.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        LiverLens Pipeline                           │
│                                                                     │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐  │
│  │  Stage A     │     │  Stage B         │     │  Stage C          │  │
│  │  Vision      │ ──► │  Deterministic   │ ──► │  Radiologist      │  │
│  │  Specialist  │     │  Bridge          │     │  Agent            │  │
│  │  (SwinUNETR) │     │  (Python Logic)  │     │  (MedGemma 8B)    │  │
│  └─────────────┘     └─────────────────┘     └──────────────────┘  │
│        │                      │                       │             │
│   Binary Mask           Volume (cc)            Structured          │
│   (Dice 0.97)          HU Statistics         Clinical Report       │
│                        Risk Flags                                   │
└──────────────────────────────────────────────────────────────────────┘
```

| Stage | Component | Role |
|---|---|---|
| **A** | SwinUNETR (3D) | Precise liver ROI extraction via sliding-window inference |
| **B** | Python Logic | Mathematical computation of volume, HU stats, steatosis/hepatomegaly flags |
| **C** | MedGemma 8B | Clinical note synthesis from computed evidence — no hallucinated numbers |

> **Why split the roles?** LLMs hallucinate numbers. By computing all metrics deterministically in Stage B, we guarantee mathematical accuracy. MedGemma only receives verified evidence and produces the natural-language report.

---

## Features

### Core Pipeline
- **SwinUNETR Segmentation** — 3D sliding-window inference (ROI 96³, 50% overlap) achieving Dice 0.9786
- **Deterministic Metrics** — Volume (cc), mean/median/std HU, steatosis index, hepatomegaly detection
- **MedGemma 8B Reports** — Structured findings, impression, and recommendations using `Gemma3ForConditionalGeneration`

### Research-Grade Extras
- **Explainable AI (XAI)** — Interactive HU density histogram with colour-coded regions (fatty / borderline / healthy)
- **Test-Time Augmentation (TTA)** — "High-Precision Mode" that rotates the volume 90°, re-segments, and averages masks for better boundary accuracy
- **Uncertainty Quantification** — Visualises pixel-level disagreement between SwinUNETR (3D) and MedGemma segmentation (2D) as a yellow overlay

### Interface
- **3D Slice Viewer** — Scroll through axial slices with mask + uncertainty overlays
- **Mask Export** — Download the segmentation mask as a NIfTI file
- **Dark Glassmorphism UI** — Modern design with gradient backgrounds and blur effects

---

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with ≥8 GB VRAM (CUDA 11.8+)
- SwinUNETR checkpoint at `results/liver_swinunetr/best_metric_model.pth`
- MedGemma 8B model at `model_medgemma/`

### Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Gradio app
python -m liverlens.app
```

Open `http://localhost:7860` in your browser.

### Usage
1. **Upload** a NIfTI CT scan (`.nii.gz`)
2. Optionally enable **High-Precision Mode (TTA)**
3. Click **Run Analysis**
4. Browse slices with the viewer, review the HU histogram and metrics
5. Read the MedGemma-generated clinical report

---

## Project Structure

```
liverlens/
├── __init__.py          # Package marker
├── segmentation.py      # Stage A: SwinUNETR sliding-window inference
├── metrics.py           # Stage B: Volume, HU stats, risk flags
├── report.py            # Stage C: MedGemma 8B report generation
├── tta.py               # Test-Time Augmentation module
├── uncertainty.py       # SwinUNETR vs MedGemma disagreement map
└── app.py               # Gradio UI (3-column dark glassmorphism)
```

---

## Technical Details

### Preprocessing
- **Orientation**: RAS standard
- **Spacing**: Resampled to 1mm isotropic via `Spacingd`
- **Intensity**: Clipped to [-200, 200] HU, scaled to [0, 1]
- **Spatial Size**: Resized to 160×160×96 for inference

### Metrics Computed
| Metric | Formula | Clinical Significance |
|---|---|---|
| Volume (cc) | `n_voxels × effective_voxel_volume / 1000` | >1800 cc → Hepatomegaly |
| Mean HU | `mean(HU[mask > 0])` | <40 HU → Steatosis (fatty liver) |
| Std HU | `std(HU[mask > 0])` | High → heterogeneity / focal lesion |
| Low HU Fraction | `% voxels < 40 HU` | >5% → steatosis marker |

### Model Performance

| Model | Architecture | Dice Score | Parameters |
|---|---|---|---|
| SwinUNETR | Swin Transformer (3D) | **0.9786** | ~62M |
| MedGemma Seg | SigLIP + UNet decoder (2D) | 0.95 | ~4B (encoder frozen) |

---

## Key Research Concepts

- **Neuro-Symbolic AI** — Neural networks (SwinUNETR) combined with symbolic logic (metric computation) and LLM reasoning (MedGemma)
- **Agentic Workflow** — Each stage has a specialised role; the pipeline mimics a clinical workflow where the radiologist interprets measurements computed by technicians
- **Test-Time Augmentation** — Improves inference reliability without retraining by averaging predictions from multiple orientations
- **Uncertainty Quantification** — Model disagreement highlights regions that warrant human review, building trust in AI-assisted diagnosis

---

## References

- [MONAI](https://monai.io/) — Medical Open Network for AI
- [SwinUNETR](https://arxiv.org/abs/2201.01266) — Swin Transformers for 3D Medical Image Segmentation
- [MedGemma](https://ai.google.dev/gemma/docs/medgemma) — Google's medical foundation model
- [U-Net](https://arxiv.org/abs/1505.04597) — Convolutional Networks for Biomedical Image Segmentation

---

> **Disclaimer**: This is a research prototype. It is not validated for clinical use. Always consult a qualified radiologist for medical diagnosis.
