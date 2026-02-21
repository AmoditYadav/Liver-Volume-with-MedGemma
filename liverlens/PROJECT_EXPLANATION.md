# LiverLens: Project Architecture & Methodology

This document outlines the design philosophy, architectural choices, and technical approach taken to develop LiverLens, a research-grade multimodal AI pipeline for liver metabolic health profiling.

## 1. The Core Problem
Medical image analysis is traditionally disjointed. You have segmentation models (like U-Net) that generate masks, but require a radiologist to manually compute volumes, locate intensities, and synthesize findings into a report.
Conversely, end-to-end Vision-Language Models (VLMs) attempt to look at an image and directly output a diagnosis. However, LLMs and VLMs are notoriously bad at basic math and precise spatial reasoning. If you ask a VLM "what is the volume of this liver in cubic centimeters?", it will hallucinate a number.

**Our Goal:** Build an automated pipeline that can segment a 3D liver, compute precise mathematical metrics (volume, density), and generate a clinical-style report, *without allowing the AI to hallucinate the critical patient data*.

## 2. Our Approach: A "Neuro-Symbolic" Pipeline
To solve the hallucination problem, we adopted a neuro-symbolic approach. We split the pipeline into three distinct, specialized stages. The neural networks handle what they're good at (pattern recognition and language synthesis), while classical code handles what it's good at (math).

### Stage A: The Vision Specialist (Deep Learning)
*   **Component:** 3D SwinUNETR (Swin UNEt Transformers).
*   **Task:** Identify and extract the liver Region of Interest (ROI) from a 3D NIfTI CT scan.
*   **Why SwinUNETR?** The liver is a large, 3D organ. 2D slice-by-slice segmentation often loses anatomical context along the Z-axis. SwinUNETR uses a 3D sliding-window approach with a patch-based transformer architecture, allowing it to capture both local textures and global spatial dependencies. 
*   **Performance:** Achieved a Dice similarity coefficient of 0.9786.

### Stage B: The Deterministic Bridge (Symbolic Logic)
*   **Component:** Python/NumPy logic interacting with MONAI masking arrays.
*   **Task:** Extract hard mathematical metrics from the Stage A mask.
*   **How:** 
    *   **Volume:** We count the number of mask voxels (`mask > 0`) and multiply by the effective voxel spacing (calculated from the NIfTI affine matrix). This yields a precise volume in cubic centimeters.
    *   **Density (Hounsfield Units):** We calculate the mean, median, and standard deviation of the HU values exclusively within the liver mask. We also calculate the fraction of voxels falling below 40 HU (a known threshold for steatosis/fatty liver).
*   **Why:** This stage acts as a firewall against hallucinations. The LLM never sees the raw image to guess the volume; it only receives the mathematical truth.

### Stage C: The Radiologist Agent (Vision-Language Model)
*   **Component:** MedGemma 8B (specifically, `Gemma3ForConditionalGeneration` with a frozen SigLIP vision encoder).
*   **Task:** Synthesize the hard data into a clinical report (Findings, Impression, Recommendations).
*   **How:** We format the metrics from Stage B into a stringent text prompt. We essentially tell MedGemma: *"You are an expert radiologist. Here is the mathematically verified patient data: [Volume: X cc, Mean HU: Y]. Write a report based ONLY on these numbers."*
*   **Why MedGemma?** It is pre-trained by Google on medical literature and clinical notes, meaning it intrinsically understands what a "normal" liver volume is, and knows that low HU correlates with hepatic steatosis.

## 3. Advanced Diagnostic Features

To make LiverLens a true "research-grade" tool, we implemented several advanced techniques:

### Uncertainty Quantification (Model Disagreement)
Deep learning models are "black boxes" that don't know when they are wrong. We built an Uncertainty module to highlight areas where the AI is unsure.
*   **Implementation:** We pass the middle 2D slice of the CT scan to the MedGemma 2D segmentation decoder (a lightweight CNN built on top of the SigLIP encoder). We then compare this 2D mask against the 3D SwinUNETR mask. 
*   **Result:** The pixels where SwinUNETR and MedGemma disagree are highlighted in yellow on the UI. This acts as a visual flag for clinicians, saying *"Pay attention here, the models are conflicting."*

### Test-Time Augmentation (TTA)
In medical imaging, taking a single pass at an image can lead to artifacts or jagged boundaries.
*   **Implementation:** Our TTA mode rotates the 3D CT volume by 90 degrees, runs the SwinUNETR inference again on the rotated volume, rotates the output mask back to normal, and averages it with the original mask.
*   **Result:** This drastically smooths the segmentation boundaries and increases robustness against unusual spatial orientations, at the cost of doubling the inference time.

### Explainable AI (XAI) Histograms
Raw numbers aren't always intuitive. We map the liver's density profile onto a dynamic histogram, coloring regions below 40 HU as "Fatty", 40-50 HU as "Borderline", and >50 HU as "Healthy". This provides immediate, visual Explainable AI context to the mean HU metric.

## 4. Technical Stack & Deployment
*   **Core Logic:** Python, PyTorch, MONAI (Medical Open Network for AI).
*   **Language & Vision:** Hugging Face `transformers` (MedGemma integration).
*   **User Interface:** Gradio (designed with custom CSS for a modern, dark-glassmorphism aesthetic, ensuring it feels like a premium medical tool rather than a quick script).
*   **Data Handling:** `nibabel` for loading and affine-transforming high-dimensional NIfTI `.nii.gz` files.
