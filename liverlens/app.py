"""
LiverLens -- Gradio Application.

A Neuro-Symbolic Agent for Metabolic Health Profiling.
Three-stage pipeline: SwinUNETR -> Metric Extraction -> MedGemma Report.
"""

import sys
import os
import tempfile

# Ensure project root is on the path so medgemma_model can be imported
sys.path.insert(0, r"D:\Monai")

import warnings
warnings.filterwarnings("ignore")

import gradio as gr
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io


# --- Custom CSS -----------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%) !important;
}

.main-title {
    text-align: center;
    background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #6dd5ed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4em !important;
    font-weight: 700 !important;
    margin-bottom: 0 !important;
    letter-spacing: -0.5px;
}

.subtitle {
    text-align: center;
    color: #8892b0 !important;
    font-size: 1.05em !important;
    margin-top: 4px !important;
    font-weight: 300 !important;
}

.panel {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(10px) !important;
}

.status-running { color: #fbbf24 !important; font-weight: 600; }
.status-done    { color: #34d399 !important; font-weight: 600; }
.status-error   { color: #f87171 !important; font-weight: 600; }

footer { display: none !important; }
"""


# --- Visualisation helpers ------------------------------------------------

def render_slice(volume, mask, disagreement, slice_idx, show_mask, show_uncertainty):
    """Render a single axial slice with optional overlays."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Base CT image
    ct = volume[:, :, slice_idx]
    ct_disp = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
    ax.imshow(ct_disp, cmap="gray", aspect="equal")

    # Liver mask overlay (red)
    if show_mask and mask is not None:
        mask_slice = mask[:, :, slice_idx]
        mask_rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        mask_rgba[mask_slice > 0] = [1.0, 0.2, 0.2, 0.40]
        ax.imshow(mask_rgba, aspect="equal")

    # Disagreement overlay (yellow)
    if show_uncertainty and disagreement is not None:
        dis_slice = disagreement[:, :, slice_idx]
        dis_rgba = np.zeros((*dis_slice.shape, 4), dtype=np.float32)
        dis_rgba[dis_slice > 0] = [1.0, 1.0, 0.0, 0.55]
        ax.imshow(dis_rgba, aspect="equal")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Slice {slice_idx + 1} / {volume.shape[2]}",
        color="white", fontsize=11, fontweight="500", pad=8,
    )
    plt.tight_layout(pad=0.5)
    return fig


def render_histogram(metrics):
    """XAI: Liver HU density histogram."""
    counts = np.array(metrics.get("hist_counts", []))
    bins   = np.array(metrics.get("hist_bins", []))
    if counts.size == 0:
        return None

    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1e1e3a")

    # Color bars by HU value
    colors = []
    for c in centers:
        if c < 40:
            colors.append("#ff6b6b")   # fatty zone
        elif c < 55:
            colors.append("#ffd93d")   # borderline
        else:
            colors.append("#6bcb77")   # healthy

    ax.bar(centers, counts, width=(bins[1] - bins[0]) * 0.85,
           color=colors, edgecolor="none", alpha=0.9)

    # Reference lines
    ax.axvline(40, color="#ff6b6b", ls="--", lw=1, alpha=0.7)
    ax.axvline(55, color="#6bcb77", ls="--", lw=1, alpha=0.7)

    ax.text(30, ax.get_ylim()[1] * 0.9, "Fatty", color="#ff6b6b",
            fontsize=8, ha="center", fontweight="600")
    ax.text(80, ax.get_ylim()[1] * 0.9, "Healthy", color="#6bcb77",
            fontsize=8, ha="center", fontweight="600")

    ax.set_xlabel("Hounsfield Units (HU)", color="white", fontsize=9)
    ax.set_ylabel("Voxel Count", color="white", fontsize=9)
    ax.set_title("Liver Density Distribution", color="white",
                 fontsize=11, fontweight="600", pad=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    plt.tight_layout()
    return fig


def format_metrics_table(metrics):
    """Format metrics as a markdown table for display."""
    flags = []
    if metrics.get("steatosis_flag"):
        flags.append("**Steatosis Risk** -- Mean density < 40 HU")
    if metrics.get("hepatomegaly_flag"):
        flags.append("**Hepatomegaly Risk** -- Volume > 1800 cc")
    if not flags:
        flags.append("No abnormalities detected")

    return (
        "### Computed Metrics\n\n"
        "| Measure | Value | Normal Range |\n"
        "|---|---|---|\n"
        f"| Liver Volume | **{metrics['volume_cc']:.1f} cc** | 1200-1600 cc |\n"
        f"| Mean Density | **{metrics['mean_hu']:.1f} HU** | > 55 HU |\n"
        f"| Density Std Dev | **{metrics['std_hu']:.1f} HU** | -- |\n"
        f"| Median Density | **{metrics['median_hu']:.1f} HU** | -- |\n"
        f"| Voxels < 40 HU | **{metrics['low_frac_40']*100:.1f}%** | < 5% |\n"
        f"| Voxels < 50 HU | **{metrics['low_frac_50']*100:.1f}%** | < 15% |\n"
        f"| Total Liver Voxels | **{metrics['n_liver_voxels']:,}** | -- |\n\n"
        "### Risk Assessment\n\n" +
        "\n".join(flags)
    )


# --- Pipeline functions ---------------------------------------------------

def run_pipeline(nifti_file, enable_tta, progress=gr.Progress(track_tqdm=True)):
    """Execute the full 3-stage LiverLens pipeline."""
    if nifti_file is None:
        raise gr.Error("Please upload a NIfTI CT scan (.nii.gz)")

    filepath = nifti_file.name if hasattr(nifti_file, "name") else nifti_file

    # -- Stage A: Segmentation --
    if enable_tta:
        progress(0.05, desc="Stage A -- SwinUNETR + TTA (High Precision)...")
        from liverlens.tta import segment_with_tta
        seg_result = segment_with_tta(filepath)
    else:
        progress(0.1, desc="Stage A -- SwinUNETR segmentation...")
        from liverlens.segmentation import segment
        seg_result = segment(filepath)

    mask        = seg_result["mask_3d"]
    volume      = seg_result["volume_3d"]
    spacing     = seg_result["voxel_spacing"]
    eff_spacing = seg_result["effective_spacing"]
    n_slices    = seg_result["n_slices"]

    # -- Stage B: Metrics --
    progress(0.4, desc="Stage B -- Computing liver metrics...")
    from liverlens.metrics import compute_metrics
    metrics = compute_metrics(mask, volume, spacing, effective_spacing=eff_spacing)

    # -- Stage C: Report --
    progress(0.6, desc="Stage C -- MedGemma generating report...")
    from liverlens.report import generate_report
    report = generate_report(metrics)

    # -- Uncertainty --
    progress(0.8, desc="Computing model disagreement...")
    from liverlens.uncertainty import compute_disagreement
    disagreement = compute_disagreement(mask, volume)

    progress(1.0, desc="Analysis complete")

    # Build initial slice view (middle slice)
    mid = n_slices // 2
    slice_fig      = render_slice(volume, mask, disagreement, mid, True, True)
    hist_fig       = render_histogram(metrics)
    metrics_md     = format_metrics_table(metrics)

    tta_label = " (TTA enabled)" if enable_tta else ""
    status = f"Analysis complete -- {n_slices} slices processed{tta_label}"

    return (
        volume,                # state: volume
        mask,                  # state: mask
        disagreement,          # state: disagreement
        n_slices,              # state: n_slices
        metrics,               # state: metrics
        gr.update(value=mid, maximum=n_slices - 1, interactive=True),  # slider
        slice_fig,             # viewer
        hist_fig,              # histogram
        metrics_md,            # metrics table
        report,                # report text
        status,                # status
    )


def update_slice(slice_idx, volume, mask, disagreement, show_mask, show_uncertainty):
    """Callback: re-render when user moves the slice slider."""
    if volume is None:
        return None
    fig = render_slice(volume, mask, disagreement, int(slice_idx),
                       show_mask, show_uncertainty)
    return fig


def download_mask(mask, volume):
    """Save the binary mask as a NIfTI file for download."""
    if mask is None:
        raise gr.Error("Run analysis first to generate a mask")
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False, prefix="liverlens_mask_")
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4))
    nib.save(nii, tmp.name)
    return tmp.name


# --- Build Gradio UI -----------------------------------------------------

def create_app():
    with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="LiverLens") as demo:
        # -- Header --
        gr.HTML(
            '<h1 class="main-title">LiverLens</h1>'
            '<p class="subtitle">'
            'A Neuro-Symbolic Agent for Metabolic Health Profiling &nbsp;|&nbsp; '
            'SwinUNETR  ->  Metric Bridge  ->  MedGemma 8B'
            '</p>'
        )

        # -- State --
        st_volume       = gr.State(None)
        st_mask          = gr.State(None)
        st_disagreement  = gr.State(None)
        st_nslices       = gr.State(0)
        st_metrics       = gr.State(None)

        # -- Main layout --
        with gr.Row(equal_height=False):
            # - Left column: Controls -
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Controls")
                file_input = gr.File(
                    label="Upload CT Scan",
                    file_types=[".nii", ".nii.gz", ".gz"],
                    type="filepath",
                )

                tta_cb = gr.Checkbox(
                    value=False,
                    label="High-Precision Mode (TTA)",
                )
                gr.Markdown(
                    '<small style="color:#888;">Runs inference twice with '
                    'rotation augmentation for better boundary accuracy. '
                    'Takes ~2x longer.</small>'
                )

                run_btn = gr.Button(
                    "Run Analysis",
                    variant="primary",
                    size="lg",
                )
                status_box = gr.Textbox(
                    label="Status",
                    value="Waiting for upload...",
                    interactive=False,
                    lines=1,
                )

                gr.Markdown("---")
                gr.Markdown("### Viewer Controls")
                slice_slider = gr.Slider(
                    minimum=0, maximum=1, step=1, value=0,
                    label="Axial Slice", interactive=False,
                )
                show_mask_cb = gr.Checkbox(
                    value=True, label="Show Liver Mask (red)"
                )
                show_unc_cb = gr.Checkbox(
                    value=True, label="Show Disagreement (yellow)"
                )

                gr.Markdown("---")
                gr.Markdown("### Export")
                download_btn = gr.Button("Download Mask (.nii.gz)", size="sm")
                mask_file = gr.File(label="Mask File", visible=False)

            # - Center column: Viewer -
            with gr.Column(scale=2, min_width=400):
                gr.Markdown("### 3D Slice Viewer")
                viewer = gr.Plot(label="CT + Overlay")

                gr.Markdown("### Explainability -- HU Histogram")
                histogram = gr.Plot(label="Liver Density Distribution")

            # - Right column: Report -
            with gr.Column(scale=2, min_width=350):
                gr.Markdown("### Computed Metrics")
                metrics_display = gr.Markdown(
                    value="*Run analysis to see metrics*"
                )

                gr.Markdown("### MedGemma Clinical Report")
                report_display = gr.Markdown(
                    value="*Run analysis to generate report*"
                )

        # -- Footer --
        gr.HTML(
            '<div style="text-align:center; padding:20px 0 10px; color:#555; font-size:0.85em;">'
            'This is a research prototype -- not for clinical diagnosis. '
            'Always consult a qualified radiologist.'
            '</div>'
        )

        # -- Wiring --
        run_btn.click(
            fn=run_pipeline,
            inputs=[file_input, tta_cb],
            outputs=[
                st_volume, st_mask, st_disagreement, st_nslices, st_metrics,
                slice_slider, viewer, histogram,
                metrics_display, report_display, status_box,
            ],
        )

        # Slice slider -> update viewer
        for trigger in [slice_slider.change, show_mask_cb.change, show_unc_cb.change]:
            trigger(
                fn=update_slice,
                inputs=[slice_slider, st_volume, st_mask, st_disagreement,
                        show_mask_cb, show_unc_cb],
                outputs=[viewer],
            )

        # Download mask
        download_btn.click(
            fn=download_mask,
            inputs=[st_mask, st_volume],
            outputs=[mask_file],
        )

    return demo


# --- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
