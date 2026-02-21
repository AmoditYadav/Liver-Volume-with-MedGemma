"""
Stage C — Radiologist Agent: Medical Report Generation.

Uses the local MedGemma 8B model to synthesize a structured clinical note
from the computed liver metrics.  The model is loaded once and cached.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH = r"D:\Monai\model_medgemma"


# ─── Model singleton ─────────────────────────────────────────────────────────

_model = None
_processor = None


def _get_model():
    """Lazy-load MedGemma 8B for text generation (cached)."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[LiverLens] Loading MedGemma 8B for report generation...")

    _processor = AutoProcessor.from_pretrained(MODEL_PATH)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    print("[LiverLens] MedGemma 8B ready")
    return _model, _processor


# ─── Prompt Builder ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert radiologist specializing in abdominal CT imaging. "
    "Given automated liver CT measurements, write a structured clinical note "
    "with EXACTLY these three sections:\n\n"
    "Findings\n"
    "Impression\n"
    "Recommendation\n\n"
    "Use precise medical terminology. Reference normal ranges. "
    "Do NOT fabricate data. Only reference the numbers provided. "
    "Do NOT include any thinking or reasoning. Go straight to the report."
)


def _build_prompt(metrics: dict) -> str:
    flags = []
    if metrics.get("steatosis_flag"):
        flags.append("WARNING: STEATOSIS RISK -- Mean density below 40 HU")
    if metrics.get("hepatomegaly_flag"):
        flags.append("WARNING: HEPATOMEGALY RISK -- Volume exceeds 1800 cc")

    flags_str = "\n".join(flags) if flags else "No immediate flags."

    return (
        f"Automated Liver CT Measurements:\n"
        f"Liver Volume:       {metrics['volume_cc']:.1f} cc  "
        f"(Normal: 1200-1600 cc)\n"
        f"Mean Density:       {metrics['mean_hu']:.1f} HU  "
        f"(Normal: >55 HU; <40 HU suggests steatosis)\n"
        f"Density Std Dev:    {metrics['std_hu']:.1f} HU  "
        f"(High values suggest heterogeneity / focal lesion)\n"
        f"Median Density:     {metrics['median_hu']:.1f} HU\n"
        f"10th Percentile:    {metrics['p10_hu']:.1f} HU\n"
        f"90th Percentile:    {metrics['p90_hu']:.1f} HU\n"
        f"Voxels < 40 HU:    {metrics['low_frac_40']*100:.1f}%\n"
        f"Voxels < 50 HU:    {metrics['low_frac_50']*100:.1f}%\n"
        f"Total Liver Voxels: {metrics['n_liver_voxels']:,}\n"
        f"\nRisk Flags:\n{flags_str}\n"
        f"\nWrite a structured clinical note with Findings, Impression, "
        f"and Recommendation sections. Start directly with the Findings."
    )


# ─── Post-processing ────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove Gemma3 thinking/reasoning prefix if present."""
    import re
    # Gemma3 may prefix output with internal thinking blocks
    # Look for the start of the actual report content
    for marker in ["## Findings", "**Findings**", "Findings\n", "FINDINGS"]:
        idx = text.find(marker)
        if idx != -1:
            return text[idx:]
    # If no marker found, try stripping everything before first ##
    idx = text.find("##")
    if idx > 0:
        return text[idx:]
    # Last resort: strip known thinking prefixes
    text = re.sub(r'^.*?(?:thought|Thinking Process:).*?\n', '', text,
                  flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_report(metrics: dict) -> str:
    """
    Generate a clinical report from liver metrics using MedGemma 8B.

    Parameters
    ----------
    metrics : dict from liverlens.metrics.compute_metrics()

    Returns
    -------
    str -- markdown-formatted clinical note
    """
    model, processor = _get_model()

    user_prompt = _build_prompt(metrics)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

    # Decode only the generated tokens (skip the input)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = processor.tokenizer.decode(generated, skip_special_tokens=True)

    # Strip any thinking/reasoning prefix
    report = _strip_thinking(raw_text)

    if not report.strip():
        # Fallback if model produced only thinking tokens
        report = (
            "## Findings\n"
            f"Liver volume: {metrics['volume_cc']:.1f} cc. "
            f"Mean density: {metrics['mean_hu']:.1f} HU. "
            f"Std deviation: {metrics['std_hu']:.1f} HU.\n\n"
            "## Impression\n"
            + ("Findings suggest possible hepatic steatosis.\n\n"
               if metrics.get("steatosis_flag")
               else "No significant abnormality detected.\n\n")
            + "## Recommendation\n"
            "Clinical correlation recommended."
        )

    return report.strip()
