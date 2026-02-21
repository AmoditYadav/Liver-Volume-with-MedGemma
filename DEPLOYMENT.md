# 🚀 Deploying Liver Segmentation AI to Hugging Face Spaces

## Prerequisites
1. A [Hugging Face account](https://huggingface.co/join)
2. [Git](https://git-scm.com/downloads) installed
3. [Git LFS](https://git-lfs.github.com/) installed (for large model files)

---

## Step-by-Step Deployment (CPU - Free Tier)

### 1️⃣ Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Owner**: Your username
   - **Space name**: `liver-segmentation-ai`
   - **License**: MIT
   - **SDK**: Select **Gradio**
   - **Hardware**: **CPU basic** (free)
3. Click **Create Space**

---

### 2️⃣ Clone the Space Repository

```powershell
# Clone your new Space (replace YOUR_USERNAME with your HF username)
git clone https://huggingface.co/spaces/YOUR_USERNAME/liver-segmentation-ai
cd liver-segmentation-ai
```

---

### 3️⃣ Copy Required Files

Copy these files from `D:\Monai` to the cloned Space folder:

```powershell
# Required app files
copy D:\Monai\app.py .
copy D:\Monai\requirements.txt .
copy D:\Monai\README.md .

# Copy BOTH model weights (rename for clarity)
copy D:\Monai\results\liver_swinunetr\best_metric_model.pth .\swinunetr_model.pth
copy D:\Monai\results\liver_unet_baseline\best_metric_model.pth .\unet_model.pth
```

Your Space folder should now contain:
```
liver-segmentation-ai/
├── README.md              # HF Spaces config (YAML header)
├── app.py                 # Gradio app
├── requirements.txt       # Dependencies
├── swinunetr_model.pth    # SwinUNETR weights (~245MB)
└── unet_model.pth         # UNet weights (~73MB)
```

---

### 4️⃣ Set Up Git LFS for Large Files

Model files are large, so we need Git LFS:

```powershell
# Initialize Git LFS
git lfs install

# Track .pth files with LFS
git lfs track "*.pth"

# Add the .gitattributes file
git add .gitattributes
```

---

### 5️⃣ Commit and Push

```powershell
# Add all files
git add .

# Commit
git commit -m "Deploy: Liver Segmentation AI with SwinUNETR and UNet comparison"

# Push to Hugging Face (this may take a few minutes for model files)
git push
```

**Credentials:**
- **Username**: Your HF username
- **Password**: Your HF access token (create at [hf.co/settings/tokens](https://huggingface.co/settings/tokens))

---

### 6️⃣ Wait for Build (~5-10 minutes)

1. Go to: `https://huggingface.co/spaces/YOUR_USERNAME/liver-segmentation-ai`
2. Click the **"Logs"** tab to monitor build progress
3. Wait for dependencies to install
4. Your app will be live! 🎉

---

## 📁 Ground Truth Files for Testing

Your ground truth segmentation masks are in:
```
D:\Monai\LiverData\TestSegmentation\
├── liver_104.nii.gz
├── liver_105.nii.gz
├── ... (27 files)
└── liver_130.nii.gz
```

Corresponding CT volumes are in:
```
D:\Monai\LiverData\TestVolumes\
├── liver_104.nii
├── liver_105.nii
├── ... 
└── liver_130.nii
```

**To test:** Upload a CT volume AND its matching segmentation to see Dice scores!

---

## ⚠️ Troubleshooting

### "Model file too large" Error
```powershell
# Verify LFS is tracking model files
git lfs ls-files
# Should show: swinunetr_model.pth, unet_model.pth
```

### Build Timeout / Out of Memory
The free CPU tier has limited resources. If builds fail:
- Try uploading just one model first (comment out the other in app.py)
- Consider upgrading to paid GPU tier for better performance

### Models Not Loading
Check that file names match exactly:
- `swinunetr_model.pth` (not `best_metric_model.pth`)
- `unet_model.pth`

---

## 🔄 Updating Your Space

After making changes locally:
```powershell
git add .
git commit -m "Update: description of changes"
git push
```

---

## 📊 Expected Results

With proper ground truth, you should see Dice scores like:
- **SwinUNETR**: ~0.94-0.97 (94-97%)
- **UNet**: ~0.90-0.94 (90-94%)

The Vision Transformer (SwinUNETR) typically outperforms CNN (UNet) by 2-5% Dice score!

---

## 🎮 Quick Test Commands

After deployment, you can test locally first:

```powershell
# Activate your venv
cd D:\Monai
.\.venv\Scripts\activate

# Run locally
python app.py
```

Open http://localhost:7860 to test before deploying.
