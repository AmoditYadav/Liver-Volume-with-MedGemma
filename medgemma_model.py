"""
MedGemma Segmentation Model: SigLIP Vision Encoder + Segmentation Decoder.

Extracts the medically pre-trained SigLIP vision encoder from MedGemma 1.5 4B
and adds a lightweight UNet-style decoder for pixel-level liver segmentation.
Only the decoder is trained; the encoder stays frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoConfig
import math

class DecoderBlock(nn.Module):
    """Upsampling decoder block: upsample → concat skip → conv → conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SimpleDecoder(nn.Module):
    """Decoder without skip connections — upsamples from patch grid to full image."""
    def __init__(self, in_ch, out_ch=1, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        self.decode = nn.Sequential(
            nn.Conv2d(in_ch, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(32, out_ch, kernel_size=1)
        
    def forward(self, x):
        x = self.decode(x)
        if x.shape[2] != self.target_size[0] or x.shape[3] != self.target_size[1]:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        x = self.head(x)
        return x

class MedGemmaSegModel(nn.Module):
    """
    Segmentation model using MedGemma's SigLIP vision encoder.
    
    Architecture:
        Input (3×224×224) → SigLIP Encoder (frozen) → Patch Embeddings
        → Reshape to spatial grid → Decoder → Binary Mask (1×224×224)
    
    Args:
        model_path: Path to local MedGemma model directory
        img_size: Input image size (default: 224)
        freeze_encoder: Whether to freeze the vision encoder (default: True)
    """
    def __init__(self, model_path, img_size=224, freeze_encoder=True):
        super().__init__()
        self.img_size = img_size
        print('Loading MedGemma vision encoder...')
        config = AutoConfig.from_pretrained(model_path)
        vision_config = config.vision_config
        self.hidden_size = vision_config.hidden_size
        self.patch_size = vision_config.patch_size
        self.orig_img_size = vision_config.image_size
        self.grid_size = self.img_size // self.patch_size
        
        full_model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        self.vision_encoder = full_model.vision_tower
        del full_model
        torch.cuda.empty_cache()
        
        print(f'Vision encoder loaded: hidden_size={self.hidden_size}, patch_size={self.patch_size}, grid_size={self.grid_size}')
        if freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()
            print('Vision encoder frozen.')
        
        self.decoder = SimpleDecoder(in_ch=self.hidden_size, out_ch=1, target_size=(self.img_size, self.img_size))
        
        n_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"Decoder parameters: {n_params:,} ({n_params / 1000000.0:.1f}M)")

    def encode(self, x):
        context = torch.enable_grad() if self.vision_encoder.training else torch.no_grad()
        with context:
            vision_outputs = self.vision_encoder(pixel_values=x, interpolate_pos_encoding=True)
            hidden_states = vision_outputs.last_hidden_state
        B, N, D = hidden_states.shape
        grid_h = grid_w = int(math.sqrt(N))
        features = hidden_states.permute(0, 2, 1).reshape(B, D, grid_h, grid_w)
        return features

    def forward(self, x):
        features = self.encode(x)
        logits = self.decoder(features)
        return logits

def build_model(model_path, img_size=224, device='cuda'):
    model = MedGemmaSegModel(model_path, img_size=img_size)
    model = model.to(device)
    return model

if __name__ == '__main__':
    print('Testing MedGemma Segmentation Model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(r'D:\Monai\model_medgemma', img_size=224, device=device)
    
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f'Input:  {dummy_input.shape}')
    print(f'Output: {output.shape}')
    print(f'Output range: [{output.min():.4f}, {output.max():.4f}]')
    print('✓ Model test passed!')
