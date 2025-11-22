from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm

# === Constantes del entrenamiento ===
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Debe coincidir con tu código:
MORPH_COLS = [
    'area_px','perim_px','bbox_w','bbox_h',
    'aspect','hu1','hu2','hu3','fill_frac'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class BackboneTIMM(nn.Module):
    def __init__(self, model_name='wide_resnet50_2', trainable=False):
        super().__init__()
        m = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = m
        self.out_dim = m.num_features  # 2048 para wide_resnet50_2
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)
