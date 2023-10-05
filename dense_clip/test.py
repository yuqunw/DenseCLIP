from pathlib import Path
import os

import torch
from dense_clip import DenseCLIP
import urllib

from PIL import Image

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"

image = load_image_from_url(EXAMPLE_IMAGE_URL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseCLIP('ViT-L/14@336px').to(device)
model.eval()


with torch.no_grad():
    feature = model(image)

