# Light-weight wrapper repository for running MaskCLIP

# License
The license for the model and the weights are entirely attributed to the original authors of the [OpenAI CLIP](https://github.com/OpenAI/clip) and [MaskCLIP](https://github.com/chongzhou96/MaskCLIP)

# Installation
```bash
pip install git+http://github.com/leejaeyong7/DenseCLIP
```

# Example Usage
```python
from dense_clip import DenseCLIP
from PIL import Image

device = torch.device('cuda')

model = DenseCLIP('ViT-L/14@336px').to(device)
model.eval()

image = Image.open(image_file)
image.resize((336, 336))

with torch.no_grad():
    feature = model(image) # HxWxC feature for image size of HxW
```