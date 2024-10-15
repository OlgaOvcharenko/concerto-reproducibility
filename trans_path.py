from urllib.request import urlopen
from PIL import Image
import timm
import numpy as np


# get example histology image
img = Image.open(
  'staining_examples/your_fileajbkcoho-1.jpeg',
#   urlopen(
#     "https://github.com/owkin/HistoSSLscaling/raw/main/assets/example.tif"
#   )
)
img = np.array(img)
img = Image.fromarray(img)
print(img)

# load model from the hub
model = timm.create_model(
  model_name="hf-hub:1aurent/vit_small_patch16_224.transpath_mocov3",
  pretrained=True,
  num_heads=12,
).eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

data = transforms(img).unsqueeze(0)  # input is (batch_size, num_channels, img_size, img_size) shaped tensor
output = model(data)  # output is (batch_size, num_features) shaped tensor

print(output)
print(output.shape)
