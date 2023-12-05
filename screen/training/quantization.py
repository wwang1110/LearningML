#only works for linux

import torch
from transformers import CLIPProcessor, CLIPModel, BitsAndBytesConfig
import torch

print(torch.cuda.is_available())

#model_name = "openai/clip-vit-base-patch32"
model_name = "openai/clip-vit-large-patch14"

'''
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
'''
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
#model = CLIPModel.from_pretrained(model_name, quantization_config=bnb_config)
#processor = CLIPProcessor.from_pretrained(model_name, quantization_config=bnb_config)
#model.config.use_cache = False

def model_size_in_MB(model):
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))  

model = CLIPModel.from_pretrained(model_name, quantization_config=bnb_config)
model_size_in_MB(model)

model = CLIPModel.from_pretrained(model_name)
model_size_in_MB(model)