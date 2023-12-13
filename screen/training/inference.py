import torch
from peft import PeftModel
from screen_model import ScreenModel, ScreenConfiguration
from transformers import AutoTokenizer
from transformers import CLIPProcessor

from PIL import Image
import albumentations as A
import numpy as np

def get_transforms(img_size):
    return A.Compose(
        [
            A.ToRGB(),
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )

def build_inputs(image_file, text_file, labels, transforms, clip_processor, roberta_tokenizer):
    item = {}
    pil_image = Image.open(image_file, 'r')
    image = transforms(image=np.array(pil_image))['image']
    image = torch.tensor(image).permute(2, 0, 1).float()
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)

    item['input_ids'] = inputs['input_ids']
    item['attention_mask'] = inputs['attention_mask']
    item['pixel_values'] = inputs['pixel_values']

    with open(text_file, 'r') as f:
        metadata = f.read()
    encoded_metadata = roberta_tokenizer(metadata, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    item['encoded_metadata'] = encoded_metadata['input_ids']
    item['metadata_attention_mask'] = encoded_metadata['attention_mask']

    return item   

config = ScreenConfiguration()
base_model = ScreenModel.from_pretrained("./base")
model = PeftModel.from_pretrained(base_model, './lora1', adapter_name="adapter1")
model.load_adapter('./lora2', adapter_name="adapter2")

roberta_tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_name)
clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)

image_file = 'D:/Adams/dataset/CUBTest_val/Acadian_Flycatcher_0014_795607.jpg'
text_file = 'D:/Adams/dataset/CUBTest_val/Acadian_Flycatcher_0014_795607.txt'

labels = ['Acadian Flycatcher', 'Scarlet Tanager', 'Western Meadowlark']

inputs = build_inputs(image_file, text_file, labels, get_transforms(224), clip_processor, roberta_tokenizer)

model.set_adapter("adapter1")
for x in model.parameters():
  x.requires_grad = False
model.eval()
output = model(**inputs)
print(output)

model.set_adapter("adapter2")
for x in model.parameters():
  x.requires_grad = False
model.eval()
output = model(**inputs)
print(output)
