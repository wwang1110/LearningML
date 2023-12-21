import torch
from peft import PeftModel
from screen_model import ScreenModel, ScreenConfiguration
from transformers import AutoTokenizer
from transformers import CLIPProcessor
from training.helper import build_inputs, get_transforms
from transformers import BitsAndBytesConfig

inf_nf4_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float32,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4'
)

config = ScreenConfiguration()
base_model = ScreenModel.from_pretrained("./base",
                                load_in_4bit=True,
                                torch_dtype=torch.float32,
                                quantization_config=inf_nf4_config)

model = PeftModel.from_pretrained(base_model, './lora1', adapter_name="adapter1")
model.load_adapter('./lora2', adapter_name="adapter2")
model.to("cuda:0")

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
