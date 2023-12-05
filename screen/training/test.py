
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
import torch

print(torch.cuda.is_available())

model_name = "openai/clip-vit-base-patch32"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = CLIPModel.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
processor = CLIPProcessor.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
model.config.use_cache = False
