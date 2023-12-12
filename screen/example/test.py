import torch
from peft import PeftModel

from example.mlp import MLP
from example.helper import print_model_info

base_model = MLP.from_pretrained("./base")

model = PeftModel.from_pretrained(base_model, './lora1', adapter_name="adapter1")
model.load_adapter('./lora2', adapter_name="adapter2")


model.set_adapter("adapter1")
inputs1 = {'input_ids' : torch.rand(2, 20).float()}
for x in model.parameters():
  x.requires_grad = False
model.eval()
output = model(**inputs1)
print(output)

model.set_adapter("adapter2")
inputs2 = {'input_ids' : torch.rand(2, 20).float()}
for x in model.parameters():
  x.requires_grad = False
model.eval()
output = model(**inputs2)
print(output)