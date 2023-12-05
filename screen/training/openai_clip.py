from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from torch import nn

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img_path = 'D:/Adams/dataset/CUBTest/Acadian_Flycatcher_0003_29094.jpg'
image = Image.open(img_path)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

text_embeds = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])

logit_scale_init_value = 2.6592
logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

# normalized features
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# cosine similarity as logits
logit_scale = logit_scale.exp()
logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
logits_per_image = logits_per_text.t()
print (logits_per_text)
#loss = clip_loss(logits_per_text)
#print(loss)

outputs = model(**inputs)
print(outputs.logits_per_text)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities