import torch
from torch import nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .metadata_encoder import MetadataEncoder
from .mlp_header import MLPHeader
from transformers import CLIPProcessor, CLIPModel
from transformers import RobertaModel, RobertaTokenizer
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

clip_models = [
    #projection_dim = 512
    "openai/clip-vit-base-patch32", 
    #projection_dim = 768
    "openai/clip-vit-large-patch14"
    ]
xlm_roberta_models = [
    "xlm-roberta-base", 
    "xlm-roberta-large"
    ]

class ScreenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.metadata_encoder = MetadataEncoder()
        self.temperature = config.temperature

        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
        for p in self.clip_model.parameters():
            p.requires_grad = config.trainable
        self.metadata_encoder = RobertaModel.from_pretrained(config.metadata_model_name)
        for p in self.metadata_encoder.parameters():
            p.requires_grad = config.trainable

        self.projection_dim = config.projection_dim
        self.clip_dim = config.clip_dim
        self.roberta_dim = config.roberta_dim
        self.metadata_projection = nn.Linear(self.roberta_dim, self.projection_dim, bias=False)
        self.clip_text_projection = nn.Linear(self.clip_dim, self.projection_dim, bias=False)
        self.clip_vision_projection = nn.Linear(self.clip_dim, self.projection_dim, bias=False)
        self.mlp = MLPHeader(embedding_dim=config.clip_dim, projection_dim=config.projection_dim, dropout=config.dropout)

    def get_text_features(self, input_ids, attention_mask: Optional[torch.Tensor] = None):
        text_feature = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return self.clip_text_projection(text_feature)

    def get_image_features(self, pixel_values, encoded_metadata, metadata_attention_mask):
        #extract image embeds
        image_feature = self.clip_model.get_image_features(pixel_values=pixel_values)
        image_feature = self.clip_vision_projection(image_feature)
        image_embeds = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)

        #extract metadata embeds
        metadata_output = self.metadata_encoder(input_ids=encoded_metadata, attention_mask=metadata_attention_mask)
        metadata_feature = metadata_output[1]
        metadata_feature = self.metadata_projection(metadata_feature)
        metadata_embeds = metadata_feature / metadata_feature.norm(p=2, dim=-1, keepdim=True)
        
        #concat image and metadata embeds
        embeds = torch.cat((image_embeds, metadata_embeds), dim=1)

        #mlp layer
        return self.mlp(embeds)

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)

        pred = torch.argmax(logits, dim=1).tolist()
        target = [x for x in range(len(pred))]
        return loss.mean(), pred, target

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()