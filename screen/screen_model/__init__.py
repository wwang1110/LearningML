import torch
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from screen_model.screen_configuration import ScreenConfiguration
from screen_model.mlp_header import MLPHeader

from transformers import CLIPModel
from transformers import XLMRobertaModel
#from transformers import PreTrainedModel, PreTrainedTokenizer

class ScreenModel(PreTrainedModel):
    config_class = ScreenConfiguration

    def __init__(self, config):
        super().__init__(config)
        self.temperature = config.temperature

        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
        for p in self.clip_model.parameters():
            p.requires_grad = config.clip_trainable
        self.metadata_encoder = XLMRobertaModel.from_pretrained(config.roberta_model_name)
        for p in self.metadata_encoder.parameters():
            p.requires_grad = config.roberta_trainable

        self.metadata_projection = nn.Linear(config.roberta_txt_dim, config.projection_dim, bias=False)
        self.clip_text_projection = nn.Linear(config.clip_txt_dim, config.projection_dim, bias=False)
        self.clip_vision_projection = nn.Linear(config.clip_vit_dim, config.projection_dim, bias=False)
        self.mlp = MLPHeader(embedding_dim=config.projection_dim * 2, projection_dim=config.projection_dim, dropout=config.dropout)

    def get_text_features(self, input_ids, attention_mask):
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        text_embeds = text_outputs[1]
        return self.clip_text_projection(text_embeds)

    def get_image_features(self, pixel_values, encoded_metadata, metadata_attention_mask):
        #extract image embeds
        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        image_embeds = vision_outputs[1]
        image_feature = self.clip_vision_projection(image_embeds)
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

    def forward(self, input_ids, attention_mask, pixel_values, encoded_metadata, metadata_attention_mask, labels=None):
        # Getting Image and Text Features
        image_embeddings = self.get_image_features(pixel_values, encoded_metadata, metadata_attention_mask)
        text_embeddings = self.get_text_features(input_ids, attention_mask)

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        pred = torch.argmax(logits, dim=1).tolist()

        if labels is None:
            return {'logits': logits, 'pred': pred}
        else:
            # Calculating the Loss
            images_similarity = image_embeddings @ image_embeddings.T
            texts_similarity = text_embeddings @ text_embeddings.T
            targets = F.softmax(
                (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
            )
            texts_loss = cross_entropy(logits, targets, reduction='none')
            images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
            return {'loss': loss.mean(), 'logits': logits, 'pred': pred}

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()