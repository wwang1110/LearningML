import torch
from torch import FloatTensor, nn

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

    def get_text_features(self, clip_input_ids, clip_attention_mask):
        text_outputs = self.clip_model.text_model(
            input_ids=clip_input_ids,
            attention_mask=clip_attention_mask,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        text_embeds = text_outputs[1]
        return self.clip_text_projection(text_embeds)

    def get_image_features(self, clip_pixel_values, metadata_input_ids, metadata_attention_mask):
        #extract image embeds
        vision_outputs = self.clip_model.vision_model(
            pixel_values=clip_pixel_values,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        image_embeds = vision_outputs[1]
        image_feature = self.clip_vision_projection(image_embeds)
        image_embeds = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)

        #extract metadata embeds
        metadata_output = self.metadata_encoder(input_ids=metadata_input_ids, attention_mask=metadata_attention_mask)
        metadata_feature = metadata_output[1]
        metadata_feature = self.metadata_projection(metadata_feature)
        metadata_embeds = metadata_feature / metadata_feature.norm(p=2, dim=-1, keepdim=True)
        
        #concat image and metadata embeds
        embeds = torch.cat((image_embeds, metadata_embeds), dim=1)

        #mlp layer
        return self.mlp(embeds)

    def forward(self, metadata_input_ids, metadata_attention_mask, clip_pixel_values, clip_input_ids, clip_attention_mask):
        # Getting Image and Text Features
        image_embeds = self.get_image_features(clip_pixel_values, metadata_input_ids, metadata_attention_mask)
        text_embeds = self.get_text_features(clip_input_ids, clip_attention_mask)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        #logits_per_text = self.adjust_logits(clip_input_ids, logits_per_text)
        logits_per_image = logits_per_text.t()
        probs = logits_per_image.softmax(dim=1)

        # Calculating the Loss
        loss = clip_loss(logits_per_text)
        return {'loss': loss, 'logits_per_image': logits_per_image, 'logits_per_text': logits_per_text, 'probs': probs}

    def adjust_logits(self, input_ids, logits_per_text) -> FloatTensor:
        input_map={}
        for i in range(len(input_ids)):
            key = ','.join([str(x.item()) for x in input_ids[i]])
            input_map.setdefault(key, [])
            input_map[key].append(i)

        for v in input_map.values():
            for i in v:
                for j in v:
                    if i != j:
                        logits_per_text[j][i] = 0.0
        return logits_per_text

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0