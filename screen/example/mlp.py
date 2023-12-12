import torch.nn as nn
from collections import OrderedDict
from transformers import PreTrainedModel
from example.mlp_config import MLPConfig


class MLP(PreTrainedModel):
    config_class = MLPConfig

    def __init__(self, config):
        super().__init__(config)

        self.network = nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(config.num_features, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 512)),
          ('relu2', nn.ReLU()),
          ('fc3', nn.Linear(512, 256)),
          ('relu3', nn.ReLU()),
          ('fc4', nn.Linear(256, config.num_classes))
        ]))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        logits = self.network(input_ids)
        if labels is None:
          return {'logits': logits}
        else:
          loss = self.criterion(logits, labels)
          return {'loss': loss, 'logits': logits}