from torch import nn
from collections import OrderedDict

class MLPHeader(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
        super().__init__()
        self.network = nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(embedding_dim, embedding_dim)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(embedding_dim, projection_dim)),
          ('relu2', nn.ReLU()),
          ('dropout', nn.Dropout(dropout)),
          ('fc3', nn.Linear(projection_dim, projection_dim)),
          ('ln', nn.LayerNorm(projection_dim))
        ]))        

    def forward(self, x):
        return self.network(x)