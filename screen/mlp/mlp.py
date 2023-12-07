import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, d_in)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_in, d_out)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels):
        x = self.fc1(input_ids)
        x = self.activation(x)
        logits = self.fc2(x)
        loss = self.criterion(logits, labels)
        return {'loss': loss, 'logits': logits}