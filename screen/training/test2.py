import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers import PretrainedConfig, PreTrainedModel

class MLPConfig(PretrainedConfig):
    model_type = "screen_mlp"

    def __init__(
        self,
        num_features: int = 256,
        num_classes: int = 3,
        **kwargs,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        super().__init__(**kwargs)

class MLP(PreTrainedModel):
    config_class = MLPConfig

    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.num_features, config.num_features)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(config.num_features, config.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels):
        x = self.fc1(input_ids)
        x = self.activation(x)
        logits = self.fc2(x)
        loss = self.criterion(logits, labels)
        return {'loss': loss, 'logits': logits}  

class MLPDataset(torch.utils.data.Dataset):

    def __init__(self, len):
        self.len = len

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.rand(20).float()
        item['labels'] = torch.rand(3).float()
        #item['labels'] = torch.Tensor(3)
        return item

    def __len__(self):
        return self.len

train_dataset = MLPDataset(256)
val_dataset = MLPDataset(128)

training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to='none',
        save_strategy='no',
        remove_unused_columns=False
    )

mlp_config = MLPConfig(20, 3)
model = MLP(mlp_config)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

trainer.train()

trainer.save_model("./output")

model = MLP.from_pretrained("./output")