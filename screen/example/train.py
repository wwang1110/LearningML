from transformers import Trainer, TrainingArguments

from example.mlp_config import MLPConfig
from example.mlp import MLP
from example.mlp_dataset import MLPDataset

train_dataset = MLPDataset(256)
val_dataset = MLPDataset(128)

training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to='none',
        save_strategy='no',
        fp16=True,
        optim="adamw_torch",
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
model.save_pretrained("./base")