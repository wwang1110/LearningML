from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer, TrainingArguments

from example.mlp import MLP
from example.helper import print_model_info
from example.mlp_dataset import MLPDataset

def train_lora(adapter_name):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["fc1", "fc2", "fc3", "fc4"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )

    base_model = MLP.from_pretrained("./base")
    base_model = prepare_model_for_kbit_training(base_model)
    print_model_info(base_model)

    model1 = get_peft_model(base_model, lora_config)
    print_model_info(model1)

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
            remove_unused_columns=False
        )

    trainer1 = Trainer(
            model=model1,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
    trainer1.train()
    model1.save_pretrained(f'./{adapter_name}')

train_lora("lora1")
train_lora("lora2")