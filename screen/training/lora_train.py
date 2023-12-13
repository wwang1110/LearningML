from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
from transformers import AutoTokenizer
from transformers import CLIPProcessor

from screen_model import ScreenModel, ScreenConfiguration
from screen_dataset import ScreenDataset
from example.helper import print_trainable_parameters

trainable_layers=[
        "fc1", 
        "fc2", 
        "fc3", 
        "metadata_projection", 
        "clip_text_projection", 
        "clip_vision_projection", 
        "k_proj", 
        "v_proj", 
        "q_proj",
        "query",
        "key",
        "value",
        "dense",
    ]

def train_lora(adapter_name):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=trainable_layers,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )

    config = ScreenConfiguration()
    base_model = ScreenModel.from_pretrained("./base")
    base_model = prepare_model_for_kbit_training(base_model)

    model = get_peft_model(base_model, lora_config)
    print_trainable_parameters(model)

    roberta_tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_name)
    clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    dataset = ScreenDataset(dataset_path='D:/Adams/dataset/CUB_200_2011_CAP', clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)

    train_size = int(0.8* len(dataset))
    valid_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, valid_size])

    training_args = TrainingArguments(
            output_dir = config.finetune_output_dir,
            num_train_epochs = config.finetune_epochs,
            per_device_train_batch_size = config.finetune_batch_size,
            per_device_eval_batch_size = config.finetune_batch_size,
            evaluation_strategy = config.finetune_evaluation_strategy,
            optim = config.finetune_optim,
            report_to = 'none',
            save_strategy = 'no',
            fp16 = True,
            remove_unused_columns=False,
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset
        )
    trainer.train()

    results = trainer.evaluate()
    print(results)

    model.save_pretrained(f'./{adapter_name}')

train_lora("lora1")
train_lora("lora2")