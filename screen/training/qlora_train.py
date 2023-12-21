import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
from transformers import AutoTokenizer
from transformers import CLIPProcessor

from screen_model import ScreenModel, ScreenConfiguration
from screen_dataset.lmdb_dataset import LMDBDataset
#from screen_dataset.file_dataset import FileDataset
from training.helper import print_trainable_parameters

def train_qlora(adapter_name, base_model_path, dataset, config):
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

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=trainable_layers,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )

    base_model = ScreenModel.from_pretrained(base_model_path, quantization_config=nf4_config)
    base_model = prepare_model_for_kbit_training(base_model)

    model = get_peft_model(base_model, lora_config)
    print_trainable_parameters(model)

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

if __name__ == "__main__":
    screen_config = ScreenConfiguration()
    roberta_tokenizer = AutoTokenizer.from_pretrained(screen_config.roberta_model_name)
    clip_processor = CLIPProcessor.from_pretrained(screen_config.clip_model_name)
    dataset = LMDBDataset(lmdb_path='D:/Adams/lmdb', clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)
    train_qlora("lora1", "./base", dataset, screen_config)
    train_qlora("lora2", "./base", dataset, screen_config)