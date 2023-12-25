from screen_dataset.lmdb_dataset import LMDBDataset
from screen_model import ScreenModel, ScreenConfiguration
from transformers import AutoTokenizer
from transformers import CLIPProcessor
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
from helper import compute_metrics, compute_metrics_adjusted

config = ScreenConfiguration()

roberta_tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_name)
clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)

dataset = LMDBDataset(lmdb_path=config.lmdb_path, clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)

train_size = int(0.8* len(dataset))
valid_size = len(dataset) - train_size
train_subset, val_subset = random_split(dataset, [train_size, valid_size])

model = ScreenModel(config)

training_args = TrainingArguments(
        output_dir = config.checkpoint_dir,
        num_train_epochs = config.num_train_epochs,
        per_device_train_batch_size = config.batch_size,
        per_device_eval_batch_size = config.batch_size,
        evaluation_strategy = config.evaluation_strategy,
        eval_steps = config.eval_steps,
        logging_strategy = config.logging_strategy,
        logging_dir = config.logging_dir,
        logging_steps = config.logging_steps,
        report_to = config.report_to,
        save_strategy = config.save_strategy,
        save_steps = config.save_steps,
        save_total_limit = config.save_total_limit,
        optim = config.optim,
        learning_rate = config.learning_rate,
        #dataloader_num_workers = config.dataloader_num_workers,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        fp16 = True,
        remove_unused_columns=False,
    )

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        compute_metrics=compute_metrics_adjusted,
    )

#predictions = trainer.predict(val_subset)

trainer.train()

results = trainer.evaluate()
print(results)

model.save_pretrained(config.output_dir)

#resume checkpoint
#trainer.train("checkpoint-9500")