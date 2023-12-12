from screen_dataset import ScreenDataset
from screen_model import ScreenModel, ScreenConfiguration
from transformers import AutoTokenizer
from transformers import CLIPProcessor
from transformers import Trainer, TrainingArguments

config = ScreenConfiguration()

roberta_tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_name)
clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
train_dataset = ScreenDataset(dataset_path='D:/Adams/dataset/CUBTest_train', clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)
val_dataset = ScreenDataset(dataset_path='D:/Adams/dataset/CUBTest_val', clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)

train_dataset[10]

model = ScreenModel(config)

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

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

trainer.train()
model.save_pretrained("./base")