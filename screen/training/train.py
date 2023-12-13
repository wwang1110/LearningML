from screen_dataset import ScreenDataset
from screen_model import ScreenModel, ScreenConfiguration
from transformers import AutoTokenizer
from transformers import CLIPProcessor
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split

config = ScreenConfiguration()

roberta_tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_name)
clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
#train_dataset = ScreenDataset(dataset_path='D:/Adams/dataset/CUBTest_train', clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)
#val_dataset = ScreenDataset(dataset_path='D:/Adams/dataset/CUBTest_val', clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)

dataset = ScreenDataset(dataset_path='D:/Adams/dataset/CUB_200_2011_CAP', clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)

train_size = int(0.8* len(dataset))
valid_size = len(dataset) - train_size
train_subset, val_subset = random_split(dataset, [train_size, valid_size])

model = ScreenModel(config)

training_args = TrainingArguments(
        output_dir = config.output_dir,
        num_train_epochs = config.num_train_epochs,
        per_device_train_batch_size = config.batch_size,
        per_device_eval_batch_size = config.batch_size,
        evaluation_strategy = config.evaluation_strategy,
        optim = config.optim,
        report_to = 'none',
        save_strategy = 'no',
        fp16 = True,
        remove_unused_columns=False,
    )

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
    )

trainer.train()

results = trainer.evaluate()
print(results)

model.save_pretrained("./base")