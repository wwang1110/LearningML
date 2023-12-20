from screen_dataset.lmdb_dataset import LMDBDataset
from screen_model import ScreenModel, ScreenConfiguration
from transformers import AutoTokenizer
from transformers import CLIPProcessor
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split

config = ScreenConfiguration()

roberta_tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_name)
clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)

dataset = LMDBDataset(lmdb_path='D:/Adams/lmdb', clip_processor=clip_processor, roberta_tokenizer=roberta_tokenizer)

x = dataset[10]
print(x)