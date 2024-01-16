import os
import pickle
import lmdb
import logging
import base64
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A
from transformers import DataCollatorWithPadding

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, clip_processor, roberta_tokenizer):
        self.lmdb_path = lmdb_path

        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), f"The LMDB directory {lmdb_path} does not exist!"
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        assert os.path.isdir(lmdb_pairs), f"The LMDB directory {lmdb_pairs} image-text pairs does not exist!"
        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), f"The LMDB directory {lmdb_imgs} base64 strings does not exist!"
        lmdb_metas = os.path.join(lmdb_path, "metas")
        assert os.path.isdir(lmdb_metas), f"The LMDB directory {lmdb_metas} metadata does not exist!"

        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)
        self.env_metas = lmdb.open(lmdb_metas, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_metas = self.env_metas.begin(buffers=True)        

        # fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'))
        logging.info(f"LMDB file contains {self.number_images} images and {self.number_samples} pairs.")

        super(LMDBDataset, self).__init__()

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        self.max_token_length = 77
        self.eos_token_id = 49407
                
        self.clip_processor = clip_processor
        self.roberta_tokenizer = roberta_tokenizer

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()
        if hasattr(self, 'env_metas'):
            self.env_metas.close()            

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        item = {}

        sample_index = index % self.number_samples

        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode('utf-8')).tobytes())
        image_id, text_id, raw_text = pair

        image_b64 = self.txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        pil_image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))) # already resized

        encoded_clip_inputs = self.clip_processor(text=[raw_text], images=pil_image, return_tensors="pt", truncation=True)
        metadata = self.txn_metas.get("{}".format(image_id).encode('utf-8')).tobytes().decode('utf-8')
        encoded_metadata = self.roberta_tokenizer(metadata, return_tensors="pt", truncation=True)

        item['metadata_input_ids'] = encoded_metadata['input_ids'][0]
        item['metadata_attention_mask'] = encoded_metadata['attention_mask'][0]

        item['clip_pixel_values'] = encoded_clip_inputs['pixel_values'][0]
        item['clip_input_ids'] = encoded_clip_inputs['input_ids'][0]
        item['clip_attention_mask'] = encoded_clip_inputs['attention_mask'][0]

        return item
    
def lmdb_collate_function(examples, clip_tokenizer, roberta_tokenizer):
    pixel_values = torch.stack([example["clip_pixel_values"] for example in examples])
    
    metadata_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)
    metadata_inputs = [{"input_ids": example["metadata_input_ids"], "attention_mask": example["metadata_attention_mask"]} for example in examples]
    metadata_data = metadata_collator(metadata_inputs)
    
    clip_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)
    clip_inputs = [{"input_ids": example["clip_input_ids"], "attention_mask": example["clip_attention_mask"]} for example in examples]
    clip_data = clip_collator(clip_inputs)
    #input_ids = torch.stack([example["input_ids"] for example in examples])
    #labels = torch.stack([example["input_ids"] for example in examples])
    return {
        "metadata_input_ids": metadata_data['input_ids'],
        "metadata_attention_mask": metadata_data['attention_mask'],
        "clip_pixel_values": pixel_values,
        "clip_input_ids": clip_data['input_ids'],
        "clip_attention_mask": clip_data['attention_mask'],
        }