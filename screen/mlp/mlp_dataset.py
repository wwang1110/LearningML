import torch
import os
import numpy as np
from PIL import Image

class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms):
        
        self.input_ids = []
        self.labels = []
        self.images = []
        self.image_files = []
        self.metadata = []
        for file in os.listdir(dataset_path):
            if file.endswith(".png") or file.endswith(".jpg"):
                metadata_file = file.replace(".png", ".txt").replace(".jpg", ".txt")
                if os.path.exists(os.path.join(dataset_path, metadata_file)):
                    input_id = file.replace(".png", "").replace(".jpg", "")
                    wds = input_id.split("_")[:-2]
                    self.input_ids.append(input_id)
                    self.labels.append(' '.join(wds))
                    self.image_files.append(os.path.join(dataset_path, file))
                    with open(os.path.join(dataset_path, metadata_file), 'r') as f:
                        self.metadata.append(f.read())

        #self.encoded_labels = text_tokenizer(self.labels)
        #self.encoded_metadata = metadata_tokenizer(self.metadata, padding=True, truncation=True, max_length=metadata_max_length)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}
        #item['input_ids'] = self.input_ids[idx]
        image = self.transforms(image=np.array(Image.open(self.image_files[idx], 'r')))['image']
        item['input_ids'] = torch.tensor(image).mean(dim=2).flatten().float()
        #item['labels'] = self.labels[idx]
        item['labels'] = torch.tensor(self.__str_to_int(self.labels[idx]))
        #item['encoded_labels'] = self.encoded_labels[idx]
        #item['encoded_metadata'] = self.encoded_metadata[idx]
        return item

    def __str_to_int(self, str):
        r = 0
        for c in str:
            r = r + ord(c)
        return r

    def __len__(self):
        return len(self.labels)