import torch
import os
from PIL import Image
import numpy as np

class ScreenDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms, text_tokenizer):
        
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

        self.encoded_labels = text_tokenizer(self.labels)
        #self.encoded_metadata = metadata_tokenizer(self.metadata, padding=True, truncation=True, max_length=metadata_max_length)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = self.labels[idx]
        item['labels'] = self.labels[idx]
        image = self.transforms(image=np.array(Image.open(self.image_files[idx], 'r')))['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()

        item['encoded_labels'] = self.encoded_labels[idx]
        #item['encoded_metadata'] = self.encoded_metadata[idx]
        return item


    def __len__(self):
        return len(self.labels)