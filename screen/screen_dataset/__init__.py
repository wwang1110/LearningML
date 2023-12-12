import torch
import os
from PIL import Image
import albumentations as A
import numpy as np

def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )

class ScreenDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, clip_processor, roberta_tokenizer):
        
        self.action_ids = []
        self.labels = []
        self.image_files = []
        self.metadata_files = []
        for file in os.listdir(dataset_path):
            if file.endswith(".png") or file.endswith(".jpg"):
                metadata_file = file.replace(".png", ".txt").replace(".jpg", ".txt")
                if os.path.exists(os.path.join(dataset_path, metadata_file)):
                    action_id = file.replace(".png", "").replace(".jpg", "")
                    wds = action_id.split("_")[:-2]
                    self.action_ids.append(action_id)
                    self.labels.append(' '.join(wds))
                    self.image_files.append(os.path.join(dataset_path, file))
                    self.metadata_files.append(os.path.join(dataset_path, metadata_file))
        
        self.transforms = get_transforms(224)
        self.clip_processor = clip_processor
        self.roberta_tokenizer = roberta_tokenizer

    def __getitem__(self, idx):
        item = {}
        pil_image = Image.open(self.image_files[idx], 'r')
        image = self.transforms(image=np.array(pil_image))['image']
        image = torch.tensor(image).permute(2, 0, 1).float()
        inputs = self.clip_processor(text=[self.labels[idx]], images=image, return_tensors="pt", padding=True)
        item['input_ids'] = inputs['input_ids'][0]
        item['attention_mask'] = inputs['attention_mask'][0]
        item['pixel_values'] = inputs['pixel_values'][0]

        with open(self.metadata_files[idx], 'r') as f:
            metadata = f.read()
        encoded_metadata = self.roberta_tokenizer(metadata, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        item['encoded_metadata'] = encoded_metadata['input_ids'][0]
        item['metadata_attention_mask'] = encoded_metadata['attention_mask'][0]
    
        item['labels'] = inputs['input_ids'][0]
        return item

    def __len__(self):
        return len(self.labels)