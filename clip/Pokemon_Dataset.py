import torch
import numpy as np
from .CFG import CFG
from .utils import get_transforms
from datasets import load_dataset

class Pokemon_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.dataset = dataset
        self.captions = self.dataset['en_text']
        self.images = self.dataset['image']
        self.encoded_captions = tokenizer(
            self.dataset['en_text'], padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encoded_captions['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encoded_captions['attention_mask'][idx]),
        }
        image = self.transforms(image=np.array(self.images[idx]))['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)

def build_pokemon_loaders(tokenizer):
    dataset = load_dataset('svjack/pokemon-blip-captions-en-ja', split='train')
    split_dataset = dataset.train_test_split(test_size=0.2, seed=18)
    transforms = get_transforms(mode='train')
    train_dataset = Pokemon_Dataset(split_dataset['train'], tokenizer=tokenizer, transforms=transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    test_dataset = Pokemon_Dataset(split_dataset['test'], tokenizer=tokenizer, transforms=transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    return train_dataloader, test_dataloader