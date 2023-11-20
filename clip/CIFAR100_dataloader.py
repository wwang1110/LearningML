import torch
import numpy as np
from .utils import get_transforms
from datasets import load_dataset
from .CFG import CFG

classes = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
]

class CIFAR100_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, transforms):
        self.dataset = dataset
        self.classes = classes
        self.captions = self.dataset['fine_label']
        self.images = self.dataset['img']
        self.encoded_classes = tokenizer(self.classes, padding=True, truncation=True, max_length=CFG.max_length)
        self.transforms = transforms

    def __getitem__(self, idx):
        cls_idx = self.captions[idx]
        item = {
            'input_ids': torch.tensor(self.encoded_classes['input_ids'][cls_idx]),
            'attention_mask': torch.tensor(self.encoded_classes['attention_mask'][cls_idx]),
        }

        item['cls_id'] = cls_idx
        image = self.transforms(image=np.array(self.images[idx]))['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.classes[cls_idx]

        return item


    def __len__(self):
        return len(self.captions)

def build_cifar100_loaders(tokenizer):
    dataset = load_dataset('cifar100')
    transforms = get_transforms(mode='train')
    train_dataset = CIFAR100_Dataset(dataset['train'], tokenizer=tokenizer, transforms=transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    test_dataset = CIFAR100_Dataset(dataset['test'], tokenizer=tokenizer, transforms=transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    return train_dataloader, test_dataloader