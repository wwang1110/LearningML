from datasets import load_dataset
import albumentations as A
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import numpy as np

def get_transforms():
    return A.Compose(
        [
            A.ToGray(always_apply=True),
            A.Resize(16, 16, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )

dataset = load_dataset('cifar100')
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

class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
        }
        image = self.transforms(image=np.array(self.dataset[idx]['img']))['image']
        item['input_ids'] = torch.tensor(image).permute(2, 0, 1).float().mean(dim=0).flatten()
        item['labels'] = self.dataset[idx]['fine_label']
        item['clabels'] = self.dataset[idx]['coarse_label']

        return item


    def __len__(self):
        return len(self.dataset)
    
class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, d_in)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_in, d_out)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, clabels=None):
        x = self.fc1(input_ids)
        x = self.activation(x)
        logits = self.fc2(x)
        loss = self.criterion(logits, labels)
        return {'loss': loss, 'logits': logits}
    


# device = torch.device('hpu')
device = 'cpu'
n_classes = 200
n_features = 256

training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to='none',
        save_strategy='no',
        remove_unused_columns=False
    )

model = MLP(n_features, n_classes)

val_dataset = MLPDataset(dataset['test'], get_transforms())
train_dataset = MLPDataset(dataset['train'], get_transforms())

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

trainer.train()