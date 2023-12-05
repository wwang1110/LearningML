#from screen import TextEncoder, ImageEncoder, MetadataEncoder
from data import ScreenDataset
import open_clip
from transformers import Trainer, TrainingArguments
from mlp import MLP
import albumentations as A
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel

CLIPModel.from_pretrained("openai/clip-vit-base-patch16", )
def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )
def _convert_to_rgb(image):
    return image.convert('RGB')

if __name__ == "__main__":

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    text_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    #dataset_path = 'D:/Adams/dataset/CUB_200_2011_CAP'
    dataset_path = 'D:/Adams/dataset/CUBTest'

    '''
    im = Image.open('D:/Adams/dataset/CUBTest/Acadian_Flycatcher_0012_795612.jpg')

    mytr1 = get_transforms(224)
    r1 = mytr1(image = np.array(im))
    mytransforms = transforms.Compose(
        [
            #transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
        ]
    )
    r2 = mytransforms(im)

    images = []
    for file in os.listdir(dataset_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            with Image.open(os.path.join(dataset_path, file), 'r') as img:
                images.append(img)

    #r3 = mytr1(image = np.array(images[10]))
    #r4 = mytransforms(images[10])
    
    '''
    
    dataset = ScreenDataset(dataset_path, get_transforms(224), text_tokenizer)


    # device = torch.device('hpu')
    device = 'cpu'
    n_classes = 3
    bs = 10
    n_features = 20

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

    val_dataset = dataset
    train_dataset = dataset

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

    trainer.train()