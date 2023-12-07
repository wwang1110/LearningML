#from screen import TextEncoder, ImageEncoder, MetadataEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import Trainer, TrainingArguments
from mlp.mlp import MLP
from mlp.mlp_dataset import MLPDataset
import albumentations as A


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )

if __name__ == "__main__":

    #dataset_path = 'D:/Adams/dataset/CUB_200_2011_CAP'
    dataset_train_path = 'D:/Adams/dataset/CUBTest_train'
    dataset_val_path = 'D:/Adams/dataset/CUBTest_val'

    train_dataset = MLPDataset(dataset_train_path, get_transforms(16))
    val_dataset = MLPDataset(dataset_val_path, get_transforms(16))


    # device = torch.device('hpu')
    device = 'cpu'
    n_classes = 3
    bs = 8
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

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

    trainer.train()