
from PIL import Image
import albumentations as A
import numpy as np
import torch
from disjoint_set import DisjointSet
from sklearn.metrics import f1_score, accuracy_score

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def get_transforms(img_size):
    return A.Compose(
        [
            A.ToRGB(),
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )

def build_inputs(image_file, text_file, labels, transforms, clip_processor, roberta_tokenizer):
    item = {}
    pil_image = Image.open(image_file, 'r')
    image = transforms(image=np.array(pil_image))['image']
    image = torch.tensor(image).permute(2, 0, 1).float()
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)

    item['input_ids'] = inputs['input_ids']
    item['attention_mask'] = inputs['attention_mask']
    item['pixel_values'] = inputs['pixel_values']

    with open(text_file, 'r') as f:
        metadata = f.read()
    encoded_metadata = roberta_tokenizer(metadata, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    item['encoded_metadata'] = encoded_metadata['input_ids']
    item['metadata_attention_mask'] = encoded_metadata['attention_mask']

    return item   

def compute_metrics_ex(predictions):
    (_, batch_size) = torch.from_numpy(predictions.predictions[2]).shape
    probs = torch.argmax(torch.from_numpy(predictions.predictions[2]), dim=1)

    ds = DisjointSet()
    for i in range(len(predictions.label_ids)):
        ds.find(i)
        key = ','.join([str(x.item()) for x in predictions.label_ids[i]])
        if key not in ds:
            ds.find(key)
        ds.union(key, i)

    y_pred = []
    y_true = []
    for i in range(len(probs)):
        y_true.append(ds.find(i))
        pred_idx = int((i / batch_size) * batch_size + probs[i].item())
        y_pred.append(ds.find(pred_idx))

    return {'accuray': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, average='macro')}

def compute_metrics(predictions):
    probs = torch.from_numpy(predictions.predictions[2])
    (_, batch_size) = probs.shape

    y_pred = [x.item() for x in torch.argmax(probs, dim=1)]
    y_true = []
    for i in range(len(y_pred)):
        y_true.append(i % batch_size)

    return {'accuray': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, average='macro')}