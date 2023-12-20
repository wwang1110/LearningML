from PIL import Image
from io import BytesIO
import base64
import os
import json
from tqdm import tqdm

def img_to_base64(file_name, wsize=224, hsize=224):
    img = Image.open(file_name)
    format = img.format
    img = img.resize((wsize, hsize), Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_buffer = BytesIO()
    img.save(img_buffer, format=format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    return base64_str

def build_image_label_map(raw_data_path):
    label_map = {}
    image_map = {}
    img_id = 0
    text_id = 0
    for file in os.listdir(raw_data_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            metadata_file = file.replace(".png", ".txt").replace(".jpg", ".txt")
            if os.path.exists(os.path.join(raw_data_path, metadata_file)):
                action_id = file.replace(".png", "").replace(".jpg", "")
                wds = action_id.split("_")[:-2]
                label = ' '.join(wds)
                if label not in label_map:
                    label_map[label] = {'text_id':text_id, 'text':label, 'image_ids':[]}
                    text_id += 1
                label_map[label]['image_ids'].append(img_id)
                image_map[action_id] = {'image_id':img_id, 'image':os.path.join(raw_data_path, file), 'metadata':os.path.join(raw_data_path, metadata_file)}
                img_id += 1

    return image_map, label_map

def build_dataset_input_file(image_map, dataset_input_file):    
    with open(dataset_input_file, 'w') as f:
        for v in tqdm(image_map.values()):
            image_id = v['image_id']
            image_base64_str = img_to_base64(v['image'])
            with open(v['metadata'], 'r') as meta_f:
                metadata = meta_f.read()
            metadata = metadata.replace('\n', ' ').replace('\t', ' ')
            f.write(f'{image_id}\t{image_base64_str}\t{metadata}\n')

def build_dataset_label_file(label_map, dataset_label_file):
    with open(dataset_label_file, 'w') as f:
        for v in tqdm(label_map.values()):
             f.write(f'{json.dumps(v)}\n')

def build_dataset_files(raw_data_path, dataset_name, target_dir):
    image_map, label_map = build_image_label_map(raw_data_path)
    build_dataset_input_file(image_map, os.path.join(target_dir, f'{dataset_name}_imgs.tsv'))
    build_dataset_label_file(label_map, os.path.join(target_dir, f'{dataset_name}_texts.jsonl'))

if __name__ == "__main__":
    build_dataset_files('D:/Adams/dataset/CUB_200_2011_CAP', 'CUB_200_2011_CAP', 'D:/Adams/lmdb')

