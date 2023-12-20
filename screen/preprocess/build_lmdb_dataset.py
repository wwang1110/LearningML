import os
from tqdm import tqdm
import lmdb
import json
import pickle

def buil_lmdb_dataset(lmbd_dir, dataset_name, dataset_path):
    lmdb_img = os.path.join(lmbd_dir, "imgs")
    env_img = lmdb.open(lmdb_img, map_size=1024**3)
    txn_img = env_img.begin(write=True)

    lmdb_meta = os.path.join(lmbd_dir, "metas")
    env_meta = lmdb.open(lmdb_meta, map_size=1024**3)
    txn_meta = env_meta.begin(write=True)

    lmdb_pairs = os.path.join(lmbd_dir, "pairs")
    env_pairs = lmdb.open(lmdb_pairs, map_size=1024**3)
    txn_pairs = env_pairs.begin(write=True)

    # write LMDB file storing (image_id, text_id, text) pairs
    pairs_annotation_path = os.path.join(dataset_path, f"{dataset_name}_texts.jsonl")
    with open(pairs_annotation_path, "r", encoding="utf-8") as fin_pairs:
        write_idx = 0
        for line in tqdm(fin_pairs):
            line = line.strip()
            obj = json.loads(line)
            for field in ("text_id", "text", "image_ids"):
                assert field in obj, "Field {} does not exist in line {}. \
                    Please check the integrity of the text annotation Jsonl file."
            for image_id in obj["image_ids"]:
                dump = pickle.dumps((image_id, obj['text_id'], obj['text'])) # encoded (image_id, text_id, text)
                txn_pairs.put(key="{}".format(write_idx).encode('utf-8'), value=dump)  
                write_idx += 1
                if write_idx % 5000 == 0:
                    txn_pairs.commit()
                    txn_pairs = env_pairs.begin(write=True)
        txn_pairs.put(key=b'num_samples',
                value="{}".format(write_idx).encode('utf-8'))
        txn_pairs.commit()
        env_pairs.close()
    print(f"Finished serializing {write_idx} pairs into {lmdb_pairs}.")

    # write LMDB file storing image base64 strings
    base64_path = os.path.join(dataset_path, f"{dataset_name}_imgs.tsv")
    with open(base64_path, "r", encoding="utf-8") as fin_imgs:
        write_idx = 0
        for line in tqdm(fin_imgs):
            line = line.strip()
            image_id, b64, metadata = line.split("\t")
            txn_img.put(key="{}".format(image_id).encode('utf-8'), value=b64.encode("utf-8"))
            txn_meta.put(key="{}".format(image_id).encode('utf-8'), value=metadata.encode("utf-8"))
            write_idx += 1
            if write_idx % 1000 == 0:
                txn_img.commit()
                txn_img = env_img.begin(write=True)
        txn_img.put(key=b'num_images',
                value="{}".format(write_idx).encode('utf-8'))
        txn_meta.put(key=b'num_images',
                value="{}".format(write_idx).encode('utf-8'))
        txn_img.commit()
        env_img.close()
        txn_meta.commit()
        env_meta.close()               
    print(f"Finished serializing {write_idx} images into {lmdb_img}.")

    print("done!")

buil_lmdb_dataset('D:/Adams/lmdb', 'CUB_200_2011_CAP', 'D:/Adams/lmdb')