import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Configuration for tokenizer and max length
class Config:
    tokenizer_name = 'bert-base-uncased'
    max_length = 512
    shard_size = 10000

config = Config()

def tokenize_function(examples, tokenizer, columns):
    return tokenizer(examples[columns], padding="max_length", truncation=True, max_length=config.max_length)

def save_tokenized_shards(dataset, shard_size, data_path):
    for split in dataset:
        split_dir = os.path.join(data_path, split)
        os.makedirs(split_dir, exist_ok=True)
        num_shards = len(dataset[split]) // shard_size + 1
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = (shard_idx + 1) * shard_size
            shard = dataset[split].select(range(start_idx, min(end_idx, len(dataset[split]))))
            shard.save_to_disk(os.path.join(split_dir, f"shard_{shard_idx}.arrow"))

def process_and_save(dataset_name, data_path, columns_to_tokenize=None, image_column=None, trust_remote_code=False):
    dataset = load_dataset(dataset_name, trust_remote_code=trust_remote_code)
    if columns_to_tokenize:
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
        tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer, columns_to_tokenize), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns([col for col in dataset.column_names if col not in columns_to_tokenize])
    else:
        tokenized_datasets = dataset

    if image_column:
        # Convert images to tensors and save them
        for split in tokenized_datasets:
            for i in tqdm(range(len(tokenized_datasets[split])), desc=f"Processing {split} images"):
                image = tokenized_datasets[split][i][image_column]
                if isinstance(image, Image.Image):
                    tokenized_datasets[split][i][image_column] = np.array(image)

    save_tokenized_shards(tokenized_datasets, config.shard_size, data_path)

if __name__ == "__main__":
    datasets_info = [
        {
            "dataset_name": "facebook/textvqa",
            "data_path": "C:/Users/Admin/DATAS/textvqa",
            "columns_to_tokenize": ["question", "answers"],
            "image_column": "image",
            "trust_remote_code": True
        },
        {
            "dataset_name": "hbXNov/multi_scene_video_text_data",
            "data_path": "C:/Users/Admin/DATAS/multi_scene_video_text_data",
            "columns_to_tokenize": ["captions", "video_segments"]
        },
        {
            "dataset_name": "yifeihu/ACL-23-Paper-OCR-Markdown",
            "data_path": "C:/Users/Admin/DATAS/ACL-23-Paper-OCR-Markdown",
            "columns_to_tokenize": ["paper_id", "title", "url", "abstract", "ocr_markdown"]
        },
        {
            "dataset_name": "Suchinthana/Sinhala_OCR_Dataset_Synthetic",
            "data_path": "C:/Users/Admin/DATAS/Sinhala_OCR_Dataset_Synthetic",
            "image_column": "image"
        },
        {
            "dataset_name": "mychen76/invoices-and-receipts_ocr_v1",
            "data_path": "C:/Users/Admin/DATAS/invoices-and-receipts_ocr_v1",
            "image_column": "image",
            "columns_to_tokenize": ["id", "parsed_data", "raw_data"]
        }
    ]

    for info in datasets_info:
        process_and_save(info["dataset_name"], info["data_path"], info.get("columns_to_tokenize"), info.get("image_column"), info.get("trust_remote_code", False))
