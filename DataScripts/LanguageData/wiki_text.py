import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

# Configuration for tokenizer and max length
class Config:
    tokenizer_name = 'bert-base-uncased'
    max_length = 512
    shard_size = 10000

config = Config()

# Define the tokenization function
def tokenize_function(examples):
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config.max_length)

# Function to save tokenized shards
def save_tokenized_shards(dataset, shard_size=config.shard_size):
    for split in dataset:
        split_dir = f"C:/Users/Admin/DATAS/wikipedia_shards/{split}"
        os.makedirs(split_dir, exist_ok=True)
        num_shards = len(dataset[split]) // shard_size + 1
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = (shard_idx + 1) * shard_size
            shard = dataset[split].select(range(start_idx, min(end_idx, len(dataset[split]))))
            shard.save_to_disk(os.path.join(split_dir, f"shard_{shard_idx}.arrow"))

# Load the Wikipedia dataset
#dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")


# Apply the tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])  # Remove original text to only keep tokenized versions

# Save the tokenized shards
save_tokenized_shards(tokenized_datasets)

# Convert to PyTorch tensors after processing
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
train_loader = DataLoader(tokenized_datasets, batch_size=2, shuffle=True)
