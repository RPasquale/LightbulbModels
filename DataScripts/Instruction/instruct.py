import os
import numpy as np
import torch
from transformers import BertTokenizer
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

# Configuration for tokenizer and max length
class Config:
    tokenizer_name = 'bert-base-uncased'
    max_length = 512
    shard_size = 10000

config = Config()

def preprocess_text(text):
    if isinstance(text, list):
        return " ".join([preprocess_text(t) for t in text])
    if isinstance(text, dict):
        return " ".join([f"{k}: {v}" for k, v in text.items()])
    return str(text)

def tokenize_function(examples, tokenizer, columns):
    tokenized_inputs = {key: [] for key in tokenizer.model_input_names}
    for column in columns:
        if column in examples:
            texts = [preprocess_text(text) for text in examples[column]]
            print(f"Tokenizing column: {column} with texts: {texts[:5]}")  # Print first 5 texts for debugging
            tokenized_column = tokenizer(
                texts, padding="max_length", truncation=True, max_length=config.max_length
            )
            for key in tokenized_column:
                tokenized_inputs[key].extend(tokenized_column[key])
    return tokenized_inputs

def save_tokenized_shards(dataset, shard_size, data_path):
    for split in dataset:
        split_dir = os.path.join(data_path, split)
        os.makedirs(split_dir, exist_ok=True)
        num_shards = (len(dataset[split]) + shard_size - 1) // shard_size  # Ensure correct number of shards
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, len(dataset[split]))  # Ensure end_idx is within range
            shard = dataset[split].select(range(start_idx, end_idx))
            shard_path = os.path.join(split_dir, f"shard_{shard_idx}.arrow")
            shard.save_to_disk(shard_path)
            print(f"Saved shard {shard_idx} to {shard_path}")

def process_and_save(dataset_name, data_path, columns_to_tokenize, config_name=None):
    if config_name:
        dataset = load_dataset(dataset_name, config_name)
    else:
        dataset = load_dataset(dataset_name)

    print(f"Loaded dataset: {dataset_name} with columns: {dataset.column_names}")
    assert isinstance(dataset, DatasetDict), "Dataset should be of type DatasetDict"

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
    print(f"Loaded tokenizer: {config.tokenizer_name}")

    # Test tokenization on a small sample
    sample = dataset['train'].select(range(5))
    print(f"Sample for testing tokenization: {sample}")
    tokenized_sample = tokenize_function(sample, tokenizer, columns_to_tokenize)
    print(f"Tokenized sample: {tokenized_sample}")
    assert all(isinstance(val, list) for val in tokenized_sample.values()), "Tokenization output should be lists"

    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer, columns_to_tokenize), batched=True, remove_columns=dataset['train'].column_names)
    print(f"Tokenized dataset: {tokenized_datasets}")
    print(f"Remaining columns: {tokenized_datasets['train'].column_names}")

    save_tokenized_shards(tokenized_datasets, config.shard_size, data_path)

if __name__ == "__main__":
    datasets_info = [
        {
            "dataset_name": "instruction-pretrain/ft-instruction-synthesizer-collection",
            "config_name": "squad",
            "data_path": "C:/Users/Admin/DATAS/ft-instruction-synthesizer-collection",
            "columns_to_tokenize": ["context", "QA_list", "QA_type", "rc_name"]
        },
        {
            "dataset_name": "alespalla/chatbot_instruction_prompts",
            "data_path": "C:/Users/Admin/DATAS/chatbot_instruction_prompts",
            "columns_to_tokenize": ["response", "prompt"]
        },
        {
            "dataset_name": "vilm/Pretrain-Instruction-1",
            "data_path": "C:/Users/Admin/DATAS/Pretrain-Instruction-1",
            "columns_to_tokenize": ["text"]
        },
        {
            "dataset_name": "vilm/Pretrain-Instruction-2",
            "data_path": "C:/Users/Admin/DATAS/Pretrain-Instruction-2",
            "columns_to_tokenize": ["text"]
        }
    ]

    for info in datasets_info:
        print(f"Processing dataset: {info['dataset_name']}")
        process_and_save(info["dataset_name"], info["data_path"], info["columns_to_tokenize"], info.get("config_name"))
        print(f"Finished processing dataset: {info['dataset_name']}")
