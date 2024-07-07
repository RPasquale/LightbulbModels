import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Configuration for tokenizer and max length
class Config:
    tokenizer_name = 'bert-base-uncased'
    max_length = 512
    shard_size = 10000

config = Config()

def tokenize_function(examples, tokenizer, columns):
    concatenated_columns = [" ".join([str(examples[col]) for col in columns])]
    return tokenizer(concatenated_columns, padding="max_length", truncation=True, max_length=config.max_length)

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

def process_and_save(dataset_name, data_path, columns_to_tokenize=None):
    dataset = load_dataset(dataset_name)
    if columns_to_tokenize:
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
        tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer, columns_to_tokenize), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns([col for col in dataset.column_names if col not in columns_to_tokenize])
    else:
        tokenized_datasets = dataset

    save_tokenized_shards(tokenized_datasets, config.shard_size, data_path)

if __name__ == "__main__":
    datasets_info = [
        {
            "dataset_name": "codeparrot/codeparrot-clean",
            "data_path": "C:/Users/Admin/DATAS/codeparrot_clean",
            "columns_to_tokenize": ["content"]
        },
        {
            "dataset_name": "suvadityamuk/huggingface-transformers-code-dataset",
            "data_path": "C:/Users/Admin/DATAS/huggingface_transformers_code",
            "columns_to_tokenize": ["text"]
        },
        {
            "dataset_name": "Fraser/python-state-changes",
            "data_path": "C:/Users/Admin/DATAS/python_state_changes",
            "columns_to_tokenize": ["start", "code", "end"]
        },
        {
            "dataset_name": "lucasmccabe-lmi/gpt4all_code",
            "data_path": "C:/Users/Admin/DATAS/gpt4all_code",
            "columns_to_tokenize": ["instruction", "input", "output"]
        },
        {
            "dataset_name": "sahil2801/CodeAlpaca-20k",
            "data_path": "C:/Users/Admin/DATAS/CodeAlpaca_20k",
            "columns_to_tokenize": ["instruction", "input", "output"]
        },
        {
            "dataset_name": "lucasmccabe-lmi/codex_math_qa_alpaca_style",
            "data_path": "C:/Users/Admin/DATAS/codex_math_qa_alpaca_style",
            "columns_to_tokenize": ["instruction", "input", "output"]
        },
        {
            "dataset_name": "lucasmccabe-lmi/instruct_to_code_alpaca_style",
            "data_path": "C:/Users/Admin/DATAS/instruct_to_code_alpaca_style",
            "columns_to_tokenize": ["instruction", "input", "output"]
        },
        {
            "dataset_name": "TokenBender/code_instructions_122k_alpaca_style",
            "data_path": "C:/Users/Admin/DATAS/code_instructions_122k_alpaca_style",
            "columns_to_tokenize": ["instruction", "input", "output", "text"]
        },
        {
            "dataset_name": "HydraLM/GPTeacher_codegen_alpaca",
            "data_path": "C:/Users/Admin/DATAS/GPTeacher_codegen_alpaca",
            "columns_to_tokenize": ["input", "output"]
        }
    ]

    for info in datasets_info:
        process_and_save(info["dataset_name"], info["data_path"], info.get("columns_to_tokenize"))
