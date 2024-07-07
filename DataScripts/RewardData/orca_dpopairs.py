import os
import numpy as np
import torch
from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def preprocess_dpo_data(examples):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(tokenizer.vocab)  # Make sure this matches your embedding layer's vocab size

    # Define max sequence length
    max_seq_length = 512 // 3

    # Tokenize 'question', 'chosen', and 'rejected' fields
    tokenized_questions = tokenizer(examples['question'], padding='max_length', truncation=True, max_length=max_seq_length)
    tokenized_chosen = tokenizer(examples['chosen'], padding='max_length', truncation=True, max_length=max_seq_length)
    tokenized_rejected = tokenizer(examples['rejected'], padding='max_length', truncation=True, max_length=max_seq_length)

    # Prepare the labels: 1 for 'chosen' and 0 for 'rejected'
    labels = [1 if i % 2 == 0 else 0 for i in range(len(examples['question']))]

    return {
        'input_ids_question': tokenized_questions['input_ids'],
        'attention_mask_question': tokenized_questions['attention_mask'],
        'input_ids_chosen': tokenized_chosen['input_ids'],
        'attention_mask_chosen': tokenized_chosen['attention_mask'],
        'input_ids_rejected': tokenized_rejected['input_ids'],
        'attention_mask_rejected': tokenized_rejected['attention_mask'],
        'labels': labels
    }

def save_tokenized_shards(dataset, shard_size=10000):
    for split in dataset:
        split_dir = f"C:/Users/Admin/DATAS/dpo_shards/{split}"
        os.makedirs(split_dir, exist_ok=True)
        num_shards = len(dataset[split]) // shard_size + 1
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = (shard_idx + 1) * shard_size
            shard = dataset[split].select(range(start_idx, min(end_idx, len(dataset[split]))))
            shard.save_to_disk(os.path.join(split_dir, f"shard_{shard_idx}.arrow"))

# Load the DPO dataset from Hugging Face
dpo_dataset = load_dataset("Intel/orca_dpo_pairs")

# Apply the preprocessing to the dataset
dpo_dataset = dpo_dataset.map(preprocess_dpo_data, batched=True)

# Save the tokenized shards
save_tokenized_shards(dpo_dataset)

# Convert to PyTorch tensors after processing
dpo_dataset.set_format(type='torch', columns=['input_ids_question', 'attention_mask_question', 'input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected', 'labels'])

train_loader = DataLoader(dpo_dataset['train'], batch_size=2, shuffle=True)
