# Run training Example:
#  python train_language_tasks.py --dataset_names C:/Users/Admin/DATAS/codeparrot_clean C:/Users/Admin/DATAS/reward_model_anthropic --task_types codegen dpo --epochs 3 2


import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
from datasets import load_from_disk

# Configuration
class Config:
    seq_len = 512
    learning_rate = 5e-5
    warmup_steps = 100
    logging_steps = 10
    model_save_path = "gpt2_finetuned"
    best_model_save_path = "gpt2_best_finetuned"
    reward_model_path = 'path/to/reward_model'  # Path to the reward model

config = Config()

# Argument parser
parser = argparse.ArgumentParser(description="Train language model on selected datasets and tasks")
parser.add_argument("--dataset_names", nargs='+', required=True, help="List of dataset names")
parser.add_argument("--task_types", nargs='+', required=True, choices=['codegen', 'dpo', 'instruct'], help="List of task types")
parser.add_argument("--epochs", nargs='+', type=int, required=True, help="Number of epochs for each dataset")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
args = parser.parse_args()

config.batch_size = args.batch_size
config.max_steps = args.max_steps

# Custom Dataset Classes
class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, columns):
        self.dataset = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.columns = columns

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = " ".join([example[col] for col in self.columns])
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask

class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, columns):
        self.dataset = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.columns = columns

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        inputs = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask, chosen, rejected

class MathDPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, columns):
        self.dataset = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.columns = columns

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = example['prompt']
        initial_reason_steps = example['initial_reason_steps']
        chosen = example['chosen']
        rejected = example['rejected']
        inputs = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask, initial_reason_steps, chosen, rejected

# Initialize the tokenizer and models
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
reward_model = BertForSequenceClassification.from_pretrained(config.reward_model_path)

# Data paths and columns to tokenize for different tasks
datasets_info = {
    "codegen": [
        {"dataset_name": "C:/Users/Admin/DATAS/codeparrot_clean", "columns_to_tokenize": ["content"]},
        {"dataset_name": "C:/Users/Admin/DATAS/huggingface_transformers_code", "columns_to_tokenize": ["text"]},
        {"dataset_name": "C:/Users/Admin/DATAS/python_state_changes", "columns_to_tokenize": ["start", "code", "end"]},
        {"dataset_name": "C:/Users/Admin/DATAS/gpt4all_code", "columns_to_tokenize": ["instruction", "input", "output"]},
        {"dataset_name": "C:/Users/Admin/DATAS/CodeAlpaca_20k", "columns_to_tokenize": ["instruction", "input", "output"]},
        {"dataset_name": "C:/Users/Admin/DATAS/codex_math_qa_alpaca_style", "columns_to_tokenize": ["instruction", "input", "output"]},
        {"dataset_name": "C:/Users/Admin/DATAS/instruct_to_code_alpaca_style", "columns_to_tokenize": ["instruction", "input", "output"]},
        {"dataset_name": "C:/Users/Admin/DATAS/code_instructions_122k_alpaca_style", "columns_to_tokenize": ["instruction", "input", "output", "text"]},
        {"dataset_name": "C:/Users/Admin/DATAS/GPTeacher_codegen_alpaca", "columns_to_tokenize": ["input", "output"]}
    ],
    "dpo": [
        {"dataset_name": "C:/Users/Admin/DATAS/reward_model_anthropic", "columns_to_tokenize": ["prompt", "response", "chosen", "rejected", "output"]},
        {"dataset_name": "C:/Users/Admin/DATAS/Math-Step-DPO-10K", "columns_to_tokenize": ["dataset", "prompt", "initial_reason_steps", "chosen", "rejected", "full_chosen", "full_rejected", "answer"]},
        {"dataset_name": "C:/Users/Admin/DATAS/MetaMath_DPO_FewShot", "columns_to_tokenize": ["prompt", "chosen", "rejected"]},
        {"dataset_name": "C:/Users/Admin/DATAS/multilingual-ultrafeedback-dpo-v0.1", "columns_to_tokenize": ["prompt", "chosen", "rejected"]},
        {"dataset_name": "C:/Users/Admin/DATAS/orca_dpo_pairs", "columns_to_tokenize": ["question", "chosen", "rejected"]}
    ],
    "instruct": [
        {"dataset_name": "C:/Users/Admin/DATAS/ft-instruction-synthesizer-collection", "columns_to_tokenize": ["context", "QA_list", "QA_type", "rc_name"]},
        {"dataset_name": "C:/Users/Admin/DATAS/chatbot_instruction_prompts", "columns_to_tokenize": ["response", "prompt"]},
        {"dataset_name": "C:/Users/Admin/DATAS/Pretrain-Instruction-1", "columns_to_tokenize": ["text"]},
        {"dataset_name": "C:/Users/Admin/DATAS/Pretrain-Instruction-2", "columns_to_tokenize": ["text"]}
    ]
}

# Create datasets and dataloaders for each task
def create_dataloader(dataset_info, task_type):
    data_path = dataset_info["dataset_name"]
    columns = dataset_info["columns_to_tokenize"]
    if task_type == 'codegen' or task_type == 'instruct':
        dataset = TextDataset(data_path, tokenizer, config.seq_len, columns)
    elif task_type == 'dpo':
        if "initial_reason_steps" in columns:
            dataset = MathDPODataset(data_path, tokenizer, config.seq_len, columns)
        else:
            dataset = DPODataset(data_path, tokenizer, config.seq_len, columns)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Set up task-specific optimizers and schedulers
def create_optimizer_and_scheduler(model, epochs):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps * epochs)
    return optimizer, scheduler

optimizers_schedulers = {
    'codegen': [create_optimizer_and_scheduler(model, epoch) for epoch in args.epochs],
    'dpo': [create_optimizer_and_scheduler(model, epoch) for epoch in args.epochs],
    'instruct': [create_optimizer_and_scheduler(model, epoch) for epoch in args.epochs]
}

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
reward_model.to(device)
best_val_loss = float('inf')

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Training Loss')
line2, = ax.plot([], [], label='Validation Loss')
ax.set_xlabel('Steps')
ax.set_ylabel('Loss')
ax.legend()

def update_plot(train_losses, val_losses):
    line1.set_xdata(np.arange(len(train_losses)))
    line1.set_ydata(train_losses)
    line2.set_xdata(np.arange(len(val_losses)))
    line2.set_ydata(val_losses)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def compute_reward(prompt, response):
    inputs = tokenizer(prompt + response, return_tensors="pt", padding=True, truncation=True, max_length=config.seq_len)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = reward_model(**inputs)
    reward = outputs.logits.squeeze().mean().item()
    return reward

def train_task(dataloader, optimizer, scheduler, task_type, task_index):
    model.train()
    train_losses = []
    val_losses = []

    for step, batch in enumerate(dataloader):
        if step >= config.max_steps:
            break

        if task_type == 'dpo':
            input_ids, attention_mask, *responses = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            chosen_responses = responses[0]
            rejected_responses = responses[1]
            optimizer.zero_grad()
            chosen_loss = 0.0
            rejected_loss = 0.0

            for chosen, rejected in zip(chosen_responses, rejected_responses):
                chosen_reward = compute_reward(tokenizer.decode(input_ids[0]), chosen)
                rejected_reward = compute_reward(tokenizer.decode(input_ids[0]), rejected)
                loss = -torch.log(chosen_reward / (chosen_reward + rejected_reward + 1e-10))
                chosen_loss += loss.item()
                loss.backward()

            optimizer.step()
            scheduler.step()
            train_losses.append(chosen_loss / len(chosen_responses))
        else:  # CodeGen or Instruct tasks
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % config.logging_steps == 0:
            print(f"Step {step}/{config.max_steps} - Loss: {loss.item()}")
            model.eval()
            val_loss = 0.0
            for val_batch in dataloader:
                val_input_ids, val_attention_mask = val_batch[:2]
                val_input_ids = val_input_ids.to(device)
                val_attention_mask = val_attention_mask.to(device)
                with torch.no_grad():
                    val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_input_ids)
                    val_loss += val_outputs.loss.item()
            val_losses.append(val_loss / len(dataloader))
            print(f"Validation Loss: {val_loss / len(dataloader)}")

            # Save best model
            global best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(config.best_model_save_path)

            update_plot(train_losses, val_losses)
            model.train()

# Training the tasks
for dataset_name, task_type, epochs in zip(args.dataset_names, args.task_types, args.epochs):
    dataset_info = next(item for item in datasets_info[task_type] if item["dataset_name"] == dataset_name)
    dataloader = create_dataloader(dataset_info, task_type)
    optimizer, scheduler = optimizers_schedulers[task_type][0]  # Just using the first optimizer for now
    train_task(dataloader, optimizer, scheduler, task_type, 0)

# Save the final model
model.save_pretrained(config.model_save_path)

plt.ioff()
plt.show()


# Run training Example:
#  python train_language_tasks.py --dataset_names C:/Users/Admin/DATAS/codeparrot_clean C:/Users/Admin/DATAS/reward_model_anthropic --task_types codegen dpo --epochs 3 2
