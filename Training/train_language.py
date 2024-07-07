# Example training run:
# python train_language.py --dataset_paths C:/Users/Admin/DATAS/edu_fineweb10B C:/Users/Admin/DATAS/wikipedia_shards --epochs 3 2

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
class Config:
    batch_size = 4
    seq_len = 512
    max_steps = 1000
    learning_rate = 5e-5
    warmup_steps = 100
    logging_steps = 10
    model_save_path = "gpt2_finetuned"
    best_model_save_path = "gpt2_best_finetuned"

config = Config()

# Argument parser
parser = argparse.ArgumentParser(description="Train language model on selected shards")
parser.add_argument("--dataset_paths", nargs='+', required=True, help="List of dataset paths")
parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for each dataset")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
args = parser.parse_args()

config.batch_size = args.batch_size
config.max_steps = args.max_steps

# Custom Dataset Class for Shards
class ShardDataset(Dataset):
    def __init__(self, data_dir, seq_len):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.shard_files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.npy') or f.endswith('.arrow')]

    def __len__(self):
        return len(self.shard_files)

    def __getitem__(self, idx):
        if self.shard_files[idx].endswith('.npy'):
            tokens = np.load(self.shard_files[idx])
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)
        else:
            data = torch.load(self.shard_files[idx])
            input_ids = data['input_ids'][:, :-1]
            labels = data['input_ids'][:, 1:]
            input_ids = input_ids.flatten()
            labels = labels.flatten()

        return input_ids[:self.seq_len], labels[:self.seq_len]

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Create datasets and dataloaders
dataloaders = []
for dataset_path in args.dataset_paths:
    dataset = ShardDataset(dataset_path, config.seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    dataloaders.append(dataloader)

# Set up the optimizer and scheduler
def create_optimizer_and_scheduler(model, epochs):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps * epochs)
    return optimizer, scheduler

optimizers_schedulers = [create_optimizer_and_scheduler(model, args.epochs) for _ in dataloaders]

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
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

def train_model(dataloader, optimizer, scheduler, model, epochs):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        for step, batch in tqdm(enumerate(dataloader), total=config.max_steps):
            if step >= config.max_steps:
                break

            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % config.logging_steps == 0:
                print(f"Step {step}/{config.max_steps} - Loss: {loss.item()}")
                val_loss = 0.0
                model.eval()
                for val_batch in dataloader:
                    val_input_ids, val_labels = val_batch
                    val_input_ids = val_input_ids.to(device)
                    val_labels = val_labels.to(device)
                    with torch.no_grad():
                        val_outputs = model(val_input_ids, labels=val_labels)
                        val_loss += val_outputs.loss.item()
                val_loss /= len(dataloader)
                val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss}")

                # Save best model
                global best_val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(config.best_model_save_path)

                update_plot(train_losses, val_losses)
                model.train()

# Training the tasks
for dataloader, (optimizer, scheduler) in zip(dataloaders, optimizers_schedulers):
    train_model(dataloader, optimizer, scheduler, model, args.epochs)

# Save the final model
model.save_pretrained(config.model_save_path)

plt.ioff()
plt.show()

# Example training run:
# python train_language.py --dataset_paths C:/Users/Admin/DATAS/edu_fineweb10B C:/Users/Admin/DATAS/wikipedia_shards --epochs 3 2
