import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt


# Configuration
class Config:
    image_size = 224
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    max_objects = 10
    model_save_path = "transformer_object_detection.pth"
    best_model_save_path = "best_transformer_object_detection.pth"
    datasets = []
    additional_datasets = [
        "facebook/textvqa",
        "hbXNov/multi_scene_video_text_data",
        "yifeihu/ACL-23-Paper-OCR-Markdown",
        "mychen76/invoices-and-receipts_ocr_v1"
    ]
    additional_datasets_save_path = "C:/Users/Admin/DATAS/additional_features"

config = Config()

# Argument parser
parser = argparse.ArgumentParser(description="Train vision model on various datasets")
parser.add_argument('--dataset_paths', nargs='+', help='Paths to datasets')
parser.add_argument('--epochs', nargs='+', type=int, help='Number of epochs for each dataset')
args = parser.parse_args()

# Override configuration with command-line arguments
if args.dataset_paths:
    config.datasets = args.dataset_paths
if args.epochs:
    config.num_epochs = args.epochs[0]

# Custom Dataset Class for Feature Data
class FeatureDataset(Dataset):
    def __init__(self, feature_files, max_objects):
        self.feature_files = feature_files
        self.max_objects = max_objects

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        data = np.load(self.feature_files[idx])
        features = torch.tensor(data['features'], dtype=torch.float32)
        bboxes = torch.tensor(data['bboxes'], dtype=torch.float32)
        labels = torch.tensor(data['labels'], dtype=torch.long)
        return features, bboxes, labels

def custom_collate_fn(batch):
    features, bboxes, labels = zip(*batch)
    features = torch.cat(features, dim=0)
    bboxes = torch.cat(bboxes, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, bboxes, labels

# Load feature files
feature_files = []
for dataset_path in config.datasets:
    feature_files.extend([os.path.join(dataset_path, f) for f in sorted(os.listdir(dataset_path)) if f.endswith('.npz')])

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pre-trained ResNet model
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])

# Function to extract features from images
def extract_features_from_image(image):
    if transform:
        image = transform(image)
    features = resnet_model(image.unsqueeze(0)).squeeze(0).detach().numpy()
    return features

# Function to process each dataset differently
def process_dataset(dataset_name, record, i):
    if dataset_name == "facebook/textvqa":
        image = record['image']
        features = extract_features_from_image(image)
        bboxes = [[0, 0, 0, 0]] * config.max_objects
        labels = [0] * config.max_objects

    elif dataset_name == "hbXNov/multi_scene_video_text_data":
        # Assuming processing only images and ignoring video segments
        image_path = record['video_segments'][0]
        image = Image.open(image_path).convert("RGB")
        features = extract_features_from_image(image)
        bboxes = [[0, 0, 0, 0]] * config.max_objects
        labels = [0] * config.max_objects

    elif dataset_name == "yifeihu/ACL-23-Paper-OCR-Markdown":
        # Assuming processing OCR text as features
        text = record['ocr_markdown']
        features = np.random.rand(config.image_size, config.image_size, 3)  # Placeholder for actual text feature extraction
        bboxes = [[0, 0, 0, 0]] * config.max_objects
        labels = [0] * config.max_objects

    elif dataset_name == "mychen76/invoices-and-receipts_ocr_v1":
        image = Image.open(record['image']).convert("RGB")
        features = extract_features_from_image(image)
        bboxes = [[0, 0, 0, 0]] * config.max_objects
        labels = [0] * config.max_objects

    else:
        return None

    data = {
        'features': features,
        'bboxes': np.array(bboxes[:config.max_objects]),
        'labels': np.array(labels[:config.max_objects])
    }
    feature_file = os.path.join(config.additional_datasets_save_path, f"{dataset_name}_{i:06d}.npz")
    np.savez(feature_file, **data)
    return feature_file

# Load additional datasets and process them
for dataset_name in config.additional_datasets:
    dataset = load_dataset(dataset_name, split='train')
    for i, record in enumerate(dataset):
        feature_file = process_dataset(dataset_name, record, i)
        if feature_file:
            feature_files.append(feature_file)

# Create the dataset and dataloader
feature_dataset = FeatureDataset(feature_files, config.max_objects)
feature_dataloader = DataLoader(feature_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_classes, num_heads=8, num_layers=6, max_objects=10):
        super(TransformerModel, self).__init__()
        self.feature_dim = feature_dim
        self.max_objects = max_objects
        self.pos_encoder = PositionalEncoding(feature_dim)
        self.pos_decoder = PositionalEncoding(feature_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        decoder_layers = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

        hidden_dim = 512  # You can adjust this value as needed
        self.mlp_bbox = MLPBlock(feature_dim, hidden_dim, 4)
        self.mlp_class = MLPBlock(feature_dim, hidden_dim, num_classes)

    def forward(self, src, tgt):
        # Encoder
        src = src.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, feature_dim)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)

        # Decoder
        tgt = tgt.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, feature_dim)
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        output = output.permute(1, 0, 2)  # Change shape back to (batch_size, max_objects, feature_dim)

        # Apply MLP blocks
        bboxes = self.mlp_bbox(output)
        classes = self.mlp_class(output)

        return bboxes, classes

# Define the model
feature_dim = 2048  # This is the dimension of the ResNet features
num_classes = len(load_dataset('detection-datasets/coco', split='train').features['objects'].feature['category'].names)
model = TransformerModel(feature_dim, num_classes)

criterion_bbox = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Training BBox Loss')
line2, = ax.plot([], [], label='Training Class Loss')
line3, = ax.plot([], [], label='Validation BBox Loss')
line4, = ax.plot([], [], label='Validation Class Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()

def update_plot(train_bbox_losses, train_class_losses, val_bbox_losses, val_class_losses):
    line1.set_xdata(np.arange(len(train_bbox_losses)))
    line1.set_ydata(train_bbox_losses)
    line2.set_xdata(np.arange(len(train_class_losses)))
    line2.set_ydata(train_class_losses)
    line3.set_xdata(np.arange(len(val_bbox_losses)))
    line3.set_ydata(val_bbox_losses)
    line4.set_xdata(np.arange(len(val_class_losses)))
    line4.set_ydata(val_class_losses)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def train_model(model, dataloader, criterion_bbox, criterion_class, optimizer, num_epochs=10):
    model.train()
    best_val_loss = float('inf')
    train_bbox_losses = []
    train_class_losses = []
    val_bbox_losses = []
    val_class_losses = []

    for epoch in range(num_epochs):
        running_loss_bbox = 0.0
        running_loss_class = 0.0
        for i, (features, bboxes, labels) in enumerate(dataloader):
            # Prepare target input for the decoder (could be shifted version of features)
            tgt = features

            optimizer.zero_grad()
            outputs_bboxes, outputs_classes = model(features, tgt)

            loss_bbox = criterion_bbox(outputs_bboxes, bboxes)
            loss_class = criterion_class(outputs_classes.view(-1, num_classes), labels.view(-1))
            loss = loss_bbox + loss_class
            loss.backward()
            optimizer.step()
            running_loss_bbox += loss_bbox.item()
            running_loss_class += loss_class.item()

        avg_train_bbox_loss = running_loss_bbox / len(dataloader)
        avg_train_class_loss = running_loss_class / len(dataloader)
        train_bbox_losses.append(avg_train_bbox_loss)
        train_class_losses.append(avg_train_class_loss)

        # Validation
        model.eval()
        val_loss_bbox = 0.0
        val_loss_class = 0.0
        with torch.no_grad():
            for features, bboxes, labels in dataloader:
                tgt = features
                outputs_bboxes, outputs_classes = model(features, tgt)
                loss_bbox = criterion_bbox(outputs_bboxes, bboxes)
                loss_class = criterion_class(outputs_classes.view(-1, num_classes), labels.view(-1))
                val_loss_bbox += loss_bbox.item()
                val_loss_class += loss_class.item()

        avg_val_bbox_loss = val_loss_bbox / len(dataloader)
        avg_val_class_loss = val_loss_class / len(dataloader)
        val_bbox_losses.append(avg_val_bbox_loss)
        val_class_losses.append(avg_val_class_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train BBox Loss: {avg_train_bbox_loss}, Train Class Loss: {avg_train_class_loss}, Val BBox Loss: {avg_val_bbox_loss}, Val Class Loss: {avg_val_class_loss}")

        # Save best model
        if avg_val_bbox_loss + avg_val_class_loss < best_val_loss:
            best_val_loss = avg_val_bbox_loss + avg_val_class_loss
            torch.save(model.state_dict(), config.best_model_save_path)

        model.train()
        update_plot(train_bbox_losses, train_class_losses, val_bbox_losses, val_class_losses)

    return train_bbox_losses, train_class_losses, val_bbox_losses, val_class_losses

train_bbox_losses, train_class_losses, val_bbox_losses, val_class_losses = train_model(model, feature_dataloader, criterion_bbox, criterion_class, optimizer, num_epochs=config.num_epochs)

# Save the final model
torch.save(model.state_dict(), config.model_save_path)

plt.ioff()
plt.show()

# Example training run:
# python train_vision.py --dataset_paths C:/Users/Admin/DATAS/imagenet_features C:/Users/Admin/DATAS/coco_features --epochs 3
