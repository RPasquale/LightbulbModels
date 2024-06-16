import os
import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from datasets import load_dataset
from tqdm import tqdm

# Custom Dataset Class for COCO Data
class COCODataset(Dataset):
    def __init__(self, dataset, transform=None, max_objects=10):
        self.dataset = dataset
        self.transform = transform
        self.max_objects = max_objects

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        image = record['image'].convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        bboxes = record['objects']['bbox']
        labels = record['objects']['category']
        
        # Pad bboxes and labels
        bboxes = bboxes + [[0, 0, 0, 0]] * (self.max_objects - len(bboxes))
        labels = labels + [0] * (self.max_objects - len(labels))
        
        return image, torch.tensor(bboxes[:self.max_objects], dtype=torch.float32), torch.tensor(labels[:self.max_objects], dtype=torch.long)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def custom_collate_fn(batch):
    images, bboxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    bboxes = torch.stack(bboxes, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, bboxes, labels

def extract_features(dataloader, model, max_objects):
    all_features = []
    all_bboxes = []
    all_labels = []
    
    with torch.no_grad():
        for images, bboxes, labels in dataloader:
            outputs = model(images)
            features = outputs.view(outputs.size(0), -1)  # Flatten the tensor
            features = features.unsqueeze(1).expand(-1, max_objects, -1)  # (batch_size, max_objects, feature_dim)
            all_features.append(features)
            all_bboxes.append(bboxes)
            all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_bboxes), torch.cat(all_labels)

def write_datafile(filename, features, bboxes, labels):
    np.savez(filename, features=features, bboxes=bboxes, labels=labels)

def shard_data(features, bboxes, labels, shard_size, local_dir):
    num_samples = features.shape[0]
    num_shards = (num_samples + shard_size - 1) // shard_size

    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, num_samples)
        
        shard_features = features[start_idx:end_idx]
        shard_bboxes = bboxes[start_idx:end_idx]
        shard_labels = labels[start_idx:end_idx]
        
        filename = os.path.join(local_dir, f"coco_shard_{i:06d}.npz")
        write_datafile(filename, shard_features, shard_bboxes, shard_labels)

def main():
    # Parameters
    local_dir = "coco_features"
    shard_size = 10000  # Adjust this value based on your requirement
    os.makedirs(local_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset("detection-datasets/coco", split='train')
    
    # Create the dataset and dataloader
    coco_dataset = COCODataset(dataset, transform=transform)
    coco_dataloader = DataLoader(coco_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    
    # Load pre-trained ResNet model
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.eval()
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
    
    # Extract features from COCO data
    features, bboxes, labels = extract_features(coco_dataloader, resnet_model, max_objects=10)
    
    # Shard the data
    shard_data(features, bboxes, labels, shard_size, local_dir)
    print("Sharding completed.")

if __name__ == '__main__':
    main()
