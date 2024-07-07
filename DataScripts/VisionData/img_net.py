import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image

# Custom Dataset Class for ImageNet Data
class ImageNetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        image = record['image'].convert("RGB")
        label = record['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def extract_features(dataloader, model):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            features = outputs.view(outputs.size(0), -1)  # Flatten the tensor
            all_features.append(features)
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)

def write_datafile(filename, features, labels):
    np.savez(filename, features=features.numpy(), labels=labels.numpy())

def shard_data(features, labels, shard_size, local_dir):
    num_samples = features.shape[0]
    num_shards = (num_samples + shard_size - 1) // shard_size
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, num_samples)
        shard_features = features[start_idx:end_idx]
        shard_labels = labels[start_idx:end_idx]
        filename = os.path.join(local_dir, f"imagenet_shard_{i:06d}.npz")
        write_datafile(filename, shard_features, shard_labels)

def main():
    # Parameters
    local_dir = "C:/Users/Admin/DATAS/imagenet_features"
    shard_size = 10000  # Adjust this value based on your requirement
    os.makedirs(local_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset("zh-plus/tiny-imagenet", split='train')
    
    # Create the dataset and dataloader
    imagenet_dataset = ImageNetDataset(dataset, transform=transform)
    imagenet_dataloader = DataLoader(imagenet_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    
    # Load pre-trained ResNet model
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_model.eval()
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
    
    # Extract features from ImageNet data
    features, labels = extract_features(imagenet_dataloader, resnet_model)
    
    # Shard the data
    shard_data(features, labels, shard_size, local_dir)
    print("Sharding completed.")

if __name__ == '__main__':
    main()
