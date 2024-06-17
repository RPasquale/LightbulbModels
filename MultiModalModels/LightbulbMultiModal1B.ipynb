import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

# Vision Model Components
from torchvision import models, transforms
from PIL import Image

# Language Model Components
import numpy as np
import os
import inspect

# Unified Config
@dataclass
class UnifiedConfig:
    vocab_size: int = 50257  # from language model
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    image_feature_dim: int = 2048  # from vision model
    max_objects: int = 10
    max_seq_len: int = 1024
    num_classes: int = 91  # for COCO dataset
    block_size: int = 1024  # for language model

# Custom Dataset and Dataloader for Language Model
class NpyDataset(Dataset):
    def __init__(self, data_dir, file_prefix):
        self.data_dir = data_dir
        self.file_names = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.startswith(file_prefix) and f.endswith('.npy')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        tokens_np = np.load(self.file_names[idx])
        tokens_tensor = torch.tensor(tokens_np, dtype=torch.long)
        return tokens_tensor

class CustomDataLoaderLite:
    def __init__(self, dataset, batch_size, seq_len):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.current_position = 0

    def __iter__(self):
        self.current_position = 0
        return self

    def __next__(self):
        if self.current_position >= len(self.dataset):
            raise StopIteration

        batch = []
        for _ in range(self.batch_size):
            if self.current_position >= len(self.dataset):
                break
            tokens = self.dataset[self.current_position]
            batch.append(tokens[:self.seq_len])
            self.current_position += 1

        x = torch.stack([tokens[:-1] for tokens in batch])
        y = torch.stack([tokens[1:] for tokens in batch])

        return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.scale_init = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Create causal mask dynamically based on input length
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        y = torch.matmul(attn_weights, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, attn_weights

# MLP
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.scale_init = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x, attn_weights = self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights

# Positional Encoding
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

# Vision Model Components
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
        bboxes = bboxes + [[0, 0, 0, 0]] * (self.max_objects - len(bboxes))
        labels = labels + [0] * (self.max_objects - len(labels))
        return image, torch.tensor(bboxes[:self.max_objects], dtype=torch.float32), torch.tensor(labels[:self.max_objects], dtype=torch.long)

# Transformation for Vision Model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_coco_dataset():
    from datasets import load_dataset
    dataset = load_dataset("detection-datasets/coco", split='train[:1%]')
    return dataset

def generate_random_dataset(num_samples=100):
    images = torch.randn(num_samples, 3, 224, 224)  # Random images
    bboxes = torch.rand(num_samples, 10, 4)  # Random bounding boxes
    labels = torch.randint(0, 91, (num_samples, 10))  # Random labels for 91 classes
    return images, bboxes, labels

def custom_collate_fn(batch):
    images, bboxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    bboxes = torch.stack(bboxes, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, bboxes, labels

def create_dataloader(images, bboxes, labels, batch_size=2):
    class RandomDataset(Dataset):
        def __init__(self, images, bboxes, labels):
            self.images = images
            self.bboxes = bboxes
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.bboxes[idx], self.labels[idx]

    dataset = RandomDataset(images, bboxes, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    return dataloader

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

# GPT Class
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x, _ = block(x)  # Add underscore to capture attention weights
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = UnifiedConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            elif k in sd_keys:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

class ChameleonEmbeddings(nn.Module):
    def __init__(self, config: UnifiedConfig, vision_model, text_model):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.image_proj = nn.Linear(config.image_feature_dim, config.n_embd)
        self.text_proj = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_encoder = PositionalEncoding(config.n_embd, max_len=config.max_seq_len + config.max_objects)

    def forward(self, text_input, image_input):
        text_emb = self.text_proj(text_input)
        print(f"text_emb shape: {text_emb.shape}")

        image_features = self.vision_model(image_input)
        print(f"image_features shape after vision model: {image_features.shape}")

        image_features = image_features.view(image_features.size(0), -1)  # Flatten the tensor
        image_features = image_features.unsqueeze(1).expand(-1, config.max_objects, -1)  # Expand to (B, max_objects, feature_dim)
        image_emb = self.image_proj(image_features)
        print(f"image_emb shape after projection: {image_emb.shape}")

        combined_emb = torch.cat([text_emb, image_emb], dim=1)
        print(f"combined_emb shape after concatenation: {combined_emb.shape}")

        combined_emb = self.pos_encoder(combined_emb)
        print(f"combined_emb shape after positional encoding: {combined_emb.shape}")

        return combined_emb

class ChameleonBlock(nn.Module):
    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x, attn_weights = self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights

class ChameleonModel(nn.Module):
    def __init__(self, config: UnifiedConfig, vision_model, text_model):
        super().__init__()
        self.embeddings = ChameleonEmbeddings(config, vision_model, text_model)
        self.transformer_blocks = nn.ModuleList([ChameleonBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.bbox_head = nn.Linear(config.n_embd, 4)
        self.class_head = nn.Linear(config.n_embd, config.num_classes)

    def forward(self, text_input, image_input):
        combined_emb = self.embeddings(text_input, image_input)
        x = combined_emb
        attn_weights_list = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attn_weights_list.append(attn_weights)
        x = self.ln_f(x)
        text_logits = self.lm_head(x[:, :text_input.size(1)])
        bbox_logits = self.bbox_head(x[:, text_input.size(1):])
        class_logits = self.class_head(x[:, text_input.size(1):])
        return text_logits, bbox_logits, class_logits, attn_weights_list

# Instantiate the Chameleon Model
config = UnifiedConfig()
vision_model = models.resnet50(pretrained=True)
vision_model.eval()
vision_model = nn.Sequential(*list(vision_model.children())[:-1])  # Pre-trained vision model
text_model = GPT(config)  # Pre-trained language model

chameleon_model = ChameleonModel(config, vision_model, text_model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
chameleon_model.to(device)

#####################################################
# Set testing flag to True to use fake data to test the model
# flag = False uses coco
testing = True

if testing:
    images, bboxes, labels = generate_random_dataset(num_samples=100)
else:
    dataset = load_coco_dataset()
    coco_dataset = COCODataset(dataset, transform=transform)
    dataloader = DataLoader(coco_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    images, bboxes, labels = extract_features(dataloader, vision_model, config.max_objects)
#####################################################
# Create dataloader for testing
dataloader = create_dataloader(images, bboxes, labels)

# Example inputs (replace with actual data)
text_input = torch.randint(0, config.vocab_size, (2, config.max_seq_len)).to(device)
image_input = next(iter(dataloader))[0].to(device)

# Forward pass
text_logits, bbox_logits, class_logits, attn_weights_list = chameleon_model(text_input, image_input)
print("Text logits shape:", text_logits.shape)
print("BBox logits shape:", bbox_logits.shape)
print("Class logits shape:", class_logits.shape)
print(f"Number of attention weight layers: {len(attn_weights_list)}")
if attn_weights_list:
    print(f"Shape of attention weights from first layer: {attn_weights_list[0].shape}")


import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Define your image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Top-k sampling decoding for multimodal model
def topk_multimodal_sampling(chameleon_model, text_model, vision_model, text_prompt, image_path, num_return_seq=5, max_length=30, top_k=50, device='cuda'):
    chameleon_model.eval()
    
    # Encode text prompt
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text_prompt)
    text_input = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_seq, 1).to(device)
    
    # Encode image
    image_input = preprocess_image(image_path).repeat(num_return_seq, 1, 1, 1).to(device)
    
    # Process image input through vision model to get features
    with torch.no_grad():
        image_features = vision_model(image_input)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten the tensor
        image_features = image_features.unsqueeze(1).expand(-1, config.max_objects, -1)  # Expand to (B, max_objects, feature_dim)
    
    # Initialize output tokens with the encoded text prompt
    output_tokens = text_input.clone()

    torch.manual_seed(111)
    torch.cuda.manual_seed(111)

    while output_tokens.size(1) < max_length:
        with torch.no_grad():
            text_logits, bbox_logits, class_logits, _ = chameleon_model(output_tokens, image_input)  # Use image_input here
            logits = text_logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, ix)
            output_tokens = torch.cat((output_tokens, next_token), dim=1)
    
    for i in range(num_return_seq):
        decoded = enc.decode(output_tokens[i, :].tolist())
        print(f">{decoded}")

# Example usage
text_prompt = "A dog in a kitchen,"
image_path = "G:/My Drive/LightbulbModels/sample_data_test/dog_kitchen_test_pic.jpg"  # Provide the path to your sample image
topk_multimodal_sampling(chameleon_model, text_model, vision_model, text_prompt, image_path, device=device)

def test_chameleon_embeddings():
    config = UnifiedConfig()
    vision_model = models.resnet50(pretrained=True)
    vision_model.eval()
    vision_model = nn.Sequential(*list(vision_model.children())[:-1])  # Pre-trained vision model
    text_model = GPT(config)  # Pre-trained language model

    chameleon_embeddings = ChameleonEmbeddings(config, vision_model, text_model)

    # Create dummy text and image inputs
    text_input = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
    image_input = torch.randn(1, 3, 224, 224)

    combined_emb = chameleon_embeddings(text_input, image_input)
    assert combined_emb.shape == (1, config.max_seq_len + config.max_objects, config.n_embd), \
        f"Unexpected combined embedding shape: {combined_emb.shape}"
    print("Test passed!")

test_chameleon_embeddings()

# Visualize attention weights
import matplotlib.pyplot as plt

def plot_attention_weights(attn_weights, layer, head):
    attn = attn_weights[layer][0, head].cpu().detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(attn, cmap='viridis')
    plt.title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()
    plt.show()

plot_attention_weights(attn_weights_list, layer=0, head=0)
