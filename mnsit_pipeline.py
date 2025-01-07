import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
from torchvision import datasets, transforms,
import os

from rational_kat_cu.kat.katransformer import KATVisionTransformer
from train_utils import torch_train_val_split
from train import Training

transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,))
    ])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

BATCH_SIZE = 4

train_loader, val_loader, test_loader = torch_train_val_split(
        dataset,
        batch_train=BATCH_SIZE,
        batch_eval=BATCH_SIZE)



DEVICE = 'cuda'

# Training Hyperparams
epochs = 100
overfit_batch = True

# Directory to save checkpoints
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
CP_PATH = "checkpoint.pth"
cp_path = os.path.join(checkpoint_dir, CP_PATH)

# KViT Hyperparams
input_dim = 28
in_chans=1
num_classes=10
embed_dim=768//2
num_heads=2
depth=2
# dropout = 0.4
##

# init KAT-ViT
model = KATVisionTransformer(
    img_size=input_dim,
    in_chans=in_chans,
    num_classes=num_classes,
    embed_dim=768//4,
    num_heads=2,
    depth=2).to(DEVICE)
##

# Optimizer
lr = 1e-4
weight_decay = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

##
# init training
KViT = Training(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    cp_path,
    DEVICE,
    overfit_batch=overfit_batch,
    )
# start training
KViT.train_with_eval()