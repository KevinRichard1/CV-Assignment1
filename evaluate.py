import os
import argparse
from pathlib import Path
import random
import time
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.ConvNeXtTiny import ConvNeXtTiny, evaluate

SEED = 0
VAL_FRACTION = 0.20

def set_random_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(data_dir, img_size, batch_size):
    '''Load dataset'''
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f'Classes ({num_classes}): {class_names}')
    print(f'Test images: {len(test_dataset)}')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=torch.cuda.is_available())

    print(f'Test: {len(test_dataset)}')
    return num_classes, test_loader

def evaluate_model(model, test_loader, device, ckpt_path):
    '''Evaluate best model'''
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f'Final test loss: {test_loss:.4f}')
    print(f'Final test accuracy: {test_acc:.4f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--img_size', default=224)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--ckpt_path', default='./convnext_finetune.pth')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    set_random_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    num_classes, test_loader = load_dataset(args.data_dir, args.img_size, args.batch_size)

    model = ConvNeXtTiny(num_classes=num_classes)
    print(model)

    evaluate_model(model, test_loader, device, args.ckpt_path)

if __name__ == "__main__":
    main()