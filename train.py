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
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import v2
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
    '''Load dataset and add augmentation'''
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)

    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f'Classes ({num_classes}): {class_names}')
    print(f'Total training images: {len(full_train_dataset)}')

    val_size = int(round(len(full_train_dataset) * VAL_FRACTION))
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=torch.cuda.is_available())

    print(f'Train: {len(train_dataset)} | Validation: {len(val_dataset)}')
    return num_classes, train_loader, val_loader

def train_model(model, train_loader, val_loader, optimizer, device, epochs=20, patience=5, ckpt_path='./best_model.pth'):
    '''Train model on dataset'''
    criterion = nn.CrossEntropyLoss()
    mixup = v2.MixUp(num_classes=model.num_classes, alpha=0.2)
    cutmix = v2.CutMix(num_classes=model.num_classes, alpha=1.0)
    aug = v2.RandomChoice([mixup, cutmix])
    model = model.to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            images, labels = aug(images, labels)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_seen += labels.size(0)

        train_loss = total_loss / total_seen
        val_loss, val_acc = evaluate(model, val_loader, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, ckpt_path)

            print(f'New best validation accuracy: {best_val_acc:.4f}')
            print(f'Checkpoint saved to: {ckpt_path}')
        else:
            patience_counter += 1

        elapsed = time.time() - start
        print(f'Epoch {epoch:02d}/{epochs} | '
              f'lr {optimizer.param_groups[0]["lr"]:.2e} | '
              f'train loss {train_loss:.4f} | val loss {val_loss:.4f} | '
              f'val acc {val_acc:.4f} | {elapsed:.1f}s')

        scheduler.step()

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            print(f'Best validation accuracy: {best_val_acc:.4f}')
            break

    checkpoint = torch.load(
        ckpt_path,
        map_location=device
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return model, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--img_size', default=224)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--patience', default=5)
    parser.add_argument('--ckpt_path', default='./convnext_finetune.pth')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    set_random_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    num_classes, train_loader, val_loader = load_dataset(args.data_dir, args.img_size, args.batch_size)

    model = ConvNeXtTiny(num_classes=num_classes)
    print(model)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        ckpt_path=args.ckpt_path
    )

if __name__ == '__main__':
    main()