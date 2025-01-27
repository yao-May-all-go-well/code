import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random
from model_Two.Resnet import cearte_resnet

# 定义自定义数据集类
class BiSpectrumDataset(Dataset):
    def __init__(self, data_dir, csv_file):
        self.data = []
        self.labels = []
        df = pd.read_csv(csv_file, delimiter='\t', header=None, names=['file', 'label'])
        for _, row in df.iterrows():
            file_path = os.path.join(data_dir, f"{row['file']}.npy")
            if os.path.exists(file_path):
                self.data.append(file_path)
                label = 0 if row['label'] == 1 else 1  # 0代表正常，1代表异常
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        sample = np.load(file_path)
        sample = sample.reshape(1, 256, 256)
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), label

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 禁用CuDNN以确保确定性

def train_with_early_stopping(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs, scheduler=None, patience=5):
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    if best_model:
        model.load_state_dict(best_model)

def main():
    setup_seed(3407)

    # 创建数据集和数据加载器
    train_datasets = []
    for subfolder in ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']:
        feature_dir = f'/root/autodl-tmp/heart-classification/features_Two/train/{subfolder}'
        csv_file = f'/root/autodl-tmp/heart-classification/features_Two/train/{subfolder}/REFERENCE.csv'
        train_datasets.append(BiSpectrumDataset(feature_dir, csv_file))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = BiSpectrumDataset(
        feature_dir='/root/autodl-tmp/heart-classification/features_Two/validation',
        csv_file='/root/autodl-tmp/heart-classification/features_Two/validation/REFERENCE.csv'
    )

    g = torch.Generator()
    g.manual_seed(3407)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # 创建模型
    model = create_resnet()

    # 设置训练参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    num_epochs = 30
    patience = 5

    # 训练模型
    train_with_early_stopping(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs, scheduler, patience)

if __name__ == '__main__':
    main()
