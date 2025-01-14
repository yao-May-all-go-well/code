import os
#from model.Resnet import create_resnet
#from model.model import create_resnet
#from model.ResDenseNet import create_resdensenet
#from model.Xception import xception
#from model.MobileNetV2 import MobileNetV2
#from model.SEResNet18 import SEResNet18
#from model.Resnext import create_resnet
#from model.GoogleNet import googlenet
#from model.ACBlock_Resnet import create_resnet
#from model.Sc_Resnet import create_resnet
#from model.PCTMF import PCTMFNet
from model.res_mCNN import FusionNet
#from model.mCNN import mCNN
#from model.Res_transformer import CNN_Transformer
from train2 import train_with_early_stopping
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np


class BiSpectrumDataset(Dataset):
    def __init__(self, data_dir, class_list):
        self.data = []
        self.labels = []
        for idx, class_name in enumerate(class_list):
            path = os.path.join(data_dir, class_name)
            for file_name in os.listdir(path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(path, file_name)
                    self.data.append(file_path)
                    self.labels.append(idx)

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
    # Python随机种子
    random.seed(seed)
    # NumPy随机种子
    np.random.seed(seed)
    # PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 禁用CuDNN以确保确定性

def main():
    setup_seed(3407)

    # 创建数据集和数据加载器 heart-classification jupyter-heart
    train_dataset_1 = BiSpectrumDataset('/root/autodl-tmp/heart-classification/features_11/train', ['AS', 'MR', 'MS', 'MVP', 'N'])
    train_dataset_2 = BiSpectrumDataset('/root/autodl-tmp/heart-classification/features_11/val', ['AS', 'MR', 'MS', 'MVP', 'N'])
    test_dataset = BiSpectrumDataset('/root/autodl-tmp/heart-classification/features_11/test', ['AS', 'MR', 'MS', 'MVP', 'N'])

    # 合并train和val数据集
    combined_train_data = train_dataset_1.data + train_dataset_2.data
    combined_train_labels = train_dataset_1.labels + train_dataset_2.labels

    class CombinedDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            file_path = self.data[idx]
            sample = np.load(file_path)
            sample = sample.reshape(1, 256, 256)
            label = self.labels[idx]
            return torch.tensor(sample, dtype=torch.float32), label

    train_dataset = CombinedDataset(combined_train_data, combined_train_labels)

    g = torch.Generator()
    g.manual_seed(3407)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # 创建模型
    #net = create_resnet()
    # net.eval()
    #net = PCTMFNet()
    # net = xception()
    # net = ResNet34()
    # net = MobileNetV2()
    # net = googlenet()
    #net = mCNN()
    net = FusionNet()
    # net = seresnet18()
    # net = CNN_Transformer()

    # 设置训练参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    #optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    num_epochs = 120
    patience = 100

    class_names = ["AS", "MR", "MS", "MVP", "N"]
    save_path = "/root/autodl-tmp/heart-classification/confusion_matrix.png"
    # 训练模型
    train_with_early_stopping(net, train_loader, test_loader, 32, loss_fn, optimizer, device, num_epochs, scheduler, patience,class_names,save_path)
if __name__ == '__main__':
    main()
