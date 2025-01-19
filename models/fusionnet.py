import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# SEBlock
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ECABlock
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, attention_type=None, dropout_prob=0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        if attention_type == 'SE':
            self.attention = SEBlock(num_channels)
        elif attention_type == 'ECA':
            self.attention = ECABlock(num_channels)
        else:
            self.attention = None

        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        if self.attention:
            Y = self.attention(Y)
        if self.dropout:
            Y = self.dropout(Y)
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False, attention_type=None, dropout_prob=0):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2, attention_type=attention_type, dropout_prob=dropout_prob))
        else:
            blk.append(Residual(num_channels, num_channels, attention_type=attention_type, dropout_prob=dropout_prob))
    return blk

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.initial_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)  

        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(*resnet_block(64, 64, 1, first_block=True, attention_type='SE', dropout_prob=0))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2, attention_type='ECA', dropout_prob=0))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2, attention_type=None, dropout_prob=0))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2, attention_type=None, dropout_prob=0))

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 27, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(27),
            nn.ReLU()
        )  

        self.conv7 = nn.Sequential(
            nn.Conv2d(27, 9, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(9),
            nn.ReLU()
        )  

        self.conv8 = nn.Sequential(
            nn.Conv2d(9, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )  

        self.fc1 = nn.Linear(2 * 2 * 3, 12)
        self.fc2 = nn.Linear(12, 5)  # 假设有5个分类结果

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# 
if __name__ == "__main__":
    model = FusionNet()
    dummy_input = torch.randn(32, 1, 256, 256)  # Batch size: 32, Channel: 1, Height: 256, Width: 256
    output = model(dummy_input)
    print(output.shape)

