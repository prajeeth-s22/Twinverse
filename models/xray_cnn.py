"""
A small CNN for the TwinVerse X-ray module.

Per project guidance ("use a pretrained model such as ResNet50 / EfficientNet for
Review-2"), the architecture below is intentionally lightweight because this
environment has no bundled real chest-X-ray dataset. Swapping in
torchvision.models.resnet50(weights=...) as the backbone is a drop-in upgrade once
a real dataset + internet access to pretrained weights is available (see README).

The last conv layer (self.conv3) is used as the Grad-CAM target layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class XrayCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Grad-CAM target layer
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._activations = None
        self._gradients = None

    def activations_hook(self, grad):
        self._gradients = grad

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 128 -> 64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 64 -> 32
        x = F.relu(self.bn3(self.conv3(x)))               # 32 -> 32
        self._activations = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        pooled = self.gap(x).flatten(1)
        out = self.fc(pooled)
        return out

    def get_activations(self):
        return self._activations

    def get_gradients(self):
        return self._gradients
