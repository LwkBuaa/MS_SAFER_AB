import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
import torchvision


class SpResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(SpResNet18, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=pretrained)
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (2, 2), 1)
        self.net = nn.Sequential(*list(self.net.children())[1:-1])
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.net(x)
        x = x.view(-1, 512*1*1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    data = torch.rand((16, 3, 24, 24))
    net = SpResNet18()
    print(net)
    a = net(data)
    print(a.shape)
