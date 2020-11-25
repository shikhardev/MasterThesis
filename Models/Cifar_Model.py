import torch.nn as nn
import torch.nn.functional as F


class CifarModel(nn.Module):
    config = None

    def __init__(self, config):
        self.config = config
        super(CifarModel, self).__init__()
        self.conv1 = nn.Conv2d(
            3, config['conv1_channels'], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(config['conv1_channels'],
                               config['conv2_channels'], 5)
        self.fc1 = nn.Linear(config['conv2_channels'] *
                             5 * 5, config['hidden_nodes'])
        self.fc2 = nn.Linear(config['hidden_nodes'], 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.config['conv2_channels'] * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
