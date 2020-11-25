import numpy as np
import torch.nn as nn


class TwitterModel(nn.Module):
    def __init__(self, num_classes, example_size):
        super().__init__()
        self.example_size = example_size
        self.main = nn.Sequential(
            nn.Linear(example_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
            )

        self.init_weights()

    def init_weights(self):
        self.main[0].weight.data.normal_(0, 1/np.sqrt(self.example_size))
        self.main[0].bias.data.zero_()
        self.main[2].weight.data.normal_(0, 1/np.sqrt(256))
        self.main[2].bias.data.zero_()
        self.main[4].weight.data.normal_(0, 1/np.sqrt(256))
        self.main[4].bias.data.zero_()


    def forward(self, x):
        return self.main(x)
