import torch
from torch import nn
import torch.nn.functional as F


class MnistModel(torch.nn.Module):
    def __init__(self, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, dropout_rate, num_fc_units,
                 kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=kernel_size)
        self.conv2 = None
        self.conv3 = None

        output_size = (28 - kernel_size + 1) // 2
        num_output_filters = num_filters_1

        if num_conv_layers > 1:
            self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=kernel_size)
            num_output_filters = num_filters_2
            output_size = (output_size - kernel_size + 1) // 2

        if num_conv_layers > 2:
            self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=kernel_size)
            num_output_filters = num_filters_3
            output_size = (output_size - kernel_size + 1) // 2

        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv_output_size = num_output_filters * output_size * output_size

        self.fc1 = nn.Linear(self.conv_output_size, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, 10)

    def forward(self, x):

        # switched order of pooling and relu compared to the original example
        # to make it identical to the keras worker
        # seems to also give better accuracies
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        if self.conv2 is not None:
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        if self.conv3 is not None:
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = self.dropout(x)

        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
