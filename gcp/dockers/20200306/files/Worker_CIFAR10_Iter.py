import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging
import torch.optim as optim

logging.basicConfig(level=logging.DEBUG)

"""
    hyperparameter_ranges = {
        'lr': ContinuousParameter(0.0001, 0.01),
        'hidden_nodes': IntegerParameter(20, 100),
        'batch_size': CategoricalParameter([128, 256, 512]),
        'conv1_channels': CategoricalParameter([32, 64, 128]),
        'conv2_channels': CategoricalParameter([64, 128, 256, 512]),
    }

"""


class Net(nn.Module):
    config = None

    def __init__(self, config):
        self.config = config
        super(Net, self).__init__()
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


class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.NUM_WORKERS = 8

        # Setup the database
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root="../data", train=True, transform=transform, download=True)
        # self.train_loader = torch.utils.data.DataLoader(trainset,
        #                                                 batch_size=config.batch_size,
        #                                                 shuffle=True,
        #                                                 num_workers=self.NUM_WORKERS)

        # self.train_loader, self.val_loader = self.get_train_validate_loaders(trainset,
        #                                                                      train_batch_size=config.batch_size,
        #                                                                      val_batch_size=config.batch_size,
        #                                                                      shuffle=True,
        #                                                                      train_size=40000,
        #                                                                      val_size=10000)

        self.testset = torchvision.datasets.CIFAR10(root="../data", train=False, transform=transform, download=True)
        # self.test_loader = torch.utils.data.DataLoader(testset,
        #                                                batch_size=config.batch_size,
        #                                                shuffle=False,
        #                                                num_workers=self.NUM_WORKERS)

    def evaluate_accuracy(self, model, data_loader):
        model.eval()
        correct = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
        # import pdb; pdb.set_trace()
        accuracy = correct / len(data_loader.sampler)
        return (accuracy)

    def get_train_validate_loaders(self, full_dataset, train_size=40000, val_size=10000, train_batch_size=64,
                                   val_batch_size=1024, shuffle=True):
        """
        Returns the training and validation data loaders. Assume train_size + val_size < 50,000
        """
        # 50000 is the total trainset for CIFAR
        if train_size + val_size == 50000:
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        else:
            train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, (50000 - train_size)])
            _, val_dataset = torch.utils.data.random_split(full_dataset, [(50000 - val_size), val_size])

        t_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=self.NUM_WORKERS
        )
        v_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=shuffle, num_workers=self.NUM_WORKERS
        )
        return t_loader, v_loader

    def get_data_loaders(self, config):
        train_loader, val_loader = self.get_train_validate_loaders(self.trainset,
                                                                   train_batch_size=config['batch_size'],
                                                                   val_batch_size=config['batch_size'],
                                                                   shuffle=True,
                                                                   train_size=40000,
                                                                   val_size=10000)
        test_loader = torch.utils.data.DataLoader(self.testset,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False,
                                                  num_workers=self.NUM_WORKERS)
        return train_loader, val_loader, test_loader

    def compute(self, config_id, config, budget, working_directory):
        budget = int(budget)
        MOMENTUM = 0.9

        train_loader, val_loader, test_loader = self.get_data_loaders(config)
        net = Net(config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=float(
            config['lr']), momentum=MOMENTUM)

        for epoch in range(budget):
            running_loss = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # zero the parameter gradients
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        train_accuracy = self.evaluate_accuracy(net, train_loader)
        val_accuracy = self.evaluate_accuracy(net, val_loader)
        test_accuracy = self.evaluate_accuracy(net, test_loader)
        return ({
            'loss': 1 - val_accuracy,  # remember: HpBandSter always minimizes!
            'info': {'test accuracy': test_accuracy,
                     'train accuracy': train_accuracy,
                     'validation accuracy': val_accuracy,
                     }

        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, default_value='1e-2', log=True)
        hidden_nodes = CSH.UniformIntegerHyperparameter('hidden_nodes', lower=20, upper=100)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [128, 256, 512])
        conv1_channels = CSH.CategoricalHyperparameter('conv1_channels', [32, 64, 128])
        conv2_channels = CSH.CategoricalHyperparameter('conv2_channels', [64, 128, 256, 512])
        cs.add_hyperparameters([lr, hidden_nodes, batch_size, conv1_channels, conv2_channels])
        return cs


if __name__ == '__main__':
    worker = PyTorchWorker(run_id='0')
    cs = worker.get_configspace()
    c = cs.sample_configuration().get_dictionary()
    print(c)
    res = worker.compute(config_id='1', config=c, budget=2, working_directory='.')
    print(res)
