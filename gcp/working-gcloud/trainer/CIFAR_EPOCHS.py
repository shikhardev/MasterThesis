import logging
import datetime as dt
import torch.nn as nn
import torch.optim as optim
from hpbandster.core.worker import Worker

DEBUG = False
DATA_LOCATION = "~/Data/CIFAR10/"

if DEBUG:
    from CIFAR.CIFAR_MODEL import Net as Net
    import utils
else:
    import trainer.util as utils
    from trainer.CIFAR_MODEL import Net as Net

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


class CifarEpochWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainset, self.testset = utils.get_dataset('CIFAR10')
        # self.train_loader, self.val_loader, self.test_loader = utils.get_dataloaders('CIFAR10')

    def compute(self, config_id, config, budget, working_directory):
        budget = int(budget / 2)
        MOMENTUM = 0.9

        train_size = 40000
        val_size = 10000
        full_training_size = 50000
        train_batch_size = config['batch_size']

        train_loader, val_loader, test_loader = utils.datasets_to_dataloaders(trainset=self.trainset,
                                                                              testset=self.testset,
                                                                              train_size=train_size,
                                                                              val_size=val_size,
                                                                              full_training_size=full_training_size,
                                                                              train_batch_size=train_batch_size)

        net, old_budget = utils.load_model(working_directory, config_id)
        if net is None:
            old_budget = 0
            net = Net(config)

        device = utils.get_device()
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=float(
            config['lr']), momentum=MOMENTUM)

        delta = int(budget - old_budget)

        for epoch in range(delta):
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # zero the parameter gradients
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        utils.save_model(working_directory, budget, config_id, net)

        train_accuracy = utils.evaluate_accuracy(net, train_loader)
        val_accuracy = utils.evaluate_accuracy(net, val_loader)
        test_accuracy = utils.evaluate_accuracy(net, test_loader)

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
        return utils.get_cifar_configspace()


if __name__ == '__main__':
    worker = CifarEpochWorker(run_id='0')
    cs = worker.get_configspace()
    c = cs.sample_configuration().get_dictionary()
    print(c)
    number_of_epochs = 2
    start_time = dt.datetime.now()
    res = worker.compute(config_id=(0, 0, 0), config=c, budget=number_of_epochs, working_directory='.')
    delta = dt.datetime.now() - start_time
    print('Time spent: ', delta)
    print(res)
