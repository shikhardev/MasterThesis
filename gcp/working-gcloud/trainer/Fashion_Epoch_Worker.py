import logging
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
from hpbandster.core.worker import Worker
from torch.utils.data import DataLoader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

DEBUG = False
DATA_LOCATION = "~/Data/MNIST/"
NUM_WORKER = 4
TEST_BATCH_SIZE = 1024
if torch.cuda.is_available():
    PIN_MEMORY = True
else:
    PIN_MEMORY = False

if DEBUG:
    from fashion.Fashion_Model import FashionModel as Model
    import utils
else:
    import trainer.util as utils
    from trainer.Fashion_Model import FashionModel as Model

logging.basicConfig(level=logging.DEBUG)
DEVICE = utils.get_device()


class FashionEpochWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainset, self.testset = utils.get_dataset('FASHION')
        self.trainset, self.valset = torch.utils.data.random_split(self.trainset, [50000, 10000])
        self.val_loader = DataLoader(self.valset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)
        self.test_loader = DataLoader(self.testset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)

    def train_model(self, model, data_loader, epochs, config):
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        for epoch in range(epochs):
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                # zero the parameter gradients
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return model

    def compute(self, config_id, config, budget, working_directory):
        train_loader = DataLoader(dataset=self.trainset, batch_size=config['batch_size'],
                                          shuffle=True, num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)

        model, old_budget = utils.load_model(working_directory, config_id)
        if model is None:
            old_budget = 0
            model = Model(config)

        delta = int(budget - old_budget)
        model = self.train_model(model, train_loader, delta, config)
        utils.save_model(working_directory, budget, config_id, model)

        train_accuracy = utils.evaluate_accuracy(model, train_loader)
        val_accuracy = utils.evaluate_accuracy(model, self.val_loader)
        test_accuracy = utils.evaluate_accuracy(model, self.test_loader)

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
        hyperparameter_defaults = dict(
            dropout=0.5,
            channels_one=16,
            channels_two=32,
            batch_size=100,
            learning_rate=0.001,
            epochs=1,
        )
        """
        return utils.get_fashion_configspace()


if __name__ == '__main__':
    worker = FashionEpochWorker(run_id='0')
    cs = worker.get_configspace()
    c = cs.sample_configuration().get_dictionary()
    print(c)
    number_of_epochs = 2
    start_time = dt.datetime.now()
    res = worker.compute(config_id=(0, 0, 0), config=c, budget=number_of_epochs, working_directory='.')
    delta = dt.datetime.now() - start_time
    print('Time spent: ', delta)
    print(res)
