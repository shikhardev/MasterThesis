import datetime as dt
import torch
import torch.nn.functional as F
from hpbandster.core.worker import Worker

DEBUG = False
DATA_LOCATION = "~/Data/MNIST/"

if DEBUG:
    from MNIST_CNN.MNIST_Model import MNISTConvNet
    import utils
else:
    # import gcloud_skd.trainer.util as utils
    # from gcloud_skd.trainer.MNIST_Model import MNISTConvNet
    import trainer.util as utils
    from trainer.MNIST_Model import MNISTConvNet


class MnistEpochWorker(Worker):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_loader, self.val_loader, self.test_loader = utils.get_dataloaders('MNIST')

    def compute(self, config_id, config, budget, working_directory):
        """
        if model exists in working_directory/models,
            load the model
            figure out delta budget
            run for the delta
        else,
            create a new model
            run for the passed budget
        store updated model in working_directory/models with name hp_configID_budget [hp_0_0_0_81.0.pkl]
        """
        budget /= 2
        budget = int(budget)
        model, old_budget = utils.load_model(working_directory, config_id)
        if model is None:
            old_budget = 0
            model = MNISTConvNet(num_conv_layers=config['num_conv_layers'],
                                 num_filters_1=config['num_filters_1'],
                                 num_filters_2=config['num_filters_2'] if 'num_filters_2' in config else None,
                                 num_filters_3=config['num_filters_3'] if 'num_filters_3' in config else None,
                                 dropout_rate=config['dropout_rate'],
                                 num_fc_units=config['num_fc_units'],
                                 kernel_size=3
                                 )
        device = utils.get_device()
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        delta = int(budget - old_budget)
        for epoch in range(int(delta)):
            loss = 0
            model.train()
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = F.nll_loss(output, y)
                loss.backward()
                optimizer.step()

        utils.save_model(working_directory, budget, config_id, model)

        train_accuracy = utils.evaluate_accuracy(model, self.train_loader)
        validation_accuracy = utils.evaluate_accuracy(model, self.val_loader)
        test_accuracy = utils.evaluate_accuracy(model, self.test_loader)

        return ({
            'loss': 1 - validation_accuracy,  # remember: HpBandSter always minimizes!
            'info': {'test accuracy': test_accuracy,
                     'train accuracy': train_accuracy,
                     'validation accuracy': validation_accuracy
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
        return utils.get_mnist_configspace()


if __name__ == '__main__':
    worker = MnistEpochWorker(run_id='0')
    cs = worker.get_configspace()
    number_of_epochs = 1
    config = cs.sample_configuration().get_dictionary()
    print(config)
    start_time = dt.datetime.now()
    res = worker.compute(config=config, config_id=(0, 0, 0), budget=number_of_epochs, working_directory='.')
    delta = dt.datetime.now() - start_time
    print('Time spent: ', delta)
    print(res)
