import datetime as dt
import torch
import torch.nn.functional as F
from hpbandster.core.worker import Worker
from torch.utils.data import DataLoader

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


def calculate_data_budget(old_budget, required_budget):
    """
    x % of remaining data = required data. Required data is not 100%.
    doesn't So, 50% budget doesn't really mean 50% of current data, since
    chunks of data have been discarded in the previous budget.
    This function returns the percentage of data to be taken from the remaining dataset
    to match the original required budget
    :param old_budget: The percentage of the whole dataset that has already been used
    :param required_budget: Budget requirement for the rung, in respect to the whole dataset
    :return: Budget requirement fot the rung in respect to the remaining dataset
    """
    delta = required_budget - old_budget
    remaining_budget = 1.0 - old_budget
    # x % of remaining_budget = delta
    x = delta / remaining_budget
    return x


def get_fraction_of_data(fraction, dataset):
    """
    Returns two datasets split from input dataset in the input fraction
    :return: d1, d2:    d1 is the dataset to be used for this evaluation,
                        d2 is the remaining dataset
    """
    total_size = len(dataset)
    s1 = int(fraction * total_size)
    s2 = total_size - s1
    d1, d2 = torch.utils.data.random_split(dataset, [s1, s2])
    return d1, d2


class MnistTrainsetWorker(Worker):
    def __init__(self, **kwargs):
        TEST_BATCH_SIZE = 1024
        NUM_WORKER = 4
        if torch.cuda.is_available():
            PIN_MEMORY = True
        else:
            PIN_MEMORY = False
        super().__init__(**kwargs)
        self.trainset, self.testset = utils.get_dataset('MNIST')
        self.trainset, self.valset = torch.utils.data.random_split(self.trainset, [50000, 10000])
        self.val_loader = DataLoader(self.valset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)
        self.test_loader = DataLoader(self.testset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)

        # Need to store curr_trainset as well as a list of all encountered budgets, since
        # curr_trainset needs to be consistent across all configs and the sampling to
        # curr_trainset needs to happen once per unique budget
        self.curr_trainset = None
        self.backup_trainset = self.trainset
        self.curr_loader = None
        self.budget_spent = 1.0
        self.evals = {'budget_traversed': [], 'active_bracket': None}
        """
        Structure of evals
        {
            'active_bracket': current_active_budget
            'budget_traversed': [b1, b2, ...]
            cid_1: {'budget': b, 'loss': l},
            cid_2: {'budget': b, 'loss': l},
        }
        """

    def is_new_bracket(self, config_id):
        if config_id[0] == self.evals['active_bracket']:
            return False
        else:
            return True

    def get_budget_and_loss(self, config_id):
        """
        Returns the budget and the loss
        :param config_id: config id for which budget and loss is required
        :return: budget, loss
        """
        if config_id not in self.evals:
            return 0,0
        else:
            return self.evals[config_id]['budget'], self.evals[config_id]['loss']

    def train_model_on_data(self, config, data_loader, model=None):
        """
        Returns a trained model for the given data evaluated for epochs_per_config
        :param model: Pre-trained model if exists
        """
        if DEBUG:
            EPOCHS_PER_CONFIG = 1
        else:
            EPOCHS_PER_CONFIG = 16
        device = utils.get_device()
        if model is None:
            model = MNISTConvNet(num_conv_layers=config['num_conv_layers'],
                                 num_filters_1=config['num_filters_1'],
                                 num_filters_2=config['num_filters_2'] if 'num_filters_2' in config else None,
                                 num_filters_3=config['num_filters_3'] if 'num_filters_3' in config else None,
                                 dropout_rate=config['dropout_rate'],
                                 num_fc_units=config['num_fc_units'],
                                 kernel_size=3
                                 )
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        for epoch in range(EPOCHS_PER_CONFIG):
            loss = 0
            model.train()
            for i, (x, y) in enumerate(data_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = F.nll_loss(output, y)
                loss.backward()
                optimizer.step()
        return model

    def compute(self, config_id, config, budget, working_directory):
        TRAIN_BATCH_SIZE = 256
        NUM_WORKER = 4
        if torch.cuda.is_available():
            PIN_MEMORY = True
        else:
            PIN_MEMORY = False
        new_bracket = self.is_new_bracket(config_id)
        if new_bracket:
            print(self.evals)
            self.evals = {'budget_traversed': [], 'active_bracket': config_id[0]}

        old_b, old_l = self.get_budget_and_loss(config_id)
        is_new_trainset = False
        if self.is_new_trainset_req(old_b, budget, new_bracket):
            self.trainset = self.backup_trainset
            is_new_trainset = True
            self.budget_spent = 0
        if self.is_new_currset_req(is_new_trainset, budget):
            req = calculate_data_budget(old_b, budget)
            self.curr_trainset, self.trainset = get_fraction_of_data(req, self.trainset)
            self.curr_loader = DataLoader(dataset=self.curr_trainset, batch_size=TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=NUM_WORKER, pin_memory=PIN_MEMORY)
            self.budget_spent = 0
        self.budget_spent += budget

        model, old_budget = utils.load_model(working_directory, config_id)
        model = self.train_model_on_data(config, self.curr_loader, model=model)
        utils.save_model(working_directory, budget, config_id, model)

        train_accuracy = utils.evaluate_accuracy(model, self.curr_loader)
        validation_accuracy = utils.evaluate_accuracy(model, self.val_loader)
        test_accuracy = utils.evaluate_accuracy(model, self.test_loader)

        old_count = 0
        if config_id in self.evals:
            old_count = self.evals[config_id]['count']

        self.evals[config_id] = {'budget': budget, 'loss': 1 - validation_accuracy, 'count': old_count + len(self.curr_trainset)}

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

    def is_new_trainset_req(self, old_budget, new_budget, new_bracket):
        if old_budget == new_budget:
            return False
        if new_bracket:
            return True
        if new_budget in self.evals:
            return False
        available_data = len(self.trainset) + (old_budget * len(self.backup_trainset))
        required_data = new_budget * len(self.backup_trainset)
        if available_data < required_data:
            return True
        return False

    def is_new_currset_req(self, is_new_trainset, budget):
        """
        returns true if new curr_set is required
        :param is_new_trainset: has a new training set been drawn
        :param budget: current budget that the config is being evaluated for
        """
        if is_new_trainset:
            self.evals['budget_traversed'].append(budget)
            return True
        if budget not in self.evals['budget_traversed']:
            self.evals['budget_traversed'].append(budget)
            return True
        return False


if __name__ == '__main__':
    worker = MnistTrainsetWorker(run_id='0')
    cs = worker.get_configspace()
    c = cs.sample_configuration().get_dictionary()
    print(c)
    start_time = dt.datetime.now()
    res = worker.compute(config=c, config_id=(0, 0, 0), budget=0.01, working_directory='.')
    delta = dt.datetime.now() - start_time
    print('Time spent: ', delta)
    print(res)
