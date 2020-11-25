import logging
from math import ceil

logging.basicConfig(level=logging.DEBUG)
from hpbandster.core.worker import Worker
from torch.utils.data import DataLoader
from Models.Mnist_Model import MnistModel
from Utility.Data_Utils import *
from Utility.Worker_Utils import *
from Utility.CONFIG import *
import torch.nn.functional as F


class CNNMNISTWorker(Worker):
    def __init__(self, epc=None, **kwargs):
        super().__init__(**kwargs)
        self.trainset, self.testset = get_dataset('MNIST')
        self.trainset, self.valset = torch.utils.data.random_split(self.trainset, [MNIST_TRAINSET_SIZE, MNIST_VAL_SIZE])
        self.val_loader = dataset_to_dataloaders(self.valset)
        self.test_loader = dataset_to_dataloaders(self.testset)

        # Need to store curr_trainset as well as a list of all encountered budgets, since
        # curr_trainset needs to be consistent across all configs and the sampling to
        # curr_trainset needs to happen once per unique budget
        # The curr-loader needs to be different for each compute
        self.curr_trainset = None
        self.backup_trainset = self.trainset

        # Optimizer Interface Parameters
        self.hpo_optimizer = None

        # Trainset parameters
        self.budget_spent = 1.0
        self.fixed_execution_type = None  # This will only be set if the execution type is fixed horizon
        self.fixed_epc = None
        """
        possibilities for fixed_execution_type: 'trainset', 'epoch', 'time'
        """

        self.evals = {'budget_traversed': [], 'active_bracket': None}
        """
        Structure of evals
        {
            'active_bracket': current_active_budget
            'budget_traversed': [b1, b2, ...]
            cid_1: {'budget': b, 'loss': l, 'count': number of training samples passed },
            cid_2: {'budget': b, 'loss': l},
        }
        """

    @staticmethod
    def get_configspace():
        return get_config_space('MNIST')

    def update_fixed_exp_type(self, exp_type, epc=None):
        self.fixed_execution_type = exp_type
        self.fixed_epc = epc


    def is_new_trainset_req(self, old_budget, new_budget, new_bracket):
        if old_budget == new_budget:
            return False
        if new_bracket:
            return True
        if new_budget in self.evals:
            return False
        try:
            available_data = len(self.trainset) + (old_budget * len(self.backup_trainset))
        except:
            return True
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

    def update_trainset_params(self, config_id, budget):
        """
        Updates the self.curr_trainset and self.budget_spent as required
        :param config_id: Current config id
        :param budget: Current Budget
        :param loader_batch_size: config['batch_size']
        """
        new_bracket = config_id[0] != self.evals['active_bracket']
        if new_bracket:
            self.evals = {'budget_traversed': [], 'active_bracket': config_id[0]}

        old_b = self.evals[config_id]['budget'] if config_id in self.evals else 0

        is_new_trainset = False
        if self.is_new_trainset_req(old_b, budget, new_bracket):
            self.trainset = self.backup_trainset
            is_new_trainset = True
            self.budget_spent = 0
        if self.is_new_currset_req(is_new_trainset, budget):
            req = calculate_trainset_data_budget(old_b, budget)
            self.curr_trainset, self.trainset = get_fraction_of_data(req, self.trainset)
            self.budget_spent = 0


    def return_temp_results(self, optimizer):
        return {
            'loss': 0.9,  # remember: HpBandSter always minimizes!
            'info': {
                'epc': self.hpo_optimizer.epc,
                'trainset_budget': self.hpo_optimizer.trainset_budget,
                'epoch_multiplier': self.hpo_optimizer.epoch_multiplier,
                'test_accuracy': 0,
                'validation_accuracy': 0,
                'test_confidence': 0,
                'validation_confidence': 0,
                'trainset_consumed': 0,
                'epochs_for_time_budget': 0,
                'time_multiplier': 0
            }
        }

    def compute(self, config_id, config, budget, working_directory):
        # if self.hpo_optimizer.algo_type == 'time-based' or self.hpo_optimizer.active_test_algo == 'time-based':
        #     pass
        # else:
        #     return self.return_temp_results(None)

        if hasattr(self.hpo_optimizer, 'trainset_reset'):
            if self.hpo_optimizer.trainset_reset:
                self.trainset = self.backup_trainset
                self.hpo_optimizer.trainset_reset = False

        epoch_multiplier = 1
        time_multiplier = 1
        trainset_budget = 1
        epc = None
        epochs_for_time_budget = None
        execution_algo_type = 'epoch_based'
        if hasattr(self.hpo_optimizer, 'epoch_multiplier'):
            epoch_multiplier = self.hpo_optimizer.epoch_multiplier
        if hasattr(self.hpo_optimizer, 'trainset_budget'):
            trainset_budget = self.hpo_optimizer.trainset_budget
        if hasattr(self.hpo_optimizer, 'epc'):
            epc = self.hpo_optimizer.epc
        if hasattr(self.hpo_optimizer, 'time_multiplier'):
            time_multiplier = self.hpo_optimizer.time_multiplier
        if hasattr(self.hpo_optimizer, 'algo_type'):
            if self.hpo_optimizer.algo_type in EPOCH_BASED_METHODS \
                    or self.hpo_optimizer.active_test_algo in EPOCH_BASED_METHODS:
                execution_algo_type = 'epoch_based'
            elif self.hpo_optimizer.algo_type == 'trainset-with-increasing-epc' or \
                    self.hpo_optimizer.active_test_algo == 'trainset-with-increasing-epc':
                execution_algo_type = 'trainset_based'
            else:
                execution_algo_type = 'time_based'
        else:
            if self.fixed_execution_type == 'epoch':
                execution_algo_type = 'epoch_based'
            elif self.fixed_execution_type == 'trainset':
                execution_algo_type = 'trainset_based'
                epc = self.fixed_epc
            elif self.fixed_execution_type == 'time':
                execution_algo_type = 'time_based'

        if execution_algo_type == 'epoch_based':
            e = budget * epoch_multiplier
            trainset_consumed = trainset_budget * e
            self.update_trainset_params(config_id, trainset_budget)
            temp_set = self.curr_trainset
            self.budget_spent += trainset_budget
            self.evals[config_id] = {'budget': trainset_budget}


        elif execution_algo_type == 'trainset_based':
            e = epc
            trainset_consumed = budget * e
            self.update_trainset_params(config_id, budget)
            temp_set = self.curr_trainset
            self.budget_spent += budget
            self.evals[config_id] = {'budget': budget}

        else:
            trainset_consumed = 1
            self.update_trainset_params(config_id, trainset_budget)
            temp_set = self.curr_trainset
            self.budget_spent += trainset_budget
            self.evals[config_id] = {'budget': trainset_budget}

        curr_loader = dataset_to_dataloaders(temp_set,
                                             batch_sizes=[config['batch_size']],
                                             shuffles=[True])

        if REUSE_MODEL:
            model, old_budget = load_model(working_directory, config_id)
        else:
            model = None
            old_budget = 0

        if model is None:
            model = MnistModel(num_conv_layers=config['num_conv_layers'],
                               num_filters_1=config['num_filters_1'],
                               num_filters_2=config['num_filters_2'] if 'num_filters_2' in config else None,
                               num_filters_3=config['num_filters_3'] if 'num_filters_3' in config else None,
                               dropout_rate=config['dropout_rate'],
                               num_fc_units=config['num_fc_units'],
                               kernel_size=3
                               )
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        criterion = F.nll_loss

        if DEBUG:
            return return_fake_results(config_id, self.hpo_optimizer, trainset_consumed)


        elif execution_algo_type == 'time_based':
            dur = budget * time_multiplier
            model, epochs_for_time_budget = train_model_for_duration(curr_loader, model, dur, optimizer, criterion)
            trainset_consumed = epochs_for_time_budget * trainset_budget

        else:
            model = train_model_on_data(curr_loader, model, e, optimizer, criterion)

        if REUSE_MODEL:
            save_model(working_directory, budget, config_id, model)

        validation_accuracy, validation_confidence = evaluate_accuracy(model, self.val_loader)
        test_accuracy, test_confidence = evaluate_accuracy(model, self.test_loader)

        return {
            'loss': 1 - validation_accuracy,  # remember: HpBandSter always minimizes!
            'info': {
                'epc': epc,
                'trainset_budget': trainset_budget,
                'epoch_multiplier': epoch_multiplier,
                'epochs_for_time_budget': epochs_for_time_budget,
                'time_multiplier': time_multiplier,
                'test_accuracy': test_accuracy,
                'validation_accuracy': validation_accuracy,
                'test_confidence': test_confidence,
                'validation_confidence': validation_confidence,
                'trainset_consumed': trainset_consumed
            }
        }


if __name__ == '__main__':
    worker = CNNMNISTWorker(run_id='0')
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    # start_time = dt.datetime.now()
    res = worker.compute(config=config, config_id=(0, 0, 0), budget=0.1, working_directory='.')
    # delta = dt.datetime.now() - start_time
    # print('Time spent: ', delta)
    print(res)
