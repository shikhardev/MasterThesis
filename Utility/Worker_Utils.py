from random import uniform as r
# from numpy.random import uniform as r
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hpbandster.optimizers import RandomSearch, BOHB, HyperBand
from scipy.special import softmax
# from tensorflow.python.lib.io import file_io
from datetime import datetime as dt
from Optimizers.EpochWithIncreasingTrainset import EpochWithIncreasingTrainset
from Optimizers.TimeWithIncreasingTrainset import TimeWithIncreasingTrainset
from Optimizers.Multitune_NoBayesian import Multitune_NoBayesian
from Optimizers.Multitune import Multitune
from Optimizers.TrainsetWithIncreasingEPC import TrainsetWithIncreasingEPC
# from Optimizers.temp_Multitune import Multitune as temp_multitune
from Optimizers.Multitune_2_Exploration import Multitune_2_Exploration
from hpbandster.optimizers.bohb import BOHB
from Utility.CONFIG import SEED

# if GCLOUD:
#     from tensorflow_core.python.lib.io.file_io import delete_file_v2 as remove
#     from tensorflow_core.python.lib.io.file_io import get_matching_files_v2 as glob
#     from tensorflow.io.gfile import makedirs, isdir
# else:
#     from glob import glob
#     from os import makedirs, remove
#     from os.path import isdir
from glob import glob
from os import makedirs, remove
from os.path import isdir


def get_device(use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        print('\n', 'USING GPU!', '\n')
        device = 'cuda'
    else:
        print('\n' * 10, 'NOT USING GPU or GPU unavailable!', '\n' * 10)
        device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    return device


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    device = get_device()
    confidence_array = np.array([])
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            s = np.array(softmax(output.cpu(), axis=1))
            c = np.amax(s, axis=1)
            confidence_array = np.concatenate((confidence_array, c))
            correct += pred.eq(y.view_as(pred)).sum().item()
    accuracy = correct / len(dataloader.sampler)
    avg_confidence = np.average(confidence_array)
    return accuracy, avg_confidence


def save_model(work_dir, budget, config_id, model):
    """
    If a model with current config id already exists, delete existing model
    Store new model
    """
    config_id = '_'.join([str(i) for i in config_id])
    model_file_name = work_dir + '/models/'
    model_file_name += 'hp_' + config_id + '_'
    g = glob(model_file_name + '*.pkl')
    for f in g:
        remove(f)

    if not isdir(work_dir + '/models'):
        makedirs(work_dir + '/models')

    model_file_name += str(budget) + '.pkl'
    torch.save(model, model_file_name)
    return True


def load_model(work_dir, config_id):
    """
    If model has been trained with config of config_id previously,
        return the model with max budget
    Else
        return None, None
    """
    config_id = '_'.join([str(i) for i in config_id])
    model_file_name = work_dir + '/models/'
    model_file_name += 'hp_' + config_id
    g = glob(model_file_name + "*.pkl")
    if len(g) == 0:
        return None, 0
    else:
        # Assume ony single model is available.
        # Return the model and the budget
        model = torch.load(g[0])

        budget = g[0].split('_')[-1]
        budget = budget.rstrip('.pkl')
        budget = float(budget)
        return model, budget


def get_config_space(dataset):
    cs = CS.ConfigurationSpace(seed=SEED)

    if dataset == 'MNIST':
        # Optimizer parameters
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
        batch_size = CSH.CategoricalHyperparameter('batch_size', [64, 128, 256, 512])
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9,
                                                      log=False)
        cs.add_hyperparameters([lr, optimizer, batch_size, sgd_momentum])
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        # Architecture parameters
        num_conv_layers = CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=3, default_value=2)
        num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=64, default_value=16, log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=64, default_value=16, log=True)
        num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=64, default_value=16, log=True)
        cs.add_hyperparameters([num_conv_layers, num_filters_1, num_filters_2, num_filters_3])
        cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
        cs.add_condition(cond)
        cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
        cs.add_condition(cond)

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5,
                                                      log=False)
        num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)
        cs.add_hyperparameters([dropout_rate, num_fc_units])

    elif dataset == 'CIFAR10':
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, default_value='1e-2', log=True)
        hidden_nodes = CSH.UniformIntegerHyperparameter('hidden_nodes', lower=20, upper=100)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [64, 128, 256, 512])
        conv1_channels = CSH.CategoricalHyperparameter('conv1_channels', [32, 64, 128])
        conv2_channels = CSH.CategoricalHyperparameter('conv2_channels', [64, 128, 256, 512])
        cs.add_hyperparameters([lr, hidden_nodes, batch_size, conv1_channels, conv2_channels])

    elif dataset == 'FASHION':
        cs = CS.ConfigurationSpace()
        dropout = CSH.UniformFloatHyperparameter('dropout', lower=0.1, upper=0.8)
        channels_one = CSH.CategoricalHyperparameter('channels_one', [10, 12, 14, 16, 18, 20])
        batch_size = CSH.CategoricalHyperparameter('batch_size', [64, 128, 256, 512, 1024])
        channels_two = CSH.CategoricalHyperparameter('channels_two', [24, 28, 32, 36, 40, 44])
        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-5, upper=1e-1, default_value='1e-2', log=True)
        cs.add_hyperparameters([lr, channels_one, channels_two, dropout, batch_size])

    return cs


def get_fixed_optimizer(config_space, method, **kwargs):
    d = {
        'randomsearch': RandomSearch,
        'bohb': BOHB,
        'trainset_bohb': BOHB,
        'epoch_bohb': BOHB,
        'time_bohb': BOHB,
        'hyperband': HyperBand
    }
    if method not in d:
        raise ValueError("Unknown method %s" % method)

    opt = d[method]
    return opt(config_space, **kwargs)


def get_optimizer(config_space, method, **kwargs):
    d = {
        'EpochWithIncreasingTrainset': EpochWithIncreasingTrainset,
        'TimeWithIncreasingTrainset': TimeWithIncreasingTrainset,
        'TrainsetWithIncreasingEPC': TrainsetWithIncreasingEPC,
        'Multitune': Multitune,
        'Multitune_2_Exploration': Multitune_2_Exploration,
        'Multitune_NoBayesian': Multitune_NoBayesian
    }
    if method not in d:
        raise ValueError("Unknown method %s" % method)

    opt = d[method]
    return opt(config_space, **kwargs)


def train_model_on_data(data_loader, model, epochs, optimizer,
                        criterion=nn.CrossEntropyLoss):
    device = get_device()
    model.to(device)
    epochs = int(epochs)

    for e in range(epochs):
        loss = 0
        model.train()
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    return model


def train_model_for_duration(data_loader, model, duration, optimizer,
                             criterion=nn.CrossEntropyLoss):
    """
    :param duration: In minutes
    :return: trained model
    """
    duration = duration * 60  # Converting to seconds
    device = get_device()
    model.to(device)
    start_time = dt.now()
    epochs = 0
    duration_expired = False
    total_len = len(data_loader)

    while True:
        loss = 0
        model.train()
        curr_len = 0
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            curr_len += 1
            if (dt.now() - start_time).seconds >= duration:
                duration_expired = True
                break

        epochs += curr_len / total_len
        if duration_expired:
            break

    return model, epochs


def return_fake_results_old(config_id, optimizer, trainset_consumed=0, budget=0, epochs_for_time_budget=2):
    prioritize_algo = None
    # prioritize_algo = 'time-based'
    # prioritize_algo = 'trainset-with-increasing-epc'
    # prioritize_algo = 'epoch-with-increasing-trainset'
    subtractor = 0
    if config_id[0] < 5:
        validation_accuracy = 0.8
    elif config_id[0] < 10:
        validation_accuracy = 0.9
    else:
        validation_accuracy = 0.95

    if prioritize_algo is not None:
        if optimizer.algo_type == prioritize_algo or \
                optimizer.active_test_algo == prioritize_algo:
            subtractor = 0
        else:
            subtractor = 0.8

    validation_accuracy -= subtractor

    test_accuracy = 1
    val_confidence = 1
    test_confidence = 1
    epoch_multiplier = None
    epc = None
    trainset_budget = None
    duration = None

    if optimizer is not None:
        epoch_multiplier = optimizer.epoch_multiplier if hasattr(optimizer, 'epoch_multiplier') else None
        epc = optimizer.epc if hasattr(optimizer, 'epc') else None
        trainset_budget = optimizer.trainset_budget if hasattr(optimizer, 'trainset_budget') else None
        time_multiplier = optimizer.time_multiplier if hasattr(optimizer, 'time_multiplier') else None
        duration = budget * time_multiplier if time_multiplier is not None else None

    return {
        'loss': 1 - validation_accuracy,  # remember: HpBandSter always minimizes!
        'info': {'test_accuracy': test_accuracy,
                 'validation_accuracy': validation_accuracy,
                 'test_confidence': test_confidence,
                 'validation_confidence': val_confidence,
                 'epoch_multiplier': epoch_multiplier,
                 'epc': epc,
                 'trainset_budget': trainset_budget,
                 'trainset_consumed': trainset_consumed,
                 'time_multiplier': time_multiplier,
                 'epochs_for_time_budget': epochs_for_time_budget,
                 'duration': duration
                 }

    }


def return_fake_results(config_id, optimizer, trainset_consumed=0, budget=0, epochs_for_time_budget=2):
    if config_id[0] < 5:
        # validation_accuracy = 0.6211
        validation_accuracy = r(0, 0.6211)
    elif config_id[0] < 10:
        validation_accuracy = r(0.6211, 0.6766)
    elif config_id[0] < 14:
        validation_accuracy = r(0.6766, 0.7066)
    else:
        validation_accuracy = r(0.6864, 0.7066)

    prioritize_algo = None
    # prioritize_algo = 'time-based'
    # prioritize_algo = 'trainset-with-increasing-epc'
    # prioritize_algo = 'epoch-with-increasing-trainset'
    subtractor = 0
    if prioritize_algo is not None:
        if optimizer.algo_type == prioritize_algo or \
                optimizer.active_test_algo == prioritize_algo:
            subtractor = 0
        else:
            subtractor = 0.5

    validation_accuracy -= subtractor


    test_accuracy = 1
    val_confidence = 1
    test_confidence = 1
    epoch_multiplier = None
    time_multiplier = None
    epc = None
    trainset_budget = None
    duration = None

    if optimizer is not None:
        epoch_multiplier = optimizer.epoch_multiplier if hasattr(optimizer, 'epoch_multiplier') else None
        epc = optimizer.epc if hasattr(optimizer, 'epc') else None
        trainset_budget = optimizer.trainset_budget if hasattr(optimizer, 'trainset_budget') else None
        time_multiplier = optimizer.time_multiplier if hasattr(optimizer, 'time_multiplier') else None
        duration = budget * time_multiplier if time_multiplier is not None else None

    return {
        'loss': 1 - validation_accuracy,  # remember: HpBandSter always minimizes!
        'info': {'test_accuracy': test_accuracy,
                 'validation_accuracy': validation_accuracy,
                 'test_confidence': test_confidence,
                 'validation_confidence': val_confidence,
                 'epoch_multiplier': epoch_multiplier,
                 'epc': epc,
                 'trainset_budget': trainset_budget,
                 'trainset_consumed': trainset_consumed,
                 'time_multiplier': time_multiplier,
                 'epochs_for_time_budget': epochs_for_time_budget,
                 'duration': duration
                 }

    }


def run_optimizer_process(run_id, worker, method, res_dir, work_dir, verbose, fixed_budget=False, num_iterations=0,
                          epc=None, **kwargs):
    """
    Optimizes the worker using specified method
    :param fixed_budget: True if the optimization is to be run for fixed budget, else false
    :param method: Optimization method {'randomsearch', 'bohb', 'hyperband'}
    :param run_id: Identifier for the current run of the method
    :param min_budget: Minimum budget
    :param max_budget: Max budget
    :param worker: Worker for the current experiment {Worker object}
    :param res_dir: Directory where all results are to be stored
    :param verbose: Do you want to store all HP config details of the current run? {True, False}
    :param eta: SH ratio [Refer to thesis document]
    :param num_iterations: Number of Successive Halving iterations
    :param work_dir: Location for temporary files required during the run
    :return: None
    """
    # Start a nameserver:
    NS = hpns.NameServer(run_id=run_id, host='127.0.0.1', port=0, working_directory=work_dir)
    ns_host, ns_port = NS.start()

    # w = worker(run_id=run_id, timeout=120)
    # w.load_nameserver_credentials(work_dir=work_dir)
    # w.run(background=False)
    # print ("HPNS started")
    # exit(0)

    w = worker(run_id=run_id, host='127.0.0.1', nameserver=ns_host, nameserver_port=ns_port, timeout=120)
    w.run(background=True)

    # result_logger = hpres.json_result_logger(directory=work_dir, overwrite=False)

    if fixed_budget:
        optimizer = get_fixed_optimizer(config_space=worker.get_configspace(), method=method, run_id=run_id,
                                        working_directory=work_dir,
                                        host=ns_host, nameserver=ns_host,
                                        nameserver_port=ns_port,
                                        ping_interval=3600,
                                        **kwargs)
        # if fixed_budget == 'trainset':
        #     w.update_fixed_exp_type(fixed_budget, epc)
        if fixed_budget:
            w.update_fixed_exp_type(fixed_budget, epc)
    else:
        optimizer = get_optimizer(config_space=worker.get_configspace(), method=method, run_id=run_id,
                                  time_deadline=120, acc_saturation_check=True, working_directory=work_dir,
                                  host=ns_host, nameserver=ns_host, nameserver_port=ns_port, ping_interval=3600)

    if hasattr(w, 'hpo_optimizer'):
        w.hpo_optimizer = optimizer

    if fixed_budget:
        res = optimizer.run(n_iterations=num_iterations)
    else:
        res = optimizer.run()
    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()
    return res
