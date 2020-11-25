import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch import cat
import torch.nn.functional as F
from sklearn.externals import joblib
from os.path import isdir
from os import makedirs, remove
from glob import glob
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import pickle
import pandas as pd
from tensorflow.python.lib.io import file_io
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import RandomSearch, BOHB, HyperBand
import numpy as np
import hpbandster.core.result as hpres


def downsample_dataset(dataset, dim):
    """
    Downsamples each of the images in the dataset to the input dimension
    """
    TRANSFORM_BATCH_SIZE = 4096
    loader = data_utils.DataLoader(dataset=dataset, batch_size=TRANSFORM_BATCH_SIZE, shuffle=False)
    data_tensor = None
    lab_tensor = None
    for i, (x, y) in enumerate(loader):
        temp_x = F.interpolate(x, size=(dim, dim))
        if data_tensor is None:
            data_tensor = temp_x
            lab_tensor = y
        else:
            data_tensor = cat((data_tensor, temp_x), 0)
            lab_tensor = cat((lab_tensor, y), 0)
    res = data_utils.TensorDataset(data_tensor, lab_tensor)
    return res


def get_dataset(dataset_name):
    """
    Returns the train and test dataset
    """
    DATA_LOCATION = '~/Data/' + dataset_name
    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=DATA_LOCATION, train=True, transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = torchvision.datasets.MNIST(root=DATA_LOCATION, train=False, transform=transforms.ToTensor(),
                                                  download=True)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root=DATA_LOCATION, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=DATA_LOCATION, train=False, transform=transform, download=True)

    return train_dataset, test_dataset


def get_dataloaders(dataset_name, down_sampled_size=None, train_subset=1.0, train_batch_size=256):
    """
    Returns train, validate and test dataloaders
    :param dataset_name: Name of the dataset ['MNIST', 'CIFAR10']
    :param down_sampled_size:    If none, does not perform any image downsampling.
                                If value, downsamples each image to size val * val. Assumes all images are square
    :param train_subset:    If none, returns the dataloader with default train-validate-test split
                            (see code for dataset specific split)
                            Else, returns default val and test  dataloader and subset of train dataloader
    :param train_batch_size: training data batch size
    """

    if dataset_name == 'MNIST':
        train_dataset, test_dataset = get_dataset('MNIST')
        train_size = 50000
        val_size = 10000
        full_training_size = 60000
    # elif dataset_name == 'CIFAR10':
    else:
        # This is for CIFAR 10
        train_dataset, test_dataset = get_dataset('CIFAR10')
        train_size = 40000
        val_size = 10000
        full_training_size = 50000

    # if down_sampled_size is not None:
    #     train_dataset = downsample_dataset(train_dataset, down_sampled_size)
    #     test_dataset = downsample_dataset(test_dataset, down_sampled_size)

    train, val, test = datasets_to_dataloaders(trainset=train_dataset, testset=test_dataset,
                                               train_size=int(train_size / train_subset),
                                               val_size=val_size, full_training_size=full_training_size,
                                               train_batch_size=train_batch_size)
    return train, val, test


def construct_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    res = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                     num_workers=num_workers, pin_memory=pin_memory)
    return res


def datasets_to_dataloaders(trainset, testset, train_size, val_size, full_training_size, train_batch_size):
    """
    Returns the train, validate and test data loaders
    """
    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    num_workers = 4
    test_batch_size = 1024
    val_batch_size = 1024

    test_loader = DataLoader(dataset=testset, batch_size=test_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    if train_size + val_size == full_training_size:
        train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    else:
        train_dataset, _ = torch.utils.data.random_split(trainset, [train_size, (full_training_size - train_size)])
        _, val_dataset = torch.utils.data.random_split(trainset, [(full_training_size - val_size), val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    return device


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    device = get_device()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    accuracy = correct / len(dataloader.sampler)
    return accuracy


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
        model = joblib.load(g[0])
        budget = g[0].split('_')[-1]
        budget = budget.rstrip('.pkl')
        budget = float(budget)
        return model, budget


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
    joblib.dump(model, model_file_name)
    return True


def get_mnist_configspace():
    """
    Returns the config space of experiment designed for MNIST

    In this example implements a small CNN in PyTorch to train it on MNIST.
    The configuration space shows the most common types of hyperparameters and
    even contains conditional dependencies.


    We'll optimise the following hyperparameters:

    +-------------------------+----------------+-----------------+------------------------+
    | Parameter Name          | Parameter type |  Range/Choices  | Comment                |
    +=========================+================+=================+========================+
    | Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
    +-------------------------+----------------+-----------------+------------------------+
    | Optimizer               | categorical    | {Adam, SGD }    | discrete choice        |
    +-------------------------+----------------+-----------------+------------------------+
    | SGD momentum            |  float         | [0, 0.99]       | only active if         |
    |                         |                |                 | optimizer == SGD       |
    +-------------------------+----------------+-----------------+------------------------+
    | Number of conv layers   | integer        | [1,3]           | can only take integer  |
    |                         |                |                 | values 1, 2, or 3      |
    +-------------------------+----------------+-----------------+------------------------+
    | Number of filters in    | integer        | [4, 64]         | logarithmically varied |
    | the first conf layer    |                |                 | integer values         |
    +-------------------------+----------------+-----------------+------------------------+
    | Number of filters in    | integer        | [4, 64]         | only active if number  |
    | the second conf layer   |                |                 | of layers >= 2         |
    +-------------------------+----------------+-----------------+------------------------+
    | Number of filters in    | integer        | [4, 64]         | only active if number  |
    | the third conf layer    |                |                 | of layers == 3         |
    +-------------------------+----------------+-----------------+------------------------+
    | Dropout rate            |  float         | [0, 0.9]        | standard continuous    |
    |                         |                |                 | parameter              |
    +-------------------------+----------------+-----------------+------------------------+
    | Number of hidden units  | integer        | [8,256]         | logarithmically varied |
    | in fully connected layer|                |                 | integer values         |
    +-------------------------+----------------+-----------------+------------------------+

    """
    cs = CS.ConfigurationSpace()

    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

    # For demonstration purposes, we add different optimizers as categorical hyperparameters.
    # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
    # SGD has a different parameter 'momentum'.
    optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

    sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9,
                                                  log=False)

    cs.add_hyperparameters([lr, optimizer, sgd_momentum])

    # The hyperparameter sgd_momentum will be used,if the configuration
    # contains 'SGD' as optimizer.
    cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
    cs.add_condition(cond)

    num_conv_layers = CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=3, default_value=2)

    num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=64, default_value=16, log=True)
    num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=64, default_value=16, log=True)
    num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=64, default_value=16, log=True)

    cs.add_hyperparameters([num_conv_layers, num_filters_1, num_filters_2, num_filters_3])

    # You can also use inequality conditions:
    cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
    cs.add_condition(cond)

    dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5,
                                                  log=False)
    num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)

    cs.add_hyperparameters([dropout_rate, num_fc_units])

    return cs


def get_cifar_configspace():
    """
        hyperparameter_ranges = {
            'lr': ContinuousParameter(0.0001, 0.01),
            'hidden_nodes': IntegerParameter(20, 100),
            'batch_size': CategoricalParameter([128, 256, 512]),
            'conv1_channels': CategoricalParameter([32, 64, 128]),
            'conv2_channels': CategoricalParameter([64, 128, 256, 512]),
        }

    :return:
    """
    cs = CS.ConfigurationSpace()
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, default_value='1e-2', log=True)
    hidden_nodes = CSH.UniformIntegerHyperparameter('hidden_nodes', lower=20, upper=100)
    batch_size = CSH.CategoricalHyperparameter('batch_size', [128, 256, 512])
    conv1_channels = CSH.CategoricalHyperparameter('conv1_channels', [32, 64, 128])
    conv2_channels = CSH.CategoricalHyperparameter('conv2_channels', [64, 128, 256, 512])
    cs.add_hyperparameters([lr, hidden_nodes, batch_size, conv1_channels, conv2_channels])
    return cs


def standard_parser_args(parser):
    parser.add_argument('--res_dir',
                        type=str,
                        help='the destination directory to store all results',
                        default='../results/')

    parser.add_argument('--num_iterations',
                        type=int,
                        help='number of Hyperband iterations performed.',
                        default=5)
    parser.add_argument('--run_id',
                        type=str,
                        default=0)
    parser.add_argument('--work_dir',
                        type=str,
                        help='Directory holding live rundata. Should be shared across all nodes for parallel '
                             'optimization.',
                        default='/work/')
    parser.add_argument('--method',
                        type=str,
                        default='bohb',
                        help='Possible choices: randomsearch, bohb, hyperband, tpe, smac')
    parser.add_argument('--nic_name',
                        type=str,
                        default='lo',
                        help='name of the network interface used for communication. Note: default is only for local '
                             'execution on *nix!')

    parser.add_argument('--min_budget', type=float, help='Minimum budget for Hyperband and BOHB.')
    parser.add_argument('--max_budget', type=float, help='Maximum budget for all methods.')
    parser.add_argument('--eta', type=float, help='Eta value for Hyperband/BOHB.', default=3)

    return parser


def get_optimizer(eta, config_space, method, **kwargs):
    eta = eta
    opt = None

    if method == 'randomsearch':
        opt = RandomSearch

    if method == 'bohb':
        opt = BOHB

    if method == 'hyperband':
        opt = HyperBand

    if opt is None:
        raise ValueError("Unknown method %s" % method)

    return opt(config_space, eta=eta, **kwargs)


def extract_results_to_pickle(results_object):
    """
    Returns the best configurations over time, but also returns the cummulative budget
    Parameters:
        -----------
            all_budgets: bool
                If set to true all runs (even those not with the largest budget) can be the incumbent.
                Otherwise, only full budget runs are considered

        Returns:
        --------
            dict:
                dictionary with all the config IDs, the times the runs
                finished, their respective budgets, and corresponding losses
    :param results_object:
    :return:
    """
    all_runs = results_object.get_all_runs(only_largest_budget=False)
    all_runs.sort(key=lambda r: r.time_stamps['finished'])

    return_dict = {'config_ids': [],
                   'times_finished': [],
                   'budgets': [],
                   'losses': [],
                   'test_losses': [],
                   'cummulative_budget': [],
                   'cummulative_cost': []
                   }

    cummulative_budget = 0
    cummulative_cost = 0
    current_incumbent = float('inf')
    incumbent_budget = -float('inf')

    for r in all_runs:

        cummulative_budget += r.budget
        try:
            cummulative_cost += r.info['cost']
        except:
            pass

        if r.loss is None: continue

        if r.budget >= incumbent_budget and r.loss < current_incumbent:
            current_incumbent = r.loss
            incumbent_budget = r.budget

            return_dict['config_ids'].append(r.config_id)
            return_dict['times_finished'].append(r.time_stamps['finished'])
            return_dict['budgets'].append(r.budget)
            return_dict['losses'].append(r.loss)
            return_dict['cummulative_budget'].append(cummulative_budget)
            return_dict['cummulative_cost'].append(cummulative_cost)
            try:
                return_dict['test_losses'].append(r.info['test_loss'])
            except:
                pass

    if current_incumbent != r.loss:
        r = all_runs[-1]

        return_dict['config_ids'].append(return_dict['config_ids'][-1])
        return_dict['times_finished'].append(r.time_stamps['finished'])
        return_dict['budgets'].append(return_dict['budgets'][-1])
        return_dict['losses'].append(return_dict['losses'][-1])
        return_dict['cummulative_budget'].append(cummulative_budget)
        return_dict['cummulative_cost'].append(cummulative_cost)
        try:
            return_dict['test_losses'].append(return_dict['test_losses'][-1])
        except:
            pass

    return_dict['configs'] = {}

    id2conf = results_object.get_id2config_mapping()

    for c in return_dict['config_ids']:
        return_dict['configs'][c] = id2conf[c]

    return_dict['HB_config'] = results_object.HB_config

    return (return_dict)


def store_verbose_results(res, location, fileName=None):
    all_runs = res.get_all_runs(only_largest_budget=False)
    all_runs.sort(key=lambda r: r.time_stamps['finished'])
    id2conf = res.get_id2config_mapping()
    headers = ['cid', 'config', 'model_based', 'budget', 'tstart', 'tfinish', 'cost', 'validation_loss', 'test_loss']
    if fileName is None:
        fileName = 'results'
    res = []
    for run in all_runs:
        row = {'cid': run.config_id}
        cid = run.config_id
        row['config'] = id2conf[cid]['config']

        if 'model_based' in id2conf[cid]['config_info']:
            row['model_based'] = id2conf[cid]['config_info']['model_based_pick']

        row['budget'] = run.budget

        row['tstart'] = run.time_stamps['started']

        row['tfinish'] = run.time_stamps['finished']

        if "cost" in run.info:
            row['cost'] = run.info['cost']

        row['validation_loss'] = run.loss

        if 'test_loss' in run.info:
            row['test_loss'] = run.info['test_loss']

        res.append(row)

    print("Verbose results ready!")
    # with open(location + fileName, 'w+') as f:
    #     pickle.dump(pd.DataFrame(res), f)

    with file_io.FileIO(location + fileName, mode='w+') as f:
        pickle.dump(pd.DataFrame(res), f)


def store_experiment_details(file_location, config):
    if not isdir(file_location):
        makedirs(file_location)
    with file_io.FileIO(file_location + "experiment_details.pkl", mode='w+') as f:
        pickle.dump(config, f)


def run_optimizer_process(method, run_id, min_budget, max_budget, worker, res_dir, work_dir, verbose,
                          eta, num_iterations=5):
    """
    Optimizes the worker using specified method
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

    optimizer = get_optimizer(eta, w.get_configspace(), method=method,
                              run_id=run_id,
                              min_budget=min_budget, max_budget=max_budget,
                              host=ns_host,
                              nameserver=ns_host,
                              nameserver_port=ns_port,
                              ping_interval=3600,
                              # result_logger=result_logger,
                              working_directory=work_dir
                              )

    # store_experiment_details(res_dir, optimizer.config)
    res = optimizer.run(n_iterations=num_iterations)

    import IPython
    IPython.embed()

    with file_io.FileIO(res_dir + '{}_run_{}.pkl'.format(method, run_id), mode='w+') as f:
        pickle.dump(extract_results_to_pickle(res), f)

    if verbose:
        store_verbose_results(res, res_dir, '{}_run_{}_verbose.pkl'.format(method, run_id))

    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()
    return res

    # with open(os.path.join(res_dir, '{}_run_{}.pkl'.format(method, run_id)), 'wb') as fh:
    #     pickle.dump(extract_results_to_pickle(res), fh)


def predict_bohb_run(min_budget, max_budget, eta, n_iterations):
    """
    Prints the expected numbers of configurations, runs and budgets given BOBH's hyperparameters.
    Parameters
    ----------
    min_budget
        The smallest budget to consider.
    max_budget
        The largest budget to consider.
    eta
        The eta parameter. Determines how many configurations advance to the next round.
    n_iterations
        How many iterations of SuccessiveHalving to perform.
    """
    s_max = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1

    n_runs = 0
    n_configurations = []
    initial_budgets = []
    for iteration in range(n_iterations):
        s = s_max - 1 - (iteration % s_max)

        initial_budget = (eta ** -s) * max_budget
        initial_budgets.append(initial_budget)

        n0 = int(np.floor(s_max / (s + 1)) * eta ** s)
        n_configurations.append(n0)
        ns = [max(int(n0 * (eta ** (-i))), 1) for i in range(s + 1)]
        n_runs += sum(ns)

    print('Running BOBH with these parameters will proceed as follows:')
    print('  {} iterations of SuccessiveHalving will be executed.'.format(n_iterations))
    print('  The iterations will start with a number of configurations as {}.'.format(n_configurations))
    print('  With the initial budgets as {}.'.format(initial_budgets))
    print('  A total of {} unique configurations will be sampled.'.format(sum(n_configurations)))
    print('  A total of {} runs will be executed.'.format(n_runs))



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


def calculate_trainset_data_budget(old_budget, required_budget):
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