import pickle
import pandas as pd
from tensorflow.python.lib.io import file_io
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import RandomSearch, BOHB, HyperBand


def standard_parser_args(parser):
    parser.add_argument('--dest_dir',
                        type=str,
                        help='the destination directory to store all results',
                        default='../results/')

    parser.add_argument('--num_iterations',
                        type=int,
                        help='number of Hyperband iterations performed.',
                        default=4)
    parser.add_argument('--run_id',
                        type=str,
                        default=0)
    parser.add_argument('--working_directory',
                        type=str,
                        help='Directory holding live rundata. Should be shared across all nodes for parallel '
                             'optimization.',
                        default='/tmp/')
    parser.add_argument('--method',
                        type=str,
                        default='randomsearch',
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

    with file_io.FileIO(location + fileName, mode='w+') as f:
        pickle.dump(pd.DataFrame(res), f)


def store_experiment_details(methods, number_of_runs, budget_type, min_budget, max_budget,
                             eta, num_iterations, file_location):
    """
    Stores the result of the current batch of experiments. To be called from the b
    """
    res = {
        "methods": methods,
        "number_of_runs": number_of_runs,
        "budget_type": budget_type,
        "min_budget": min_budget,
        "max_budget": max_budget,
        "eta": eta,
        "num_iterations": num_iterations,
        "file_location": file_location
    }
    # with open(file_location + "experiment_details.json", "w") as outfile:
    #     json.dump(res, outfile, indent=4)

    with file_io.FileIO(file_location + "experiment_details.pkl", mode='w+') as f:
        pickle.dump(res, f)


def run_optimizer_process(method, run_id, min_budget, max_budget, worker, dest_dir, verbose,
                          eta, num_iterations=5, working_directory='.'):
    """
    Optimizes the worker using specified method
    :param method: Optimization method {'randomsearch', 'bohb', 'hyperband'}
    :param run_id: Identifier for the current run of the method
    :param min_budget: Minimum budget
    :param max_budget: Max budget
    :param worker: Worker for the current experiment {Worker object}
    :param dest_dir: Directory where all results are to be stored
    :param verbose: Do you want to store all HP config details of the current run? {True, False}
    :param eta: SH ratio [Refer to thesis document]
    :param num_iterations: Number of Successive Halving iterations
    :param working_directory: Location for temporary files required during the run
    :return: None
    """
    # Start a nameserver:
    NS = hpns.NameServer(run_id=run_id, host='127.0.0.1', port=0, working_directory=working_directory)
    ns_host, ns_port = NS.start()

    # w = worker(run_id=run_id, timeout=120)
    # w.load_nameserver_credentials(working_directory=working_directory)
    # w.run(background=False)
    # print ("HPNS started")
    # exit(0)

    w = worker(run_id=run_id, host='127.0.0.1', nameserver=ns_host, nameserver_port=ns_port, timeout=120)
    w.run(background=True)

    optimizer = get_optimizer(eta, w.get_configspace(), method=method, working_directory=working_directory,
                              run_id=run_id,
                              min_budget=min_budget, max_budget=max_budget,
                              host=ns_host,
                              nameserver=ns_host,
                              nameserver_port=ns_port,
                              ping_interval=3600,
                              result_logger=None,
                              )

    res = optimizer.run(n_iterations=num_iterations)

    with file_io.FileIO(dest_dir + '{}_run_{}.pkl'.format(method, run_id), mode='w+') as f:
        pickle.dump(extract_results_to_pickle(res), f)

    if verbose:
        store_verbose_results(res, dest_dir, '{}_run_{}_verbose.pkl'.format(method, run_id))


    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()

    # with open(os.path.join(dest_dir, '{}_run_{}.pkl'.format(method, run_id)), 'wb') as fh:
    #     pickle.dump(extract_results_to_pickle(res), fh)


