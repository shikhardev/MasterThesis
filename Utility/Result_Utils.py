import pandas as pd
from Utility.CONFIG import *
import pickle
# from tensorflow.python.lib.io import file_io

# if GCLOUD:
#     from tensorflow_core.python.lib.io.file_io import delete_file_v2 as remove
#     from tensorflow_core.python.lib.io.file_io import get_matching_files_v2 as glob
#     from tensorflow.io.gfile import makedirs
# else:
#     from glob import glob
#     from os import makedirs, remove
#     from os.path import isdir
from glob import glob
from os import makedirs, remove
from os.path import isdir



def result_object_to_overview(results_object):
    """
    Returns the best configurations over time, but also returns the cummulative budget
    Previously known as extract_results_to_pickle
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
        cummulative_cost += r.info['trainset_consumed']

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

    if current_incumbent != r.loss:
        r = all_runs[-1]
        return_dict['config_ids'].append(return_dict['config_ids'][-1])
        return_dict['times_finished'].append(r.time_stamps['finished'])
        return_dict['budgets'].append(return_dict['budgets'][-1])
        return_dict['losses'].append(return_dict['losses'][-1])
        return_dict['cummulative_budget'].append(cummulative_budget)
        return_dict['cummulative_cost'].append(cummulative_cost)

    return_dict['configs'] = {}

    id2conf = results_object.get_id2config_mapping()

    for c in return_dict['config_ids']:
        return_dict['configs'][c] = id2conf[c]

    return_dict['HB_config'] = results_object.HB_config

    return (return_dict)


def result_to_verbose_dataframe(res):
    all_runs = res.get_all_runs(only_largest_budget=False)
    all_runs.sort(key=lambda r: r.time_stamps['finished'])
    id2conf = res.get_id2config_mapping()
    res = []
    for run in all_runs:
        row = {'cid': run.config_id}
        cid = run.config_id
        row['config'] = id2conf[cid]['config']
        row['model_based'] = id2conf[cid]['config_info']['model_based_pick']
        row['budget'] = run.budget
        row['tstart'] = run.time_stamps['started']
        row['tfinish'] = run.time_stamps['finished']
        row['validation_error'] = run.loss
        row['epochs_per_config'] = run.info['epc'] if 'epc' in run.info else None
        row['epoch_multiplier'] = run.info['epoch_multiplier'] if 'epoch_multiplier' in run.info else None
        row['test_accuracy'] = run.info['test_accuracy'] if 'test_accuracy' in run.info else None
        row['validation_accuracy'] = run.info['validation_accuracy'] if 'validation_accuracy' in run.info else None
        row['val_confidence'] = run.info['validation_confidence'] if 'validation_confidence' in run.info else None
        row['test_confidence'] = run.info['test_confidence'] if 'test_confidence' in run.info else None
        row['trainset_consumed'] = run.info['trainset_consumed'] if 'trainset_consumed' in run.info else None
        row['trainset_budget'] = run.info['trainset_budget'] if 'trainset_budget' in run.info else None
        row['time_multiplier'] = run.info['time_multiplier'] if 'time_multiplier' in run.info else None
        row['epochs_for_time_budget'] = run.info['epochs_for_time_budget'] if 'epochs_for_time_budget' in run.info else None
        res.append(row)
    df = pd.DataFrame(res)
    return df


def store_run_details(result_object, m_r, res_dir=""):
    """
    :param d_m_r: dataset _ method _ run_id
    """
    if not isdir(res_dir):
        makedirs(res_dir)

    # TODO: Check for gcloud vs local run
    # with file_io.FileIO(res_dir + m_r + '.pkl', mode='w+') as f:
    #     pickle.dump(result_object, f)
    with open (res_dir + m_r + '.pkl', 'wb') as f:
        pickle.dump(result_object, f)


def _get_best_details_for_run(overview_result_object):
    d = overview_result_object
    res = {}
    losses = d['losses']
    min = losses[-1]
    best_row = -1
    res["Best_Accuracy"] = 1 - d['losses'][best_row]
    cid = d['config_ids'][best_row]
    res['Best_Config_ID'] = cid
    res['Best_Config'] = d['configs'][cid]['config']
    res['Acc_received_at'] = d['times_finished'][best_row] / 60
    res['Exp_Duration'] = d['times_finished'][-1] / 60
    return res


def dataset_experiments_overview(dataset_result_object, d_m_r, methods, store=True, res_dir=""):
    df = []
    for m in methods:
        all_runs = dataset_result_object[m]
        for run in all_runs:
            d = _get_best_details_for_run(all_runs[run])
            d["Experiment"] = d_m_r
            d['Method'] = m
            df.append(d)

    df = pd.DataFrame(df)
    if store:
        res_dir += '/Results/Summary_Folder/'
        df.to_csv(res_dir + 'Summary.csv')
    return df


class ExperimentResult:
    """
    Object contains details of all the runs.
    Object: self.results
        {
            dataset_name: {
                method_name: {
                    run_id: {
                        'raw_results': 'result_object',
                        'run_overview': 'overview_dict_for_one_run',
                        'verbose_df': 'verbose_dataframe_of_run'
                    }
                }
            }
        }
    """
    def __init__(self):
        self.results = {}

    def add_new_result(self, dataset_name, method_name, result_object, store_intermediaries=True, res_dir=None, run_id=0):
        if dataset_name not in self.results:
            self.results[dataset_name] = {}

        if method_name not in self.results[dataset_name]:
            self.results[dataset_name][method_name] = {}
        #     run_id = 0
        # else:
        #     run_id = max(self.results[dataset_name][method_name].keys())
        #     run_id += 1

        self.results[dataset_name][method_name][run_id] = {}
        # We'll create a temp dict and store it here at the end
        temp = {}
        temp['raw_result'] = result_object
        temp['run_overview'] = result_object_to_overview(result_object)
        temp['verbose_df'] = result_to_verbose_dataframe(result_object)
        self.results[dataset_name][method_name][run_id] = temp

        m_r = method_name + '_' + str(run_id)
        overview_dir = res_dir + 'Experiments_Overview/'
        verbose_dir = res_dir + 'Experiments_Verbose/'
        if store_intermediaries:
            store_run_details(result_object=temp['run_overview'], m_r=m_r, res_dir=overview_dir)
            store_run_details(result_object=temp['verbose_df'], m_r=m_r, res_dir=verbose_dir)

        if STORE_VERBOSE_CSV:
            temp['verbose_df'].to_csv(res_dir + m_r + '.csv')


