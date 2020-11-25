# import IPython
# IPython.embed()
import csv
import glob
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from Analysis_Module.plot_utils import merge_and_fill_trajectories, plot_losses


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# Run analysis functions


def get_best_details_for_run(pkl_file):
    with open(pkl_file, 'rb') as fh:
        d = pickle.load(fh)
    res = {}
    losses = d['losses']
    min = losses[-1]
    for i, _ in enumerate(losses[-1::-1]):
        if _ > min:
            break
    # best_row = -i + 1
    best_row = -1
    res["Best_Accuracy"] = 1 - d['losses'][best_row]
    cid = d['config_ids'][best_row]
    res['Best_Config_ID'] = cid
    res['Best_Config'] = d['configs'][cid]['config']
    res['Acc_received_at'] = d['times_finished'][best_row] / 60
    res['Exp_Duration'] = d['times_finished'][-1] / 60
    return res


def experiment_overview_analysis(home_dir, res_dir, methods, file_name=None):
    if file_name is None:
        file_name = 'analysis.csv'
    csv_fh = open(res_dir + file_name, 'w')
    writer = csv.DictWriter(csv_fh, ['Method', 'Experiment', 'Acc_received_at', 'Best_Accuracy', 'Best_Config_ID',
                                     'Best_Config', 'Exp_Duration'])
    writer.writeheader()

    for m in methods:
        for fn in glob.glob(home_dir + '/' + m + '/*.pkl'):
            d = get_best_details_for_run(fn)
            d["Experiment"] = fn.split("/")[-1].split(".")[0]
            d['Method'] = m
            writer.writerow(d)


# End of analysis functions
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# Plot functions

# def parse_data_in_plot_format(inp_dir, res_dir, optimizer_methods):
#     all_trajectories = {}
#     for m in optimizer_methods:
#         dfs = []
#         for fn in glob.glob(inp_dir + '/' + m + '/*.pkl'):
#             print("------File Name: ", fn)
#             with open(fn, 'rb') as fh:
#                 d = pickle.load(fh)
#             times = np.array(d['times_finished'])
#             df = pd.DataFrame({fn: d['losses']}, index=times)
#             dfs.append(df)
#         df = merge_and_fill_trajectories(dfs, default_value=0.9)
#
#         if df.empty:
#             continue
#
#         all_trajectories[m] = {
#             'time_stamps': np.array(df.index),
#             'losses': np.array(df.T)
#         }
#     return all_trajectories


def parse_data_in_plot_format(inp_dir, optimizer_methods, time_plot=True):
    all_trajectories = {}
    legend_hash = {
        'EpochWithIncreasingTrainset': 'Epochs with increasing trainset',
        'Multitune': 'Multitune',
        'Multitune_No_Bayesian': 'Multitune without utilizing miniature readings',
        'TimeWithIncreasingTrainset': 'Time with increasing trainset',
        'TrainsetWithIncreasingEPC': 'Trainset with increasing epochs'
    }
    for m in optimizer_methods:
        dfs = []
        for fn in glob.glob(inp_dir + '/' + m + '/*.pkl'):
            with open(fn, 'rb') as fh:
                d = pickle.load(fh)
            times = np.array(d['times_finished'])
            costs = np.array(d['cummulative_cost'])
            cummulative_budget = np.array(d['cummulative_budget'])

            if time_plot:
                df = pd.DataFrame({fn: d['losses']}, index=times)
                x_name = 'time_stamps'
            else:
                df = pd.DataFrame({fn: d['losses']}, index=cummulative_budget)
                x_name = 'costs'
            dfs.append(df)
        df = merge_and_fill_trajectories(dfs, default_value=0.9)

        if df.empty:
            continue

        all_trajectories[legend_hash[m]] = {
            x_name: np.array(df.index),
            'losses': np.array(df.T)
        }
    return all_trajectories

def get_plot_loss_call_params(metric):
    regret = False
    mean = False
    if metric == "loss_mean":
        regret = False
        mean = True
    elif metric == "loss_median":
        regret = False
        mean = False
    elif metric == "regret_mean":
        regret = True
        mean = True
    elif metric == "regret_median":
        regret = True
        mean = False
    return regret, mean


def plot_experiment_graphs(inp_dir, res_dir, optimizer_methods, required_plots, save_pdf=False, save_image=False,
                           show=True, dataset_name=None, time_plot=True):
    all_trajectories = parse_data_in_plot_format(inp_dir=inp_dir, time_plot=time_plot, optimizer_methods=optimizer_methods)
    for _ in required_plots:
        regret, mean = get_plot_loss_call_params(_)
        if dataset_name is None:
            plot_title = _
        else:
            plot_title = dataset_name

        # plot_title = None

        fig, ax = plot_losses(
            all_trajectories,
            title=plot_title,
            regret=regret,
            plot_mean=mean,
            show=show,
            time_plot=time_plot,
            yscale='linear'
        )

        file_name = plot_title
        if save_pdf:
            with PdfPages(res_dir + file_name + '_' + _ + '.pdf') as pdf:
                # fig.set_size_inches(7.5, 5)
                fig.set_size_inches(12, 8)
                pdf.savefig(fig, dpi=1200)

        if save_image:
            plt.savefig(res_dir + file_name + '_' + _ + 'loss.svg', format='svg', dpi=1200)
            # plt.savefig(res_dir + _ + 'loss.eps', format='eps')


# End of plot functions
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# Verbose code

def verbose_pkl_to_csv(inp_dir, methods, res_dir):
    res_dir = res_dir + "verbose/"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    for m in methods:
        for fn in glob.glob(inp_dir + '/' + m + '/verbose' + '/*.pkl'):
            df = pd.read_pickle(fn)
            run_name = fn.split('/')[-1].split('_v')[0]
            file_name = res_dir + '/' + m + '_' + run_name + ".csv"
            df.to_csv(file_name)


def plot_config_distribution(inp_dir, methods, res_dir, title = None):
    res_dir = res_dir + "verbose/"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    methods = ['randomsearch']
    bs_df = None
    for m in methods:
        for fn in glob.glob(inp_dir + '/' + m + '/verbose' + '/*.pkl'):
            df = pd.read_pickle(fn)
            add_loss_to_df = lambda x, y: x.update({'loss': y}) or x
            ds = pd.DataFrame ([add_loss_to_df (row[3], row[-1]) for row in df.itertuples()])
            if bs_df is None:
                bs_df = ds
            else:
                bs_df.append(ds)

    display_hp_list = ['batch_size']
    # for hp in list (ds.columns):
    for hp in display_hp_list:
        ax = bs_df.pivot(columns=hp).loss.plot(kind='hist', stacked=True, linewidth=2)
        ax.set_xlabel ("Validation Loss: Lower the better")
        ax.set_ylabel("Frequency of occurance in experiments")
        if title is not None:
            ax.set_title(title + ' : ' + hp)
        plt.show()
        print(hp)


# End of Verbose code
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


def get_list_of_files():
    d = []
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/Cifar10/Epochs/')
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/Cifar10/Trainset/')

    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/MNIST/Epochs/')
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/MNIST/Epochs2/')
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/MNIST/Trainset/')

    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/Fashion/Epochs/')
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/Fashion/Trainset/')
    return d


def old_list_of__files():
    d = []
    # directory = "/Users/shikhar/Desktop/thesis/Code/clean_bohb/analysis_modules/run_logs/trainset/"
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Cifar10/Iter/')
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/Cifar10/Trainset/')
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/MNIST/Iter/')
    d.append('/Users/shikhar/Desktop/thesis/Analysis Results/MNIST/Trainset/')
    return d

if __name__ == '__main__':
    d = get_list_of_files()
    methods = ['bohb', 'hyperband', 'randomsearch']
    required_plots = ["loss_mean", "loss_median", "regret_mean", "regret_median"]
    # plot_config_distribution(inp_dir=d[0], methods=methods, res_dir='.', title="Cifar10 iter")
    for _ in d:
        # res_dir = _ + "analysis_results/"
        res_dir = '/Users/shikhar/Desktop/temp_results/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        experiment_overview_analysis(home_dir=_, res_dir=res_dir, methods=methods)
        plot_experiment_graphs(inp_dir=_, res_dir=res_dir, optimizer_methods=methods, required_plots=required_plots,
                               time_data=True,
                               save_pdf=True, save_image=False, show=False)
        plot_experiment_graphs(inp_dir=_, res_dir=res_dir, optimizer_methods=methods, required_plots=required_plots,
                               time_data=False,
                               save_pdf=True, save_image=False, show=False)
        verbose_pkl_to_csv(inp_dir=_, methods=methods, res_dir=res_dir)
        plot_config_distribution(inp_dir=_, methods=methods, res_dir=res_dir)
