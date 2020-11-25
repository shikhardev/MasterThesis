from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Utility.CONFIG import *

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


def _get_plot_loss_call_params(metric):
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


def _merge_and_fill_trajectories(pandas_data_frames, default_value=None):
    # merge all tracjectories keeping all time steps
    df = pd.DataFrame().join(pandas_data_frames, how='outer')

    # forward fill to make it a propper step function
    df = df.fillna(method='ffill')

    if default_value is None:
        # backward fill to replace the NaNs for the early times by
        # the performance of a random configuration
        df = df.fillna(method='bfill')
    else:
        df = df.fillna(default_value)
    return df


def _plot_losses(incumbent_trajectories, title, regret=True, incumbent=None, show=True, time_data=True, yscale='log',
                 ylabel=None, xlim=None, ylim=None, plot_mean=True, ax=None, style='seaborn'):
    plt.style.use(style)
    legend_loc = 'best'
    figsize = (16, 9)
    linewidth = 3

    if time_data:
        xlabel = 'Wall clock time [s]'
        xscale = 'log'
        traj_x_name = 'time_stamps'
    else:
        xlabel = 'Iterations of dataset consumed'
        xscale = 'linear'
        traj_x_name = 'costs'

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    if regret:
        if ylabel is None: ylabel = 'regret'
        # find lowest performance in the data to update incumbent

        if incumbent is None:
            incumbent = np.inf
            for tr in incumbent_trajectories.values():
                incumbent = min(tr['losses'][:, -1].min(), incumbent)
        print('incumbent value: ', incumbent)
    if ylabel is None: ylabel = 'Error (1 - Validation Accuracy)'

    for m, tr in incumbent_trajectories.items():

        trajectory = np.copy(tr['losses'])
        if trajectory.shape[0] == 0: continue
        if regret: trajectory -= incumbent

        sem = np.sqrt(trajectory.var(axis=0, ddof=1) / tr['losses'].shape[0])
        if plot_mean:
            mean = trajectory.mean(axis=0)

        else:
            mean = np.median(trajectory, axis=0)
            sem *= 1.253

        ax.fill_between(tr[traj_x_name], mean - 1 * sem, mean + 1 * sem, alpha=0.3)
        ax.plot(tr[traj_x_name], mean, label=m, color=None, linewidth=linewidth, marker=None, markersize=10,
                markevery=(0.1, 0.1))

    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)

    label_fontsize = 12
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, labelpad=10, fontsize=label_fontsize)
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel, labelpad=15, fontsize=label_fontsize)
    ax.set_title(title, pad=15, fontsize=label_fontsize)
    ax.grid(which='both', alpha=0.3, linewidth=2)

    if not legend_loc is None:
        ax.legend(loc=legend_loc, framealpha=1)

    if show:
        plt.show()

    return fig, ax


def _parse_data_in_plot_format(dataset_result_object, time_data=True):
    all_trajectories = {}
    legend_hash = {
        'EpochWithIncreasingTrainset': 'Epoch with increasing trainset',
        'Multitune': 'Multitune',
        'Multitune_NoBayesian': 'Multitune without utilizing miniature readings',
        'TimeWithIncreasingTrainset': 'Time with increasing trainset',
        'TrainsetWithIncreasingEPC': 'Trainset with increasing epochs'
    }
    for method in dataset_result_object:
        dfs = []
        for run in dataset_result_object[method]:
            d = dataset_result_object[method][run]['run_overview']

            times = np.array(d['times_finished'])
            cummulative_cost = np.array(d['cummulative_cost'])
            cummulative_budget = np.array(d['cummulative_budget'])

            m_r = method + str(run)

            if time_data:
                df = pd.DataFrame({m_r: d['losses']}, index=times)
                x_name = 'time_stamps'
            else:
                df = pd.DataFrame({m_r: d['losses']}, index=cummulative_cost)
                x_name = 'costs'
            dfs.append(df)
        df = _merge_and_fill_trajectories(dfs, default_value=0.9)

        if df.empty:
            continue

        all_trajectories[legend_hash[method]] = {
            x_name: np.array(df.index),
            'losses': np.array(df.T)
        }
    return all_trajectories


def plot_experiment_graphs(dataset_result_object, res_dir, required_plots, save_pdf=False, save_image=False,
                           show=True, dataset_name=None, time_data=True):
    # This is temporary, need to figure out a better way to do this
    # TODO: Delete this later
    if time_data:
        res_dir += 'Time_Based/'
    else:
        res_dir += 'Trainset_Based/'

    if not isdir(res_dir):
        makedirs(res_dir)

    all_trajectories = _parse_data_in_plot_format(dataset_result_object, time_data=time_data)
    for _ in required_plots:
        regret, mean = _get_plot_loss_call_params(_)
        if dataset_name is None:
            plot_title = _
        else:
            plot_title = dataset_name

        fig, ax = _plot_losses(
            all_trajectories,
            title=plot_title,
            regret=regret,
            plot_mean=mean,
            show=show,
            time_data=time_data,
            yscale='linear'
        )
        # TODO: Create a naming mechanism
        if save_pdf:
            with PdfPages(res_dir + _ + '.pdf') as pdf:
                fig.set_size_inches(12, 8)
                pdf.savefig(fig, dpi=1200)

        if save_image:
            plt.savefig(res_dir + _ + 'loss.svg', format='svg', dpi=1200)
            # plt.savefig(res_dir + _ + 'loss.eps', format='eps')


def plot_all_results(result_object, res_dir, save_pdf=True, save_image=False, show=False, ):
    """
    Plots and saves a copy of all possible results in the result_object
    :param result_object: Instance of Utility.Result_Utils.ExperimentResult.
    :param res_dir: Directory where all results are to be stored. Nothing is added to the res_dir in this method.
    :param save_pdf: Save PDFs for each image
    :param save_image: Save PDFs for each image
    :param show: Save PDFs for each image
    :return: None
    """
    required_plots = ["loss_mean", "loss_median", "regret_mean", "regret_median"]
    for dataset in result_object:
        plot_experiment_graphs(result_object[dataset], res_dir, required_plots, save_pdf,
                               save_image, show, dataset, time_data=True)
        plot_experiment_graphs(result_object[dataset], res_dir, required_plots, save_pdf,
                               save_image, show, dataset, time_data=False)


def plot_dataset_results(result_object, dataset_name, res_dir, save_pdf=True, save_image=False, show=False, ):
    """
    Plots and saves a copy of all possible results in the result_object
    :param result_object: Instance of Utility.Result_Utils.ExperimentResult.results[dataset]
    :param res_dir: Directory where all results are to be stored. Nothing is added to the res_dir in this method.
    :param save_pdf: Save PDFs for each image
    :param save_image: Save PDFs for each image
    :param show: Save PDFs for each image
    :return: None
    """
    # required_plots = ["loss_mean", "loss_median", "regret_mean", "regret_median"]
    required_plots = ["loss_mean"]
    plot_experiment_graphs(result_object[dataset_name], res_dir, required_plots, save_pdf,
                           save_image, show, dataset_name=dataset_name, time_data=True)
    plot_experiment_graphs(result_object[dataset_name], res_dir, required_plots, save_pdf,
                           save_image, show, dataset_name=dataset_name, time_data=False)
