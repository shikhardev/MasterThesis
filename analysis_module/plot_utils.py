import glob
import os
import pickle
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os



def genLogSpace(array_size, num):
    lspace = np.around(np.logspace(0, np.log10(array_size), num)).astype(np.uint64)
    return np.array(sorted(set(lspace.tolist()))) - 1


colors = {
    'hyperband3': 'darkorange',
    'BO-HB3': 'green',
    'randomsearch': 'black',
    'tpe': 'blue',
    'my_tpe': 'blue',
    'robo': 'red',
    'lcnet': 'indigo'
}

markers = {
    'hyperband3': '^',
    'BO-HB3': 'v',
    'randomsearch': 'x',
    'tpe': 'D',
    'my_tpe': 'o',
    'robo': 's',
    'lcnet': 'h'
}

labels = {
    'hyperband3': 'HB',
    'BO-HB3': 'BOHB',
    'randomsearch': 'RS',
    'tpe': 'TPE',
    'my_tpe': 'MY_TPE',
    'robo': 'GP-MCMC',
    'lcnet': 'LC Net + HB'
}


def merge_and_fill_trajectories(pandas_data_frames, default_value=None):
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

    return (df)


def plot_losses(incumbent_trajectories, title, regret=True, incumbent=None, show=True, linewidth=3, marker_size=10,
                time_plot=True, yscale='log', ylabel=None, xlim=None, ylim=None, plot_mean=True,
                legend_loc='best',labels={}, markers={}, colors={}, figsize=(16, 9), ax=None, style='seaborn'):
    colors = {
        'Epochs with increasing trainset': 'indianred',
        'Multitune': 'dimgrey',
        'Multitune without utilizing miniature readings': 'darkcyan',
        'Time with increasing trainset': 'mediumseagreen',
        'Trainset with increasing epochs': 'steelblue'
    }

    plt.style.use(style)
    if time_plot:
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
        if (trajectory.shape[0] == 0): continue
        if regret: trajectory -= incumbent

        sem = np.sqrt(trajectory.var(axis=0, ddof=1) / tr['losses'].shape[0])
        if plot_mean:
            mean = trajectory.mean(axis=0)

        else:
            mean = np.median(trajectory, axis=0)
            sem *= 1.253

        ax.fill_between(tr[traj_x_name], mean - 1 * sem, mean + 1 * sem, alpha=0.3, color=colors.get(m, None))
        ax.plot(tr[traj_x_name], mean, label=m, color=colors.get(m, None), linewidth=linewidth, marker=None, markersize=10,
                markevery=(0.1, 0.1))

    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)

    label_fontsize = 18
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, labelpad=10, fontsize=label_fontsize)
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel, labelpad=15, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # ax.set_title(title, pad=15, fontsize=label_fontsize)
    ax.grid(which='both', alpha=0.3, linewidth=2)

    if not legend_loc is None:
        ax.legend(loc=legend_loc, framealpha=1, prop={'size': label_fontsize})

    if show:
        plt.show()

    return fig, ax



# plot_losses(all_trajectories[d], '%s'%d, regret=False, show=False, plot_mean=True)
# def plot_losses(incumbent_trajectories, title, regret=True, incumbent=None, show=True, linewidth=3, marker_size=10,
#                 xscale='log', xlabel='wall clock time [s]',
#                 yscale='log', ylabel=None,
#                 legend_loc='best',
#                 xlim=None, ylim=None,
#                 plot_mean=True, labels={}, markers={}, colors={}, figsize=(16, 9), ax=None):
#     fig = None
#     if ax is None:
#         fig, ax = plt.subplots(1, figsize=figsize)
#
#     if regret:
#         if ylabel is None: ylabel = 'regret'
#         # find lowest performance in the data to update incumbent
#
#         if incumbent is None:
#             incumbent = np.inf
#             for tr in incumbent_trajectories.values():
#                 incumbent = min(tr['losses'][:, -1].min(), incumbent)
#         print('incumbent value: ', incumbent)
#     if ylabel is None: ylabel = 'loss'
#
#     for m, tr in incumbent_trajectories.items():
#
#         trajectory = np.copy(tr['losses'])
#         if (trajectory.shape[0] == 0): continue
#         if regret: trajectory -= incumbent
#
#         sem = np.sqrt(trajectory.var(axis=0, ddof=1) / tr['losses'].shape[0])
#         if plot_mean:
#             mean = trajectory.mean(axis=0)
#
#         else:
#             mean = np.median(trajectory, axis=0)
#             sem *= 1.253
#
#         ax.fill_between(tr['time_stamps'], mean - 2 * sem, mean + 2 * sem, color=colors.get(m, 'black'), alpha=0.3)
#         ax.plot(tr['time_stamps'], mean,
#                 label=labels.get(m, m), color=colors.get(m, None), linewidth=linewidth,
#                 marker=markers.get(m, None), markersize=marker_size, markevery=(0.1, 0.1))
#
#     if not xlim is None:
#         ax.set_xlim(xlim)
#     if not ylim is None:
#         ax.set_ylim(ylim)
#
#     ax.set_xscale(xscale)
#     ax.set_xlabel(xlabel)
#     ax.set_yscale(yscale)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.grid(which='both', alpha=0.3, linewidth=2)
#
#     if not legend_loc is None:
#         ax.legend(loc=legend_loc, framealpha=1)
#
#     if show:
#         plt.show()
#
#     return (fig, ax)


def get_incumbent_trajectories(directory, file_expr='random_search_run_*.pkl', incumbent_function=None,
                               all_budgets=True):
    """ reads all files matching file_expr in directory to extract the incumbent trajectories

        Parameters:
        -----------
            directory: str
                root directory for the pickled run data
            file_expr: str
                regular expression (shell like expansions possible) for the file names
            incumbent_function: callable
                function to extract the incumbent, given a HB_result object.
                If None (default), the build in 'get_incumbent_trajectory' is used.
            all_budgets: bool
                see 'get_incumbent_trajectory'

    """
    incumbent_traj = []

    for i, fn in enumerate(glob.glob(os.path.join(directory, file_expr))):
        # print(fn)
        with open(fn, 'rb') as fh:
            res = pickle.load(fh)

        if incumbent_function is None:
            incumbent_traj.append(res.get_incumbent_trajectory(all_budgets))
        else:
            incumbent_traj.append(incumbent_function(res))

    return (incumbent_traj)


def save_pgf_data(data, directory='./pgf_data/', regret=False, reduce_points=None, clip=(-np.inf, np.inf)):
    for d in data.keys():
        subdir = directory + d

        os.makedirs(subdir, exist_ok=True)

        if regret:
            inc_val = np.inf

            for m in data[d].keys():
                inc_val = min(inc_val, np.min(data[d][m]['losses']))

        else:
            inc_val = 0

        print(inc_val)
        for m in data[d].keys():

            times = data[d][m]['time_stamps']

            if len(times) == 0:
                continue

            losses = data[d][m]['losses'] - inc_val

            losses = np.clip(losses, clip[0], clip[1])

            mean = losses.mean(axis=0)
            se_mean = np.sqrt(losses.var(axis=0, ddof=1) / losses.shape[0])

            median = np.median(losses, axis=0)
            se_median = 1.253 * np.sqrt(losses.var(axis=0, ddof=1) / losses.shape[0])

            p25 = np.percentile(losses, 25, axis=0)
            p75 = np.percentile(losses, 75, axis=0)

            # import IPython; IPython.embed()

            d2store = np.stack([times, mean, se_mean, median, se_median, p25, p75]).T

            with open(subdir + '/' + m + '.csv', 'w') as csvfile:
                fieldnames = ['time', 'mean', 'se_mean', 'median', 'se_median', 'p25', 'p75']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()

                if (not reduce_points is None) and (reduce_points < len(times)):
                    tmin, tmax = np.min(times), np.max(times)
                    ts = np.logspace(np.log10(tmin), np.log10(tmax), 512)

                    mean = np.interp(ts, times, mean)
                    se_mean = np.interp(ts, times, se_mean)
                    median = np.interp(ts, times, median)
                    se_median = np.interp(ts, times, se_median)
                    p25 = np.interp(ts, times, p25)
                    p75 = np.interp(ts, times, p75)
                    times = ts

                print('->', d, m, times.shape)

                for i in range(len(times)):
                    writer.writerow({'time': times[i],
                                     'mean': mean[i],
                                     'se_mean': se_mean[i],
                                     'median': median[i],
                                     'se_median': se_median[i],
                                     'p25': p25[i],
                                     'p75': p75[i]
                                     })


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




# def plot_experiment_graphs(directory, optimizer_methods, required_plots, save_pdf=False, title=None, save_image=False, show=True):
#     plot_count = len(required_plots)
#     if plot_count == 0:
#         return
#
#     all_trajectories = {}
#     for m in optimizer_methods:
#         dfs = []
#         for fn in glob.glob(directory + m + '*.pkl'):
#             # print("------File Name: ", fn)
#             with open(fn, 'rb') as fh:
#                 datum = pickle.load(fh)
#
#             times = np.array(datum['cummulative_cost'])
#             df = pd.DataFrame({fn: datum['losses']}, index=times)
#             dfs.append(df)
#
#         df = merge_and_fill_trajectories(dfs, default_value=0.9)
#
#         if df.empty:
#             continue
#
#         all_trajectories[m] = {
#             'time_stamps': np.array(df.index),
#             'losses': np.array(df.T)
#         }
#
#     # Starting plots
#     # Create axes first
#     if plot_count < 4:
#         row_count = plot_count
#         col_count = 1
#     else:
#         row_count = 2
#         col_count = 2
#
#     plot_index = 0
#     fig, ax = plt.subplots(row_count, col_count)
#     fig.tight_layout()
#     last = None
#     for row in ax:
#         for col in row:
#             regret, mean = get_plot_loss_call_params(required_plots[plot_index])
#             plot_losses(all_trajectories,
#                         title=required_plots[plot_index],
#                         regret=regret,
#                         show=False,
#                         plot_mean=mean,
#                         yscale='linear',
#                         ax=col)
#             plot_index += 1
#             last = col
#
#     handles, labels = last.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center')
#
#     if title != None:
#         fig.suptitle(title, fontsize=16)
#
#     if save_pdf:
#         pdf_location = directory + "analysis_results/"
#         if not os.path.exists(pdf_location):
#             os.makedirs(pdf_location)
#         with PdfPages(pdf_location + 'loss.pdf') as pdf:
#             fig.set_size_inches(12,8)
#             pdf.savefig(fig, dpi = 1200)
#
#     if save_image:
#         img_loc = directory + "analysis_results/"
#         if not os.path.exists(img_loc):
#             os.makedirs(img_loc)
#         plt.savefig(img_loc + 'loss.svg', format='svg', dpi=1200)
#         plt.savefig(img_loc + 'loss.eps', format='eps')
#
#     if show:
#         plt.show()

