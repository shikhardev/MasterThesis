import os
from Analysis_Module.analysis import experiment_overview_analysis, verbose_pkl_to_csv, plot_experiment_graphs
# from Utility.Plot_Utils import plot_experiment_graphs


def get_files():
    d = []
    # home_loc = '/Users/shikhar/Desktop/Final Experiment Data/trial_1_trainset_needs_fixing/'
    home_loc = '/Users/shikhar/Desktop/Final Experiment Data/Results/'
    d.append(home_loc + 'Cifar 10/')
    d.append(home_loc + 'Fashion/')
    d.append(home_loc + 'MNIST/')
    d.append(home_loc + 'SST/')
    d.append(home_loc + 'Twitter/')
    return d

def get_methods():
    m = [
        'EpochWithIncreasingTrainset',
        'Multitune',
        # 'Multitune_No_Bayesian',
        'TimeWithIncreasingTrainset',
        'TrainsetWithIncreasingEPC'
    ]
    return m


if __name__ == '__main__':
    required_plots = ["loss_mean"]
    d = get_files()
    methods = get_methods()
    # experiment = '2d Benchmark'
    experiment = '3d Benchmark'
    # experiment = 'Bayesian'
    res_dir = '/Users/shikhar/Desktop/Final Experiment Data/Experiment Results/' + experiment + '/'


    for _ in d:
        dataset_name = _.split('/')[-2]
        # res_dir = _ + "analysis_results/"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        experiment_overview_analysis(home_dir=_, res_dir=res_dir, methods=methods,file_name=dataset_name + '.csv')
        plot_experiment_graphs(inp_dir=_, res_dir=res_dir, optimizer_methods=methods, required_plots=required_plots,
                               save_pdf=True,
                               save_image=False, show=False, dataset_name=dataset_name)
        verbose_pkl_to_csv(inp_dir=_, methods=methods, res_dir=res_dir)
