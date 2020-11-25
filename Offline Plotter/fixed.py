import os
from Analysis_Module.analysis import plot_experiment_graphs, experiment_overview_analysis, verbose_pkl_to_csv


def get_files():
    d = []
    d.append('/Users/shikhar/Desktop/Final Experiment Data/Fixed/Cifar 10/')
    return d

def get_methods():
    m = [
        'Infinite_Epoch_With_Increasing_Trainset',
        'Exploratory Budget',
        'Infinite_Time_With_Increasing_Trainset',
        # 'Attempt-4',
        # 'Infinite_Mix_Epoch_Trainset_Time',
        # 'Infinite_Time',
        'Infinite_Trainset_With_Increasing_EPC',
        # 'Starting-at-25',
        # 'Starting-at-6-25'
    ]
    return m


if __name__ == '__main__':
    required_plots = ["loss_mean"]
    d = ['/Users/shikhar/Desktop/experiment_data/_Fixed/Cifar10/']
    methods = ['Trainset With EPC 4', 'Trainset With EPC 8', 'Trainset With EPC 16', 'Trainset With EPC 32',
               'Trainset With EPC 64', 'Trainset With EPC 128']

    for _ in d:
        dataset_name = _.split('/')[-2]
        res_dir = _ + "analysis_results/"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        experiment_overview_analysis(home_dir=_, res_dir=res_dir, methods=methods)
        plot_experiment_graphs(inp_dir=_, res_dir=res_dir, optimizer_methods=methods, required_plots=required_plots,
                               save_pdf=True,
                               save_image=True, show=False, dataset_name=dataset_name)
        verbose_pkl_to_csv(inp_dir=_, methods=methods, res_dir=res_dir)
