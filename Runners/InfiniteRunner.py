import os
import random
import numpy as np
import torch
from Utility.CONFIG import SEED
from Workers.Cifar import CifarWorker
from Workers.CNN_MNIST import CNNMNISTWorker
from Workers.Fashion import FashionWorker
from Workers.TW_Worker import TwitterWorker
from Workers.SST import SSTWorker
from Utility.Worker_Utils import run_optimizer_process
from Utility.Result_Utils import ExperimentResult
from Utility.Plot_Utils import plot_dataset_results as plot


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)


def main(job_dir, **args):

    results = ExperimentResult()
    experiment_count = 3
    d_w = {
        'CIFAR10': CifarWorker,
        'FASHION': FashionWorker,
        'MNIST_CNN': CNNMNISTWorker,
        'Twitter': TwitterWorker,
        'SST': SSTWorker
    }

    methods = [
        'EpochWithIncreasingTrainset',
        'TimeWithIncreasingTrainset',
        'TrainsetWithIncreasingEPC',
        'Multitune',
        # 'Temp_Multitune'
        'Multitune_NoBayesian'
    ]

    verbose = True
    for _ in range(experiment_count):
        for d in d_w:
            res_dir = job_dir + '/Results/' + d + '/'
            for m in methods:
                set_seed()  # Start from the same pseudo-randomness for each experiment, method and dataset
                work_dir = job_dir + '/Work/' + d + '/' + m + '/Run_' + str(_) + '/'
                Worker = d_w[d]
                if not os.path.isdir(work_dir):
                    os.makedirs(work_dir)
                if not os.path.isdir(res_dir):
                    os.makedirs(res_dir)
                res = run_optimizer_process(run_id=str(_),
                                            worker=Worker,
                                            method=m,
                                            res_dir=res_dir,
                                            work_dir=work_dir,
                                            verbose=verbose)
                results.add_new_result(d, m, res, res_dir=res_dir, run_id=_)

                print('-' * 10)
                print('Dataset: ', d)
                print("Experiment Number: ", str(_))
                print("Method: ", m)
                print('-' * 10)

            # plot_dir = res_dir + 'Plots/'
            # plot(result_object=results.results, res_dir=plot_dir, dataset_name=d)


print("Done importing and defining")

main('Results/')
