from Workers.Cifar import CifarWorker
from Workers.CNN_MNIST import CNNMNISTWorker
from Workers.Fashion import FashionWorker
from Workers.TW_Worker import TwitterWorker
from Workers.SST import SSTWorker

from Utility.Worker_Utils import run_optimizer_process
import os
from Utility.Result_Utils import ExperimentResult
from Utility.Plot_Utils import plot_dataset_results as plot


def main(job_dir, **args):
    results = ExperimentResult()
    experiment_count = 1
    d_w = {
        'CIFAR10': CifarWorker,
        'FASHION': FashionWorker,
        'MNIST_CNN': CNNMNISTWorker,
        # 'Twitter': TwitterWorker,
        # 'SST': SSTWorker
    }

    methods = [
        'trainset_bohb'
    ]
    bounds = [
        {
            'min_budget': 0.25,
            'max_budget': 4,
            'eta': 2,
        }
    ]

    verbose = True
    for _ in range(experiment_count):
        for d in d_w:
            res_dir = job_dir + '/Results/' + d + '/'
            for m in methods:
                for b in bounds:
                    bound_params = str(b['min_budget']) + '_' + str(b['max_budget']) + '_' + str(b['eta'])
                    cur_res_dir = res_dir + bound_params + '/'
                    work_dir = job_dir + '/Work/' + d + '/' + m + '_' + bound_params + '/Run_' + str(_) + '/'
                    Worker = d_w[d]
                    if not os.path.isdir(work_dir):
                        os.makedirs(work_dir)
                    if not os.path.isdir(res_dir):
                        os.makedirs(res_dir)
                    # Worker.fixed_execution_type = 'trainset'
                    # Worker.fixed_epc = 2

                    res = run_optimizer_process(
                        eta=b['eta'], min_budget=b['min_budget'], max_budget=b['max_budget'],
                        run_id=str(_),
                        worker=Worker,
                        method=m,
                        res_dir=cur_res_dir,
                        work_dir=work_dir,
                        verbose=verbose,
                        num_iterations=5,
                        fixed_budget='time',
                        epc=None
                    )
                    results.add_new_result(d, m, res, run_id=_, res_dir=cur_res_dir)

                    print('-' * 10)
                    print('Dataset: ', d)
                    print("Experiment Number: ", str(_))
                    print("Method: ", m)
                    print('-' * 10)

main('Time/')
