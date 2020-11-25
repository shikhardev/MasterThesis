import argparse

from util import run_optimizer_process, store_experiment_details
# from Worker_Iter import Worker_CIFAR10_Iter as worker
from Worker_CIFAR10_Iter import PyTorchWorker as worker

def main(job_dir, **args):
    job_dir += 'cifar10_experiments/'
    methods = ['bohb', 'randomsearch', 'hyperband']
    num_iterations = 5
    experiment_count = 10
    min_budget = 3
    max_budget = 81
    eta = 3
    verbose = True
    store_experiment_details(methods=methods,
                             number_of_runs=experiment_count,
                             budget_type='iter',
                             min_budget=min_budget,
                             max_budget=max_budget,
                             eta=eta,
                             num_iterations=num_iterations,
                             file_location=job_dir)
    for m in methods:
        for _ in range (experiment_count):
            run_optimizer_process(method=m,
                                  run_id=str(_),
                                  min_budget=min_budget,
                                  max_budget=max_budget,
                                  eta=eta,
                                  verbose=verbose,
                                  worker=worker,
                                  dest_dir=job_dir,
                                  num_iterations=num_iterations)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    # run_optimizer_process('bohb', '0', 1, 2, worker, job_dir, True, 2, 1)
    main(**arguments)

