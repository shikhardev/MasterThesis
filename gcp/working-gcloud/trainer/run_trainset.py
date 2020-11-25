import argparse
from trainer.util import run_optimizer_process

from trainer.Fashion_Trainset_Worker import FashionTrainsetWorker as worker
# from trainer.MNIST_TRAINSET import MnistTrainsetWorker as worker
# from trainer.CIFAR_TRAINSET import CifarTrainsetWorker as worker

def main(job_dir, **args):
    job_dir += '20200412/FASHION/Trainset/'
    methods = ['bohb', 'hyperband', 'randomsearch']
    num_iterations = 5
    experiment_count = 5
    min_budget = 1/128
    max_budget = 1.0


    # min_budget = 0.0032
    # max_budget = 0.04
    # experiment_count = 1
    # methods = ['bohb']
    # num_iterations = 5

    eta = 3
    verbose = True
    for m in methods:
        for _ in range(experiment_count):
            temp_dir = job_dir + 'Run_' + str(_) + '/'
            res_dir = temp_dir + 'Results/'
            work_dir = temp_dir + 'Work/'
            res = run_optimizer_process(method=m,
                                        run_id=str(_),
                                        min_budget=min_budget,
                                        max_budget=max_budget,
                                        eta=eta,
                                        verbose=verbose,
                                        worker=worker,
                                        res_dir=res_dir + m + '/',
                                        work_dir=work_dir + m + '/',
                                        num_iterations=num_iterations)
            print ('\n'*10)
            print ("Method: ", m)
            print ("Experiment Number: ", str(_))
            print ('\n'*10)


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
    print (arguments)
    main(**arguments)
