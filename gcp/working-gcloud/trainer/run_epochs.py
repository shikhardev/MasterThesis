import argparse
from trainer.util import run_optimizer_process
from trainer.CIFAR_EPOCHS import CifarEpochWorker as worker

# from trainer.CIFAR_EPOCHS import CifarEpochWorker as worker
# from MNIST_CNN.MNIST_EPOCHS import MnistEpochWorker

# from gcloud_skd.trainer.MNIST_EPOCHS import MnistEpochWorker as worker
# from gcloud_skd.trainer.util import run_optimizer_process

def main(job_dir, **args):
    job_dir += '20200410/CIFAR/Epochs/'
    methods = ['bohb', 'hyperband', 'randomsearch']
    num_iterations = 5
    experiment_count = 5
    min_budget = 4
    max_budget = 128
    eta = 2
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
            print ('-'*10)
            print ("Method: ", m)
            print ("Experiment Number: ", str(_))
            print ('-'*10)


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