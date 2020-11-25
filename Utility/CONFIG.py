# from Workers.Cifar_Epoch import CifarEpochWorker
# # TODO: Why TF is CifarTrainsetWorker not working??
# from Workers.Cifar_Trainset import CifarTrainsetWorker
# from Workers.Cifar_Hybrid import CifarHybridWorker
# from Workers.Mnist_Epoch import MnistEpochWorker
# from Workers.Mnist_Trainset import MnistTrainsetWorker
# from Workers.Mnist_Hybrid import MnistHybridWorker

DEBUG = False
SEED = 0

GCLOUD = False
REUSE_MODEL = False
STORE_INTERMEDIARIES = True
STORE_VERBOSE_CSV = True





MNIST_TRAINSET_SIZE = 50000
MNIST_VAL_SIZE = 10000
CIFAR_TRAINSET_SIZE = 40000
CIFAR_VAL_SIZE = 10000
FASHION_TRAINSET_SIZE = 50000
FASHION_VAL_SIZE = 10000


EPOCH_BASED_METHODS = ['epoch-with-increasing-trainset', 'InfiniteEpochWithIncreasingTrainset', 'epoch_bohb']
TRAINSET_BASED_METHODS = ['trainset-with-increasing-epc', 'trainset_bohb']

"""
Each worker is to have a fixed_optimizer_status
fixed_optimizer_status = {
    0: "Not fixed budget optimizer",
    1: "Epoch based",
    2: "Trainset based",
    3: "Time based"
}
"""
