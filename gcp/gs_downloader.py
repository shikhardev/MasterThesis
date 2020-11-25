import IPython
from tensorflow_core.python.lib.io.file_io import get_matching_files_v2 as glob
from subprocess import run as run
from os.path import isdir
from os import makedirs, remove



if __name__ == '__main__':
    names = {}
    """
    gsutil cp -r "gs://data_shikhar_skd/20200410/CIFAR/Epochs/Run_5/Results" .
    """
    names['mnist_epoch_1'] = "gs://data_shikhar/20200405/MNIST/Epochs/"
    names['mnist_epoch_2'] = "gs://data_shikhar/20200406/MNIST/Epochs/"
    names['mnist_trainset'] = "gs://data_shikhar/20200410/MNIST/Trainset/"

    names['cifar_trainset'] = "gs://data_shikhar/20200410/CIFAR/Trainset/"
    names['cifar_epochs'] = "gs://data_shikhar_skd/20200410/CIFAR/Epochs/"

    names['fashion_epoch'] = "gs://data_shikhar/20200412/FASHION/Epochs/"
    names['fashion_trainset'] = "gs://data_shikhar/20200412/FASHION/Trainset/"

    home_location = '/Users/shikhar/Desktop/thesis/Analysis\ Results/2020_04_14/'

    cmd = ['gsutil', 'cp', '-r']

    for dataset in names:
        # Need to create a folder in Home location
        runs = glob(names[dataset] + '*')
        for r in runs:
            n = r + '/Results/'  # Should be able to download from here
            methods = glob(n+'*')
            for m in methods:
                method_name = m.split('/')[-1]
                local_location = home_location + dataset + '/'
                if not isdir(local_location):
                    makedirs(local_location)
                run(cmd + [m, local_location])



# s.run (['gsutil', 'cp', '-r', 'gs://data_shikhar/20200405/MNIST/Epochs/Run_0/Results/', '.'])