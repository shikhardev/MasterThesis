"""
Worker for Example 5 - Keras
============================

In this example implements a small CNN in Keras to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.

We'll optimise the following hyperparameters:

+-------------------------+----------------+-----------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices  | Comment                |
+=========================+================+=================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
+-------------------------+----------------+-----------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD }    | discrete choice        |
+-------------------------+----------------+-----------------+------------------------+
| SGD momentum            |  float         | [0, 0.99]       | only active if         |
|                         |                |                 | optimizer == SGD       |
+-------------------------+----------------+-----------------+------------------------+
| Number of conv layers   | integer        | [1,3]           | can only take integer  |
|                         |                |                 | values 1, 2, or 3      |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | logarithmically varied |
| the first conf layer    |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the second conf layer   |                |                 | of layers >= 2         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the third conf layer    |                |                 | of layers == 3         |
+-------------------------+----------------+-----------------+------------------------+
| Dropout rate            |  float         | [0, 0.9]        | standard continuous    |
|                         |                |                 | parameter              |
+-------------------------+----------------+-----------------+------------------------+
| Number of hidden units  | integer        | [8,256]         | logarithmically varied |
| in fully connected layer|                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+

Please refer to the compute method below to see how those are defined using the
ConfigSpace package.

The network does not achieve stellar performance when a random configuration is samples,
but a few iterations should yield an accuracy of >90%. To speed up training, only
8192 images are used for training, 1024 for validation.
The purpose is not to achieve state of the art on MNIST, but to show how to use
Keras inside HpBandSter, and to demonstrate a more complicated search space.
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

try:
    import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.losses import categorical_crossentropy
except:
    raise ImportError("For this example you need to install keras.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging

logging.basicConfig(level=logging.DEBUG)




DEBUG = False

class MnistCVWorker(Worker):
    def __init__(self, N_train=8192, N_valid=1024, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = 64

        img_rows = 28
        img_cols = 28
        self.num_classes = 10

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # zero-one normalization
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        # y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        # self.x_train, self.y_train = x_train[:N_train], y_train[:N_train]
        # self.x_validation, self.y_validation = x_train[-N_valid:], y_train[-N_valid:]
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        self.input_shape = (img_rows, img_cols, 1)

    def get_model_and_optimizer(self, config):
        model = Sequential()

        model.add(Conv2D(config['num_filters_1'], kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if config['num_conv_layers'] > 1:
            model.add(Conv2D(config['num_filters_2'], kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        if config['num_conv_layers'] > 2:
            model.add(Conv2D(config['num_filters_3'], kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(config['dropout_rate']))
        model.add(Flatten())
        model.add(Dense(config['num_fc_units'], activation='relu'))
        model.add(Dropout(config['dropout_rate']))
        model.add(Dense(self.num_classes, activation='softmax'))

        if config['optimizer'] == 'Adam':
            optimizer = Adam(lr=config['lr'])
        else:
            optimizer = SGD(lr=config['lr'], momentum=config['sgd_momentum'])


        return model, optimizer

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        train_sum = 0
        val_sum = 0
        test_sum = 0
        budget = int (budget)

        skf = StratifiedKFold(n_splits=budget)
        skf.get_n_splits(self.x_train, self.y_train)

        for train_index, val_index in skf.split(self.x_train, self.y_train):
            model = None
            model = Sequential()

            model.add(Conv2D(config['num_filters_1'], kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            if config['num_conv_layers'] > 1:
                model.add(Conv2D(config['num_filters_2'], kernel_size=(3, 3),
                                 activation='relu',
                                 input_shape=self.input_shape))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            if config['num_conv_layers'] > 2:
                model.add(Conv2D(config['num_filters_3'], kernel_size=(3, 3),
                                 activation='relu',
                                 input_shape=self.input_shape))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Dropout(config['dropout_rate']))
            model.add(Flatten())
            model.add(Dense(config['num_fc_units'], activation='relu'))
            model.add(Dropout(config['dropout_rate']))
            model.add(Dense(self.num_classes, activation='softmax'))

            if config['optimizer'] == 'Adam':
                optimizer = Adam(lr=config['lr'])
            else:
                optimizer = SGD(lr=config['lr'], momentum=config['sgd_momentum'])


            model.compile(loss=categorical_crossentropy,
                          optimizer=optimizer,
                          metrics=['accuracy'])

            xtrain, ytrain = self.x_train[train_index], self.y_train[train_index]

            xval, yval = self.x_train[val_index], self.y_train[val_index]

            ytrain = to_categorical(ytrain, self.num_classes)

            yval = to_categorical(yval, self.num_classes)

            if DEBUG:
                epochs = 2
                verbose = 1
            else:
                epochs = 16
                verbose = 0

            res = model.fit(xtrain, ytrain,
                            batch_size=self.batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            validation_data=(xval, yval))

            train_score = model.evaluate(xtrain, ytrain, verbose=0)
            val_score = model.evaluate(xval, yval, verbose=0)
            test_score = model.evaluate(self.x_test, self.y_test, verbose=0)
            train_sum += train_score[1]
            val_sum += val_score[1]
            test_sum += test_score[1]

        # import IPython
        # IPython.embed()
        return ({
            'loss': 1 - (val_sum / budget),  # remember: HpBandSter always minimizes!
            'info': {'test accuracy': (test_sum / budget),
                     'train accuracy': (train_sum / budget),
                     'validation accuracy': (val_sum / budget),
                     'number of parameters': model.count_params(),
                     }

        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9,
                                                      log=False)

        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        num_conv_layers = CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=3, default_value=2)

        num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=64, default_value=16, log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=64, default_value=16, log=True)
        num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=64, default_value=16, log=True)

        cs.add_hyperparameters([num_conv_layers, num_filters_1, num_filters_2, num_filters_3])

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5,
                                                      log=False)
        num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)

        cs.add_hyperparameters([dropout_rate, num_fc_units])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        # You can also use inequality conditions:
        cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
        cs.add_condition(cond)

        return cs


if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2, working_directory='.')
    print(res)
