import numpy as np
from keras.optimizers import SGD, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import clone_model,load_model
from datasets import get_data, get_training_data
from models import get_model
from util import combine_result
import time
import argparse
from tensorflow.python.lib.io import file_io
from keras.datasets import cifar10
from keras.utils import np_utils

def main(job_dir,**args):
    NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100}
    dataset = "cifar-10"
    init_noise_ratio = 0
    data_ratio = 20
    X_train, y_train, X_test, y_test, un_selected_index = get_data(dataset, init_noise_ratio, data_ratio, random_shuffle=False)

    image_shape = X_train.shape[1:]
    model = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=NUM_CLASSES[dataset])
    optimizer = Nadam(lr=0.01, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


    datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip = False,  # randomly flip images
        )
    datagen.fit(X_train)
    epochs_init = 60
    batch_size = 128
    h  =   model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0]//batch_size, epochs=epochs_init,
                            validation_data=(X_test, y_test)
                            )

    np.save(file_io.FileIO(job_dir + 'result/cifar10_nadam.npy', 'w'), h.history)


##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
