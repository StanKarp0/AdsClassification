from tensorflow import keras
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D

from classification import data_handler


MODEL_INPUT_SHAPE: tuple = (data_handler.IMAGE_SIZE, data_handler.IMAGE_SIZE, 3)


class AccuracyHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._accuracy = []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        self._accuracy.append(logs.get('acc'))

    def get_accuracy(self):
        return self._accuracy


def get_model() -> Model:
    """

    :return:
    """
    model = Sequential()

    # part 1
    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=MODEL_INPUT_SHAPE))
    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # part 2
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # part 3
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # # part 4
    # model.add(Conv2D(filters=256,
    #                  kernel_size=(3, 3),
    #                  activation='relu'))
    # model.add(Conv2D(filters=256,
    #                  kernel_size=(3, 3),
    #                  activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense
    model.add(Flatten())

    model.add(Dense(units=500,
                    activation='relu'))

    model.add(Dense(units=data_handler.NUM_CATEGORIES,
                    activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    return model


