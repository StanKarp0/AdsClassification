from tensorflow import keras
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from classification import data_handler


MODEL_INPUT_SHAPE: tuple = (data_handler.IMAGE_SIZE, data_handler.IMAGE_SIZE, 1)


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

    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     activation='relu',
                     input_shape=MODEL_INPUT_SHAPE))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=1000,
                    activation='relu'))

    model.add(Dense(units=data_handler.NUM_CATEGORIES,
                    activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    return model


