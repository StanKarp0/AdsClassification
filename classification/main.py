import numpy as np
from matplotlib import pyplot as plt

from classification import data_handler, vis, class_model


INPUT_DIRECTION: str = r'/home/wojciech/Studia/izn/ADS16/'
BATCH_SIZE = 100
EPOCHS = 50


def main():
    train_df, test_df = data_handler.get_paths(INPUT_DIRECTION)
    train_x, train_y = data_handler.get_data_set(train_df)
    test_x, test_y = data_handler.get_data_set(test_df)

    model = class_model.get_model()
    history = class_model.AccuracyHistory()

    model.fit(x=train_x,
              y=train_y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(test_x, test_y),
              callbacks=[history])

    model.save_weights('class.h5')
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.plot(range(1, EPOCHS + 1), history.get_accuracy())
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


    # train_generator = data_handler.get_data_set_generator(train_df)
    # train_x, train_y = next(train_generator)
    # train_x = np.array(list(train_df['image']))[:, :, :, None]
    # train_y = data_handler.label_to_hot_one(train_df['label'])




    # vis.show_images(train_df)


if __name__ == '__main__':
    main()