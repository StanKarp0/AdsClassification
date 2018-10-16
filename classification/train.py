import os

from matplotlib import pyplot as plt

from classification import data_handler, class_model

BATCH_SIZE = 16
EPOCHS = 1


def main():

    # Creating paths dataframes or using existing ones.
    if not (os.path.isfile(data_handler.TRAIN_CSV_FILENAME) and os.path.isfile(data_handler.TRAIN_CSV_FILENAME)):
        data_handler.construct_path_csv()

    # Creating eager generator for data loading.
    train_generator = data_handler.PittAdsSequence(data_handler.TRAIN_CSV_FILENAME, BATCH_SIZE)
    test_iterator = data_handler.PittAdsSequence(data_handler.TEST_CSV_FILENAME, BATCH_SIZE)

    # Model.
    model = class_model.get_model()
    history = class_model.AccuracyHistory()

    # Fitting model using generator functions.
    model.fit_generator(generator=train_generator,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=test_iterator,
                        callbacks=[history])

    # Saving model.
    model.save_weights('class.h5')

    # Refill generator.
    test_iterator = data_handler.PittAdsSequence(data_handler.TEST_CSV_FILENAME, BATCH_SIZE)
    score = model.evaluate(test_iterator, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.plot(range(1, EPOCHS + 1), history.get_accuracy())
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    # vis.show_images(train_df)


if __name__ == '__main__':
    main()