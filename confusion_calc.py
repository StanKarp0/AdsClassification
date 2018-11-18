import pandas as pd
import tensorflow as tf

import utils


def main():
    results = pd.read_csv('results.csv', index_col='index')
    categories = pd.read_csv('categories.csv')

    pairs = pd.merge(results, categories)[['result', 'label']]

    confusion_matrix_tensor = tf.confusion_matrix(labels=pairs['label'], predictions=pairs['result'])

    sess = tf.Session()
    with sess.as_default():
        confusion_matrix = sess.run(confusion_matrix_tensor)
        print(confusion_matrix)
        utils.plot_confusion_matrix(confusion_matrix, categories['text'], normalize=False)


if __name__ == '__main__':
    main()

    # results = pd.read_csv('results.csv', index_col='index')
    # results2 = pd.read_csv('result2.csv', index_col='index')
    #
    # results3 = pd.concat((results, results2), ignore_index=True)
    #
    # results3.to_csv('results.csv')