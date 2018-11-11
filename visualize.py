import tensorflow as tf
from tensorflow.python.platform import gfile

import utils


def main():

    graph = tf.Graph()
    with graph.as_default():
        with gfile.FastGFile(utils.PATH_TO_MERGED_GRAPH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=graph) as sess:
        tf.summary.FileWriter('./graphs', sess.graph)


if __name__ == '__main__':
    main()