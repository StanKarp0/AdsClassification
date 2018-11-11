from collections import defaultdict

import tensorflow as tf
from tensorflow.python.platform import gfile
import classification.data_handler as dh
import utils
import pandas as pd
import numpy as np

from classification import vis


def main():

    graph = tf.Graph()
    with graph.as_default():
        with gfile.FastGFile(utils.MERGE_GRAPH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    # input tensor
    image_input = graph.get_tensor_by_name('image_input:0')

    # output tensors
    tensors = ["class/final_result", "detect/detection_classes", "detect/num_detections",
               "detect/detection_scores", "detect/detection_boxes"]
    result_dict = {out: '%s:0' % out for out in tensors}
    result_dict = {out: graph.get_tensor_by_name(name) for out, name in result_dict.items()}

    # examples
    examples_number = 4
    examples = pd.read_csv(utils.LABELLING_OUTPUT_PATH)
    examples = examples.iloc[:examples_number]
    examples['image'] = dh.get_images(examples['path_to_image'], size=299)
    image_examples = np.array(examples['image'].tolist())

    with tf.Session(graph=graph) as sess:
        image_examples = (image_examples * 256).astype(np.int)
        res = sess.run(result_dict, {image_input: image_examples})

        res_list = [defaultdict(dict) for _ in range(examples_number)]

        for key, value in res.items():
            for i in range(examples_number):
                res_list[i][key] = value[i]

        vis.show_images(examples)
        for item in res_list:
            utils.construct_detection_df(item)


if __name__ == '__main__':
    main()