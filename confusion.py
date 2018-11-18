import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile

import utils
from labeling.categories import CATEGORIES


def main():

    graph = tf.Graph()
    with graph.as_default():
        with gfile.FastGFile(utils.PATH_TO_MERGED_GRAPH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    # input tensor
    image_input = graph.get_tensor_by_name('image_input:0')

    # output tensors
    tensors = ["class/final_result", "detect/detection_classes", "detect/num_detections",
               "detect/detection_scores", "detect/detection_boxes"]
    tensor_dict = {out: '%s:0' % out for out in tensors}
    tensor_dict = {out: graph.get_tensor_by_name(name) for out, name in tensor_dict.items()}

    # examples
    start = 1003
    examples_number = 500
    examples = pd.read_csv(utils.LABELLING_OUTPUT_PATH)
    examples = examples.sample(frac=1).reset_index(drop=True)
    examples = examples.iloc[start: start + examples_number]

    # categories = pd.read_csv('categories.csv')
    results = []

    with tf.Session(graph=graph) as sess:

        for row_id, row in examples.iterrows():
            image = utils.get_images(pd.Series(row['path_to_image']))[0]
            image = utils.transform_image(image)
            result = sess.run(tensor_dict, {image_input: image})
            result = {key: value[0] for key, value in result.items()}

            classification: np.ndarray = result['class/final_result']
            class_max = np.argmax(classification)
            results.append((class_max, row['label']))
            print(row_id, class_max, row['label'])

    results = pd.DataFrame(results, columns=['result', 'all_label'])
    results.to_csv('result2.csv')
    # results = pd.merge(results, categories)

    print(results)


if __name__ == '__main__':
    main()