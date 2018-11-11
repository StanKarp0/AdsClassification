import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    # input tensor
    image_input = graph.get_tensor_by_name('image_input:0')

    # output tensors
    tensors = ["class/final_result", "detect/detection_classes", "detect/num_detections",
               "detect/detection_scores", "detect/detection_boxes"]
    tensor_dict = {out: '%s:0' % out for out in tensors}
    tensor_dict = {out: graph.get_tensor_by_name(name) for out, name in tensor_dict.items()}

    # examples
    examples_number = 32
    examples = pd.read_csv(utils.LABELLING_OUTPUT_PATH)
    examples = examples.sample(frac=1).reset_index(drop=True)
    examples = examples.iloc[:examples_number]
    examples['image'] = utils.get_images(examples['path_to_image'])

    # display
    utils.show_images(examples, utils.MAPPER_CLASSIFICATION)

    with tf.Session(graph=graph) as sess:

        for _, row in examples.iterrows():
            image = utils.transform_image(row['image'])
            result = sess.run(tensor_dict, {image_input: image})
            result = {key: value[0] for key, value in result.items()}

            detections = utils.construct_detection_df(result)
            classification: np.ndarray = result['class/final_result']
            classification = utils.decode_classification(classification)
            print(detections)
            print(classification)

            image = utils.visualize_boxes_and_labels(row['image'], detections)
            plt.imshow(image)
            plt.show()


if __name__ == '__main__':
    main()