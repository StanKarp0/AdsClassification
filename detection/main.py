import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utils
from classification import data_handler, vis
# from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

# PATH_TO_FROZEN_GRAPH: str = \
#     r'/home/wojciech/Studia/izn/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}

            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                # Reframe is required to translate mask from box coordinates to image coordinates and
                # fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def main():
    examples, _ = data_handler.get_paths(utils.INPUT_DIRECTION, utils.LABELS_PATH, 1.)
    examples = examples.iloc[:16]
    examples['image'] = data_handler.get_images(examples['path_to_image'])
    vis.show_images(examples)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(utils.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # category_index = label_map_util.create_category_index_from_labelmap(utils.PATH_TO_LABELS, use_display_name=True)

    for _, row in examples.iterrows():
        output_dict = run_inference_for_single_image(row['image'], detection_graph)
        # print(output_dict)
        vis_util.visualize_boxes_and_labels_on_image_array(
            row['image'],
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        # plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(row['image'])
        plt.show()


if __name__ == '__main__':
    main()