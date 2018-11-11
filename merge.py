import tensorflow as tf
from tensorflow.python.platform import gfile

import utils


def print_graph(graph):
    print()
    print('Operations:')
    assert graph is not None
    ops = graph.get_operations()
    for op in ops:
        print('- {0:20s} "{1}" ({2} outputs)'.format(op.type, op.name, len(op.outputs)))

    print()
    print('Sources (operations without inputs):')
    for op in ops:
        if len(op.inputs) > 0:
            continue
        print('- {0}'.format(op.name))

    print()
    print('Operation inputs:')
    for op in ops:
        if len(op.inputs) == 0:
            continue
        print('- {0:20}'.format(op.name))
        print('  {0}'.format(', '.join(i.name for i in op.inputs)))

    print()
    print('Tensors:')
    for op in ops:
        for out in op.outputs:
            if len(out.name) < 40:
                print('- {0:20} {1:10} "{2}"'.format(str(out.shape), out.dtype.name, out.name))


with tf.Session() as persisted_sess:

    with gfile.FastGFile(utils.PATH_TO_CLASS_GRAPH, 'rb') as f:
        graph_def_class = tf.GraphDef()
        graph_def_class.ParseFromString(f.read())

    with gfile.FastGFile(utils.PATH_TO_DETECTION_GRAPH, 'rb') as f:
        graph_def_detection = tf.GraphDef()
        graph_def_detection.ParseFromString(f.read())

    g3 = tf.Graph()
    with g3.as_default():
        image_input = tf.placeholder(tf.uint8, name="image_input")
        image_float_input = tf.divide(tf.to_float(image_input), 256)

        res1_list = ["final_result"]
        res1_name = 'class'
        res1 = tf.import_graph_def(graph_def_class,
                                   input_map={"Placeholder": image_float_input},
                                   return_elements=res1_list,
                                   name=res1_name)

        res2_list = ["detection_classes", "num_detections", "detection_scores", "detection_boxes"]
        res2_name = 'detect'
        res2 = tf.import_graph_def(graph_def_detection,
                                   input_map={"image_tensor": image_input},
                                   return_elements=res2_list,
                                   name=res2_name)

        result_dict = {out: '%s/%s:0' % (res1_name, out) for out in res1_list}
        result_dict.update({out: '%s/%s:0' % (res2_name, out) for out in res2_list})
        result_dict = {out: g3.get_tensor_by_name(name) for out, name in result_dict.items()}

        # Finally we serialize and dump the output graph to the filesystem
        output_graph_def = tf.get_default_graph().as_graph_def()

        # We use a built-in TF helper to export variables to constants
        # output_graph_def = tf.graph_util.convert_variables_to_constants(
        #     sess,  # The session is used to retrieve the weights
        #     tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
        #     [val.name for val in result_dict.values()]  # The output node names are used to select the usefull nodes
        # )

        with tf.gfile.GFile(utils.PATH_TO_MERGED_GRAPH, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))







