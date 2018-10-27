import base64
from io import BytesIO

import flask
import numpy as np
import tensorflow as tf
from keras_applications import imagenet_utils
from keras_preprocessing.image import img_to_array
from tensorflow import keras
from tensorflow.keras.preprocessing import image as kimage

import utils
from classification import data_handler as dh

app = flask.Flask(__name__)


# we need to redefine our metric function in order
# to use it when loading the model
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc


# load the model, and pass in the custom metric function
def load_model():
    return keras.models.load_model(utils.MODEL_SAVE_PATH, custom_objects={'auc': auc})


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


graph = tf.get_default_graph()
model = load_model()


@app.route("/classify", methods=["POST"])
def classify():
    """

    :return:
    """
    data = {"success": False}

    if flask.request.method == "POST":
        pass

    return flask.jsonify(data)
    # img = kimage.img_to_array(kimage.load_img(
    #     path=BytesIO(base64.b64decode(flask.request.form['b64'])),
    #     target_size=dh.IMAGE_SHAPE)) / 255.
    # img = img.astype(np.float16)
    #
    # # get the request parameters
    # params = flask.request.json
    # print(params)
    # if params == None:
    #     params = flask.request.args
    #
    # # if parameters are found, echo the msg parameter
    # if params != None:
    #     data["response"] = params.get("msg")
    #     data["success"] = True
    #
    # # return a response in json format
    # return flask.jsonify(data)


app.run(host='0.0.0.0')