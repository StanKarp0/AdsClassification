import base64
from io import BytesIO

import flask
import numpy as np
import tensorflow as tf
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
graph = tf.get_default_graph()
model = keras.models.load_model(utils.MODEL_SAVE_PATH, custom_objects={'auc': auc})


@app.route("/classify", methods=["POST"])
def classify():
    """

    :return:
    """
    img = kimage.img_to_array(kimage.load_img(
        path=BytesIO(base64.b64decode(flask.request.form['b64'])),
        target_size=dh.IMAGE_SHAPE)) / 255.
    img = img.astype(np.float16)

    # get the request parameters
    params = flask.request.json
    print(params)
    if params == None:
        params = flask.request.args

    # if parameters are found, echo the msg parameter
    if params != None:
        data["response"] = params.get("msg")
        data["success"] = True

    # return a response in json format
    return flask.jsonify(data)


app.run(host='0.0.0.0')