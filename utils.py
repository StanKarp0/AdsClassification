import json
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
from PIL import Image, ImageDraw, ImageFont
from skimage import transform

# Data config constant
IMAGE_SIZE: int = 299
IMAGE_FLATTEN: int = IMAGE_SIZE * IMAGE_SIZE
IMAGE_SHAPE: tuple = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_INPUT_SHAPE: tuple = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Constant path variables for classification
INPUT_DIRECTION: str = r'/home/wojciech/Studia/Ads/'

# Graph paths
PATH_TO_CLASS_GRAPH: str = os.path.join(INPUT_DIRECTION, 'ads-classification.pb')
PATH_TO_DETECTION_GRAPH = os.path.join(INPUT_DIRECTION, 'ads-detection.pb')
PATH_TO_MERGED_GRAPH: str = os.path.join(INPUT_DIRECTION, 'ads.pb')

# Labeling tool paths
LABELS_PATH: str = os.path.join(INPUT_DIRECTION, 'Topics.json')
LABELLING_OUTPUT_PATH = os.path.join(INPUT_DIRECTION, 'labels.csv')
LABELLING_DROPPED_PATH = os.path.join(INPUT_DIRECTION, 'dropped.csv')
LABELLING_SORTED_DIRECTORY = os.path.join(INPUT_DIRECTION, 'Dataset')

# Project files
_current_path = os.path.dirname(os.path.realpath(__file__))
PATH_MAPPER_DETECTION = os.path.join(_current_path, r'oid_bbox_trainable_label_map.csv')
PATH_MAPPER_CLASSIFICATION = os.path.join(_current_path, r'categories.csv')
MAPPER_DETECTION: pd.DataFrame = pd.read_csv(PATH_MAPPER_DETECTION, index_col='id')
MAPPER_CLASSIFICATION: pd.DataFrame = pd.read_csv(PATH_MAPPER_CLASSIFICATION, index_col='label')


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

detection_cords_columns = ['ymin', 'xmin', 'ymax', 'xmax']


def _get_color(detection_class: int) -> str:
    """
    Function return color corresponding to detection class
    :param detection_class: int
    :return: str - color from STANDARD_COLORS array
    """
    return STANDARD_COLORS[detection_class % len(STANDARD_COLORS)]


def construct_detection_df(detections: dict) -> pd.DataFrame:

    detections_df = pd.DataFrame(detections['detect/detection_boxes'], columns=detection_cords_columns)
    detections_df['class'] = detections['detect/detection_classes'].astype(np.int)
    detections_df['score'] = detections['detect/detection_scores']

    # calculating box color
    detections_df['color'] = detections_df['class'].apply(_get_color)

    # filter detection with score 0
    detections_df = detections_df[detections_df['score'] > 0]

    # calculate display name
    detections_df = pd.merge(detections_df, MAPPER_DETECTION, left_on='class', right_on='id')

    # label name
    detections_df['label'] = detections_df.display_name + (detections_df.score * 100).map(': {:,.0f}%'.format)

    return detections_df


def visualize_boxes_and_labels(img: np.ndarray,
                               detections: pd.DataFrame,
                               min_score_thresh: float=.5,
                               max_boxes_to_draw: int=20,
                               thickness: int=2) -> np.ndarray:
    """
    Function draws detection boxes on input image.
    :param img: np.ndarray - shape (?, ?, 3)
    :param detections: pd.DataFrame
    :param max_boxes_to_draw:
    :param min_score_thresh: minimum detection score
    :param thickness: int line width
    :return:
    """
    # image variables
    im_height, im_width, _ = img.shape
    image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)

    # loading font
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # only detections with score greater then min_score_thresh and not more then max_boxes_to_draw
    detections = detections[detections['score'] > min_score_thresh].iloc[:max_boxes_to_draw]

    # iterate over every detection
    for _, row in detections.iterrows():
        ymin, xmin, ymax, xmax = tuple(row[detection_cords_columns].tolist())

        # calculating x,y positions from absolute positions: 0-1 range
        left, right, top, bottom = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        cords = [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)]

        # drawing line between points
        draw.line(cords, width=thickness, fill=row['color'])

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        text_width, text_height = font.getsize(row['label'])
        total_display_str_height = (1 + 2 * 0.05) * text_height
        margin = np.ceil(0.05 * text_height)
        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        # cords of rectangle behind text
        rectangle_cords = [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)]
        draw.rectangle(rectangle_cords, fill=row['color'])

        # text - class name
        text_cords = (left + margin, text_bottom - text_height - margin)
        draw.text(text_cords, row['label'], fill='black', font=font)

    return np.array(image_pil)


def decode_classification(preds: np.ndarray) -> pd.DataFrame:
    """
    Function returns dataframe with pretty class names and scores
    :param preds: np.ndarray
    :return: pd.DataFrame
    """
    preds_df = pd.DataFrame(preds.T, columns=['score'], index=MAPPER_CLASSIFICATION.index)
    preds_df = pd.concat((MAPPER_CLASSIFICATION, preds_df), axis=1)
    return preds_df.sort_values('score', ascending=False)


def show_images(data: pd.DataFrame, categories: pd.DataFrame, show_result: bool=False, plot_shape: tuple=(4, 4)):
    """
    Function shows first images from data.
    :param categories:
    :param data: pd.DataFrame - should contains 'label': str and 'image': np.array columns
    :param show_result: bool - show classification result require 'pred': str in data dataframe
    :param plot_shape: tuple - number of example images and plot shape
    :return: None
    """
    assert ['image', 'label'] in data and 'pred' in data if show_result else True, \
        '"label", "image" columns required. Column "pred" required when show_result == True'

    # axes
    fig, axes = plt.subplots(*plot_shape)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    axes = axes.flatten()

    # append label text
    data = pd.merge(data, categories.reset_index()).rename(columns={'text': 'text_label'})
    if show_result:
        data = pd.merge(data, categories, left_on='pred', right_on='label')\
            .rename(columns={'text': 'text_pred'})

    # iterate over subplots and rows of data
    for ax, (_, row) in zip(axes, data.iterrows()):
        ax.imshow(row['image'],  cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # label construction
        xlabel = "True: {0}".format(row['text_label'])
        if show_result:
            xlabel += ", Pred: {0}".format(row['text_pred'])
        ax.set_xlabel(xlabel)

    plt.show()


def show_error(data: pd.DataFrame, categories: pd.DataFrame, plot_shape: tuple=(4, 4)):
    """
    Function shows errors in prediction.
    :param categories:
    :param data: pd.DataFrame - 'image', 'label', 'pred' columns required
    :param plot_shape: plot shape
    :return: None
    """
    assert ['image', 'label', 'pred'] in data, '"image", "label", "pred" columns required'

    errors = data[data['label'] != data['pred']]
    show_images(errors, categories, show_result=True, plot_shape=plot_shape)


def show_categories(data: pd.DataFrame, categories: pd.DataFrame, show_result: bool=False, errors: bool=False):
    """
    Function shows examples of labels
    :param categories:
    :param data: pd.DataFrame - columns: 'label' and 'image' required
    :param show_result: bool - show predicted label
    :param errors: filter only errors
    :return: None
    """
    assert ['image', 'label'] in data and 'pred' in data if show_result or errors else True, \
        '"label", "image" columns required. Column "pred" required when show_result == True'

    data = data.sample(frac=1).reset_index(drop=True)
    if errors:
        # filter rows where label is different from prediction
        data = data[data['label'] != data['pred']]

    # loop gets first row from every label category
    first_from_cat_df = pd.DataFrame([label_df.iloc[0] for _, label_df in data.groupby('label', as_index=False)])
    show_images(first_from_cat_df, categories, show_result, (5, 4))


def _get_image(image_path: str) -> np.array:
    """
    Function process input image path
    :param image_path: path to image
    :return: np.array
    """
    image = skimage.io.imread(image_path)

    # case when png animation - first dimension contains animation frames
    if image.ndim > 3:
        image = image[0]
    # in case of gray only image.
    elif image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))

    return image


def transform_image(image: np.ndarray, size=IMAGE_SIZE):
    image_shape: tuple = (size, size)
    image_input_shape: tuple = (size, size, 3)

    # image resize and conversion
    image = transform.resize(image, image_shape)
    assert image.shape == image_input_shape, 'Image: %s' % str(image.shape)

    image = (image * 255).astype(np.int)
    image = np.expand_dims(image, 0)

    return image


def _shuffle_split_train_test(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple:
    """
    Function shuffle data and splits dataframe into train and test sets
    :param df: pd.DataFrame
    :param train_ratio: float
    :return: tuple (train_df: pd.DataFrame, test_df: pd.DataFrame)
    """
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # train, test split
    train_shape = int(df.shape[0] * train_ratio)
    train_df = df.iloc[:train_shape]
    test_df = df.iloc[train_shape:].reset_index(drop=True)

    return train_df, test_df


def get_images(paths_to_images: pd.Series) -> pd.Series:
    """

    :param paths_to_images:
    :return:
    """
    return paths_to_images.apply(_get_image)


def get_paths(input_directory: str, label_path: str, train_ratio:float=0.7) -> (pd.DataFrame, pd.DataFrame):
    """
    Function prepare data
    :param label_path:
    :param input_directory: path to data directory
    :param train_ratio:
    :return: tuple (train_df: pd.DataFrame, test_df: pd.DataFrame)
    """

    # Loading Topics json to pandas DataFrame.
    with open(label_path, 'r') as label_path_file:
        labels_json = json.load(label_path_file)

        # Topics dict structure [{path_to_image: [list of topics, various list sizes]}]
        # Flatten inner list - list of topics. Index as path and values topics. Multiple rows
        # for one image
        labels_df = pd.DataFrame.from_dict(labels_json, orient='index').unstack(0).dropna()

        # Filter out not numerical labels.
        labels_df = labels_df[labels_df.str.contains(r'^[0-9]{1,2}$')].astype(int)

        # Adding path as column and dropping duplicates of pair (path, topic).
        labels_df = labels_df.reset_index(1).drop_duplicates().reset_index(drop=True)

        # Rename df columns.
        labels_df = labels_df.rename(columns={'level_1': 'path', 0: 'label'})

        # Append absolute path to image
        labels_df['path_to_image'] = input_directory + labels_df.path.astype(str)

    # shuffle data and split into test and train set
    train, test = _shuffle_split_train_test(labels_df, train_ratio)

    return train, test
