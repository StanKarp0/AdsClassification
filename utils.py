import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constant path variables for classification
from PIL import Image, ImageDraw, ImageFont

INPUT_DIRECTION: str = r'/home/wojciech/Studia/Ads/'
LABELS_PATH: str = r'/home/wojciech/Studia/Ads/Topics.json'
MODEL_SAVE_PATH: str = r'/home/wojciech/Studia/Ads/ads_classification.json'
MODEL_SAVE_PATH_WEIGHTS: str = r'/home/wojciech/Studia/Ads/ads_classification_weights.h5'

# Object detection paths
PATH_TO_FROZEN_GRAPH: str = '/home/wojciech/Studia/Ads/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_LABELS: str = r'/home/wojciech/Dev/ObjectDetection/models/research/object_detection/data/mscoco_label_map.pbtxt'

PATH_TO_CLASS_GRAPH: str = r'/home/wojciech/Studia/Ads/ads-classification.pb'
PATH_TO_DETECTION_GRAPH = r'/home/wojciech/Studia/Ads/ads-detection.pb'

# Merged
MERGE_GRAPH: str = r'/home/wojciech/Studia/Ads/ads.pb'

# Labeling tool paths
LABELLING_OUTPUT_PATH = r'/home/wojciech/Studia/Ads/labels.csv'
LABELLING_DROPPED_PATH = r'/home/wojciech/Studia/Ads/dropped.csv'
LABELLING_SORTED_DIRECTORY = r'/home/wojciech/Studia/Ads/Dataset'


PATH_MAPPER_DETECTION = 'mscoco_label_map.csv'
PATH_MAPPER_CLASSIFICATION = 'categories.csv'
mapper_detection: pd.DataFrame = pd.read_csv(PATH_MAPPER_DETECTION, index_col='id')
mapper_classification: pd.DataFrame = pd.read_csv(PATH_MAPPER_CLASSIFICATION, index_col='label')


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

    detections_df = pd.DataFrame(detections['detection_boxes'], columns=detection_cords_columns)
    detections_df['class'] = detections['detection_classes']
    detections_df['score'] = detections['detection_scores']

    # calculating box color
    detections_df['color'] = detections_df['class'].apply(_get_color)

    # filter detection with score 0
    detections_df = detections_df[detections_df['score'] > 0]

    # calculate display name
    detections_df = pd.merge(detections_df, mapper_detection, left_on='class', right_on='id')

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
    preds_df = pd.DataFrame(preds.T, columns=['score'], index=mapper_classification.index)
    preds_df = pd.concat((mapper_classification, preds_df), axis=1)
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
    data = pd.merge(data, categories).rename(columns={'text': 'text_label'})
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


def show_error(data: pd.DataFrame, plot_shape: tuple=(4, 4)):
    """
    Function shows errors in prediction.
    :param data: pd.DataFrame - 'image', 'label', 'pred' columns required
    :param plot_shape: plot shape
    :return: None
    """
    assert ['image', 'label', 'pred'] in data, '"image", "label", "pred" columns required'

    errors = data[data['label'] != data['pred']]
    show_images(errors, show_result=True, plot_shape=plot_shape)


def show_categories(data: pd.DataFrame, show_result: bool=False, errors: bool=False):
    """
    Function shows examples of labels
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
    show_images(first_from_cat_df, show_result, (5, 4))