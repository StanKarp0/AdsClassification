import json
import sys
from functools import partial

import numpy as np
import pandas as pd
import skimage.io
from skimage import transform

# Data paths
from tensorflow.python.keras.utils import Sequence

import utils
from classification import categories

TRAIN_CSV_FILENAME: str = 'train_df.csv'
TEST_CSV_FILENAME: str = 'test_df.csv'

# Data config constant
IMAGE_SIZE: int = 224
IMAGE_FLATTEN: int = IMAGE_SIZE * IMAGE_SIZE
IMAGE_SHAPE: tuple = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_INPUT_SHAPE: tuple = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Categories
CATEGORIES: pd.DataFrame = categories.CATEGORIES
NUM_CATEGORIES: int = CATEGORIES.shape[0]


def _get_image(image_path: str, reshape=True) -> np.array:
    """
    Function process input image path
    :param image_path: path to image
    :return: np.array of size (IMAGE_SIZE, IMAGE_SIZE, 3)
    """
    image = skimage.io.imread(image_path)

    # case when png animation - first dimension contains animation frames
    if image.ndim > 3:
        image = image[0]
    # in case of gray only image.
    elif image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))

    # image resize and conversion to grey scale
    if reshape:
        image = transform.resize(image, IMAGE_SHAPE)
        assert image.shape == IMAGE_INPUT_SHAPE, 'Image: %s' % str(image.shape)
    # image = color.rgb2grey(image)

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


def label_to_hot_one(label: pd.Series) -> np.ndarray:
    """
    Function transform labels to hot_one like array.
    :param label:
    :return:
    """
    arr = np.zeros((label.shape[0], NUM_CATEGORIES))
    arr[label.index, label.as_matrix() - 1] = 1
    return arr


def _get_data_set(data: pd.DataFrame) -> tuple:
    """

    :param data:
    :return:
    """
    data['image'] = data['path_to_image'].apply(_get_image)
    x = np.array(list(data['image']))
    y = label_to_hot_one(data['label'].reset_index(drop=True))
    return x, y


def get_images(paths_to_images: pd.Series) -> pd.Series:
    return paths_to_images.apply(partial(_get_image, reshape=False))


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


class PittAdsSequence(Sequence):

    def __init__(self, path_to_df, batch_size=64):
        self._batch_size = batch_size
        self._paths_df = pd.read_csv(path_to_df)
        self._len = int(self._paths_df.shape[0] / batch_size)

    def __getitem__(self, index):
        start = index * self._batch_size
        batch_df = self._paths_df.iloc[start:start + self._batch_size]
        return _get_data_set(batch_df)

    def __len__(self):
        return self._len


def construct_path_csv(train_ratio: float, labels_csv: pd.DataFrame=pd.DataFrame()):
    """

    :return:
    """
    # check if labels pd is defined
    if labels_csv is None:
        # take all
        train_df, test_df = get_paths(utils.INPUT_DIRECTION, utils.LABELS_PATH, train_ratio)
    else:
        # take ony from labels dataframe
        train_df, test_df = _shuffle_split_train_test(labels_csv, train_ratio)

    # Save to temporary files
    train_df.to_csv(TRAIN_CSV_FILENAME, index=False)
    test_df.to_csv(TEST_CSV_FILENAME, index=False)


if __name__ == '__main__':
    construct_path_csv()
