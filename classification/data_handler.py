import glob
import pandas as pd
from skimage import transform, color

import skimage.io
from os import path
import os
import numpy as np

# Data config constant
IMAGE_SIZE: int = 100
IMAGE_FLATTEN: int = IMAGE_SIZE * IMAGE_SIZE
IMAGE_SHAPE: tuple = (IMAGE_SIZE, IMAGE_SIZE)

NUM_CATEGORIES: int = 20
CATEGORIES: pd.DataFrame = pd.DataFrame({
    'label': np.arange(NUM_CATEGORIES, dtype=np.int) + 1,
    'text': ['Clothing & Shoes',
             'Automotive',
             'Baby',
             'Health & Beauty',
             'Media',
             'Consumer Electronics',
             'Console & Video Games',
             'Tools & Hardware',
             'Outdoor Living',
             'Grocery',
             'Home',
             'Betting',
             'Jewelery & Watches',
             'Musical instruments',
             'Stationery & Office Supplies',
             'Pet Supplies',
             'Computer Software',
             'Sports',
             'Toys & Games',
             'Social Dating Sites']})


def _get_image(image_path: str) -> np.array:
    """
    Function process input image path
    :param image_path: path to image
    :return: np.array of size (IMAGE_SIZE, IMAGE_SIZE, 1)
    """
    image = skimage.io.imread(image_path)

    # case when png animation - first dimension contains animation frames
    if image.ndim > 3:
        image = image[0]

    # image resize and conversion to grey scale
    image = transform.resize(image, IMAGE_SHAPE)
    image = color.rgb2grey(image)

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


def get_paths(input_directory: str, train_ratio:float=0.7) -> tuple:
    """
    Function prepare data
    :param input_directory: path to data directory
    :param train_ratio:
    :return: tuple (train_df: pd.DataFrame, test_df: pd.DataFrame)
    """
    label_dfs = []

    # list of labels
    directories = [d for d in os.listdir(input_directory) if path.isdir(path.join(input_directory, d))]

    # iteration over categories folders
    for label in directories:
        # find every file in category folder
        label_directory = path.join(input_directory, label)
        input_paths = glob.glob(label_directory + '/*.png')

        # create dataframe for label
        label_df = pd.DataFrame({'path': input_paths, 'label': int(label)})
        label_dfs.append(label_df)

    # create one dataframe from list of dataframes with different labels
    input_df = pd.concat(label_dfs, ignore_index=True)

    # shuffle and split data
    train, test = _shuffle_split_train_test(input_df, train_ratio)

    return train, test


def _label_to_hot_one(label: pd.Series) -> np.ndarray:
    """
    Function transform labels to hot_one like array.
    :param label:
    :return:
    """
    arr = np.zeros((label.shape[0], NUM_CATEGORIES))
    arr[label.index, label.as_matrix() - 1] = 1
    return arr


def get_data_set(data: pd.DataFrame) -> tuple:
    """

    :param data:
    :return:
    """
    data['image'] = data['path'].apply(_get_image)
    x = np.array(list(data['image']))[:, :, :, None]
    y = _label_to_hot_one(data['label'])
    return x, y


def get_data_set_generator(data: pd.DataFrame, batch_size: int=100):
    """
    Function creates eager images generator.
    :param data:
    :param batch_size:
    :return:
    """
    num_generated = 0
    num_labels = data.shape[0]

    while num_generated < num_labels:
        batch_df = data.iloc[num_generated:num_generated+batch_size]
        num_generated += batch_size
        yield get_data_set(batch_df)
