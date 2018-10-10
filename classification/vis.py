from matplotlib import pyplot as plt
import pandas as pd

from classification import data_handler


def show_images(data: pd.DataFrame, show_result: bool=False, plot_shape: tuple=(4, 4)):
    """
    Function shows first images from data.
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
    data = pd.merge(data, data_handler.CATEGORIES).rename(columns={'text': 'text_label'})
    if show_result:
        data = pd.merge(data, data_handler.CATEGORIES, left_on='pred', right_on='label')\
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