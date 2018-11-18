import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils
from labeling.categories import CATEGORIES


def show_examples(examples):
    print(examples['path'])
    images = utils.get_images(examples['path_to_image'])

    # axes
    fig, axes = plt.subplots(2, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    axes = axes.flatten()

    # iterate over subplots and rows of data
    for ax, (_, image) in zip(axes, images.iteritems()):
        ax.imshow(image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.subplots_adjust(*(0.01 for _ in range(6)))
    plt.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.01, 0.01)
    plt.show()


def main():
    examples = pd.read_csv(utils.LABELLING_OUTPUT_PATH)
    examples['label'] = examples['label'].astype(np.int)
    examples = examples.sample(frac=1).reset_index(drop=True)
    examples = pd.merge(examples, CATEGORIES)

    for label, label_df in examples.groupby(['label', 'text']):
        if label_df.shape[0] > 40:
            show_examples(label_df.iloc[:6])
            print(label)


if __name__ == '__main__':
    main()