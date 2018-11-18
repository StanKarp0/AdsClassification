import os
from shutil import copyfile

import pandas as pd

import utils
from labeling.categories import CATEGORIES


def main():

    labels = pd.read_csv(utils.LABELLING_OUTPUT_PATH)
    labels['label'] = labels['label'].astype(int)
    count = labels.groupby('label', as_index=False)['path'].count().rename(columns={'path': 'count'})
    count = count[count['count'] > 30]

    cats = pd.merge(CATEGORIES, count)
    cats['text'] = cats['text'].str.replace(' ', '_')

    labels = pd.merge(labels, cats)
    labels['path'] = labels['path'].str.replace('/', '')
    labels['save_path'] = labels['text'] + '/' + labels['path']

    cats = cats.sort_values('text').reset_index(drop=True).reset_index()
    cats = cats.rename(columns={'index': 'label', 'label': 'all_label'})
    cats.to_csv(utils.PATH_MAPPER_CLASSIFICATION, index=False)

    for (label_id, label_text), label_df in labels.groupby(['label', 'text']):
        directory = os.path.join(utils.LABELLING_SORTED_DIRECTORY, label_text)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        for _, row in label_df.iterrows():
            image_dir = os.path.join(utils.LABELLING_SORTED_DIRECTORY, row['save_path'])
            if not os.path.exists(image_dir):
                copyfile(row['path_to_image'], image_dir)
                print('Copied: ', image_dir)


if __name__ == '__main__':
    main()