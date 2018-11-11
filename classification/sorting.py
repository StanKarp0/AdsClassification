import pandas as pd

import utils
from labeling.categories import CATEGORIES


def main():

    labels = pd.read_csv(utils.LABELLING_OUTPUT_PATH)
    count = labels.groupby('label', as_index=False)['path'].count().rename(columns={'path': 'count'})
    count = count[count['count'] > 30]

    cats = pd.merge(CATEGORIES, count)[['label', 'text']]
    cats['text'] = cats['text'].str.replace(' ', '_')

    labels = pd.merge(labels, cats)
    labels['path'] = labels['path'].str.replace('/', '')
    labels['save_path'] = labels['text'] + '/' + labels['path']

    to_save = pd.DataFrame({'text': cats['text'], 'description': cats['text']})\
        .reset_index().rename(columns={'index': 'label'})

    to_save.to_csv('categories.csv', index=False)

    # for (label_id, label_text), label_df in labels.groupby(['label', 'text']):
    #     directory = os.path.join(utils.LABELLING_SORTED_DIRECTORY, label_text)
    #     if not os.path.isdir(directory):
    #         os.mkdir(directory)
    #
    #     for _, row in label_df.iterrows():
    #         image_dir = os.path.join(utils.LABELLING_SORTED_DIRECTORY, row['save_path'])
    #         if not os.path.exists(image_dir):
    #             copyfile(row['path_to_image'], image_dir)
    #             print('Copied: ', image_dir)


if __name__ == '__main__':
    main()