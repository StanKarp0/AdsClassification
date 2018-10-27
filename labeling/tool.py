import os
import sys
from itertools import product

import pandas as pd
from PyQt5 import QtGui
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QButtonGroup, QGridLayout, QCheckBox, QVBoxLayout, \
    QHBoxLayout, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QTableWidget, QTableWidgetItem
from skimage import transform

import utils
from classification import data_handler as dh

SCENE_SIZE = 900


class LabelingTool(QMainWindow):

    def __init__(self):
        super().__init__()

        # general widget
        widget = QWidget(self)
        self.setCentralWidget(widget)

        # horizontal layout
        h_layout = QHBoxLayout(widget)
        widget.setLayout(h_layout)

        # pixmap
        scene_rect = QRectF(0., 0., SCENE_SIZE, SCENE_SIZE)
        graphics_scene = QGraphicsScene(scene_rect, widget)
        graphics_view = QGraphicsView(graphics_scene, widget)
        graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        h_layout.addWidget(graphics_view)

        # vertical layout
        v_layout = QVBoxLayout(widget)
        h_layout.addLayout(v_layout)

        # load labeled data
        if os.path.isfile(utils.LABELLING_OUTPUT_PATH):
            self._labeled_data: pd.DataFrame = pd.read_csv(utils.LABELLING_OUTPUT_PATH)
        else:
            self._labeled_data: pd.DataFrame = pd.DataFrame(columns=['path', 'label', 'path_to_image'])

        # load dropped data
        if os.path.isfile(utils.LABELLING_DROPPED_PATH):
            self._dropped_data: pd.DataFrame = pd.read_csv(utils.LABELLING_DROPPED_PATH)
        else:
            self._dropped_data: pd.DataFrame = pd.DataFrame(columns=['path', 'label', 'path_to_image'])

        # load all data
        unlabeled_data, _ = dh.get_paths(utils.INPUT_DIRECTION, utils.LABELS_PATH, 1)

        # filter previously labeled and dropped
        previous_labeled = unlabeled_data['path'].isin(self._labeled_data['path'])
        previous_dropped = unlabeled_data['path'].isin(self._dropped_data['path'])
        unlabeled_data = unlabeled_data[~(previous_labeled | previous_dropped)]
        self._unlabeled_data = unlabeled_data.groupby(['path', 'path_to_image']).apply(labels2list).reset_index()

        # init categories buttons
        self._category_group = QButtonGroup(widget)
        self._category_group.setExclusive(False)
        category_grid = QGridLayout(widget)
        v_layout.addItem(category_grid)
        grid_points = pd.DataFrame(list(product(range(10), range(3))), columns=['row', 'column'])
        grid_points = pd.concat((dh.CATEGORIES, grid_points.iloc[:dh.NUM_CATEGORIES]), axis=1)

        # iter over every category and create corresponding buttons
        for _, grid_row in grid_points.iterrows():
            button = QCheckBox(self.tr(grid_row['text']), widget)
            button.label = grid_row['label']
            category_grid.addWidget(button, grid_row['row'], grid_row['column'])
            self._category_group.addButton(button)

        # action buttons
        button_layout = QHBoxLayout(widget)
        v_layout.addLayout(button_layout)
        drop_button = QPushButton(self.tr('Drop'), widget)
        next_button = QPushButton(self.tr('Next'), widget)
        drop_button.clicked.connect(self._on_drop)
        next_button.clicked.connect(self._on_load_next)
        drop_button.setShortcut('A')
        next_button.setShortcut('D')
        button_layout.addWidget(drop_button)
        button_layout.addWidget(next_button)

        # result table
        self._table = QTableWidget(self)
        self._table.setRowCount(29)
        self._table.setColumnCount(2)
        for key, value in dh.CATEGORIES['text'].iteritems():
            self._table.setItem(key, 0, QTableWidgetItem(str(value)))
        v_layout.addWidget(self._table)
        self._reset_table()

        # image loading logic
        self._pixmap = QGraphicsPixmapItem()
        graphics_scene.addItem(self._pixmap)
        self._current_index = 0
        self.load_current_image()

    def load_current_image(self):
        current_row_df = self._unlabeled_data.iloc[[self._current_index]]
        # current_labels = current_row_df.iloc[0][0]
        for button in self._category_group.buttons():
            # button.setChecked(button.label in current_labels)
            button.setChecked(False)

        image = dh.get_images(current_row_df['path_to_image']).iloc[0]
        height, width, byte_value = image.shape
        byte_value = byte_value * width
        qt_image = QImage(image, width, height, byte_value, QImage.Format_RGB888)
        qt_image = qt_image.scaled(SCENE_SIZE, SCENE_SIZE)
        self._pixmap.setPixmap(QPixmap.fromImage(qt_image))

    def _on_load_next(self):
        current_row_df = self._unlabeled_data.iloc[[self._current_index]].drop(columns=0)
        current_row_df['key'] = 0
        current_labels = [button.label for button in self._category_group.buttons() if button.isChecked()]
        current_labels = pd.DataFrame({'key': 0, 'label': current_labels})
        current_row_df = pd.merge(current_row_df, current_labels).drop(columns='key')

        self._labeled_data = pd.concat((self._labeled_data, current_row_df))
        self._reset_table()
        self._current_index += 1
        self.load_current_image()

    def _on_drop(self):
        current_row_df = self._unlabeled_data.iloc[[self._current_index]]
        self._dropped_data = pd.concat((self._dropped_data, current_row_df))
        self._current_index += 1
        self.load_current_image()

    def _reset_table(self):
        # self._table.clear()
        for key, count in self._labeled_data.groupby('label')['path'].count().iteritems():
            self._table.setItem(key - 1, 1, QTableWidgetItem(str(count)))

    def closeEvent(self, event: QtGui.QCloseEvent):
        QMainWindow.closeEvent(self, event)
        self._labeled_data[['label', 'path', 'path_to_image']].to_csv(utils.LABELLING_OUTPUT_PATH, index=False)
        self._dropped_data[['label', 'path', 'path_to_image']].to_csv(utils.LABELLING_DROPPED_PATH, index=False)


def labels2list(group: pd.DataFrame) -> list:
    return list(group['label'])


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # w = QWidget()
    # w.resize(250, 150)
    # w.move(300, 300)
    # w.setWindowTitle('Simple')
    window = LabelingTool()
    window.show()

    sys.exit(app.exec_())