import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QDialog, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QSize

class ImageViewer(QDialog):
    def __init__(self, pixmap):
        super().__init__()

        self.setWindowTitle("放大影像")

        self.original_pixmap = pixmap
        self.scaled_pixmap = pixmap
        self.label = QLabel(self)
        self.label.setPixmap(self.scaled_pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.mousePressEvent = self.mousePressEvent
        self.label.mouseMoveEvent = self.mouseMoveEvent
        self.label.wheelEvent = self.wheelEvent
        self.start_pos = QPoint()

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        self.resize(800, 600)  # 初始視窗大小

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = event.pos() - self.start_pos
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_area.horizontalScrollBar().value() - delta.x())
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() - delta.y())
            self.start_pos = event.pos()

    def wheelEvent(self, event):
        num_steps = event.angleDelta().y() / 120
        scale_factor = 1.15 if num_steps > 0 else 1/1.15

        old_size = self.scaled_pixmap.size()

        new_width_float = self.scaled_pixmap.width() * scale_factor  # 保留浮點數版本
        new_height_float = self.scaled_pixmap.height() * scale_factor # 保留浮點數版本

        new_width = int(round(new_width_float))  # 使用 round() 四捨五入後轉換為 int
        new_height = int(round(new_height_float)) # 使用 round() 四捨五入後轉換為 int

        # 限制縮放倍率，並處理邊界情況
        min_width = self.original_pixmap.width() / 10
        max_width = self.original_pixmap.width() * 10

        min_height = self.original_pixmap.height() / 10
        max_height = self.original_pixmap.height() * 10

        new_width = max(int(min_width), min(new_width, int(max_width))) # 限制寬度
        new_height = max(int(min_height), min(new_height, int(max_height))) # 限制高度

        if new_width <= 0 or new_height <= 0: # 避免寬高為負數或0的情況
            return

        self.scaled_pixmap = self.original_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label.setPixmap(self.scaled_pixmap)

        mouse_pos = self.scroll_area.mapFromGlobal(event.globalPos())

        # 處理圖片比 scroll area 小的情況
        if self.scaled_pixmap.width() <= self.scroll_area.viewport().width():
            delta_x = 0
        else:
            delta_x = (new_width - old_size.width()) * (mouse_pos.x() / old_size.width())

        if self.scaled_pixmap.height() <= self.scroll_area.viewport().height():
            delta_y = 0
        else:
            delta_y = (new_height - old_size.height()) * (mouse_pos.y() / old_size.height())

        self.scroll_area.horizontalScrollBar().setValue(int(self.scroll_area.horizontalScrollBar().value() + delta_x))
        self.scroll_area.verticalScrollBar().setValue(int(self.scroll_area.verticalScrollBar().value() + delta_y))

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("影像處理")

        # 輸入影像顯示
        self.input_label = QLabel(self)
        self.input_label.setFixedSize(400, 300)
        self.input_label.setStyleSheet("border: 1px solid black;")
        self.input_label.mousePressEvent = lambda event: self.showEnlargedImage(self.input_label)

        # 輸出影像顯示
        self.output_label = QLabel(self)
        self.output_label.setFixedSize(400, 300)
        self.output_label.setStyleSheet("border: 1px solid black;")
        self.output_label.mousePressEvent = lambda event: self.showEnlargedImage(self.output_label)

        # 選擇檔案按鈕
        self.open_button = QPushButton("選擇影像", self)
        self.open_button.clicked.connect(self.openImage)

        # 處理影像按鈕
        self.process_button = QPushButton("灰階轉換", self)
        self.process_button.clicked.connect(self.processImage)
        self.process_button.setEnabled(False)

        # 佈局
        hbox = QHBoxLayout()
        hbox.addWidget(self.input_label)
        hbox.addWidget(self.output_label)

        vbox = QVBoxLayout()
        vbox.addWidget(self.open_button)
        vbox.addWidget(self.process_button)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.show()

    def openImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, "開啟影像", "", "Images (*.png *.jpg *.bmp)")
        if filename:
            self.image_path = filename
            self.img = cv2.imread(filename)
            self.displayImage(self.img, self.input_label)
            self.process_button.setEnabled(True)

    def processImage(self):
        if hasattr(self, 'img'):
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.displayImage(gray_img, self.output_label)

    def displayImage(self, img, label):
        if len(img.shape) == 3:
            h, w, c = img.shape
            bytes_per_line = 3 * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else:
            h, w = img.shape
            bytes_per_line = w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        label.pixmap_original = QPixmap(pixmap)

    def showEnlargedImage(self, label):
        if hasattr(label, 'pixmap_original'):
            viewer = ImageViewer(label.pixmap_original)
            viewer.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    sys.exit(app.exec_())