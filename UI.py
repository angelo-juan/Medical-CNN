import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("影像處理")

        # 輸入影像顯示
        self.input_label = QLabel(self)
        self.input_label.setFixedSize(400, 300) # 設定固定大小
        self.input_label.setStyleSheet("border: 1px solid black;") # 加上邊框

        # 輸出影像顯示
        self.output_label = QLabel(self)
        self.output_label.setFixedSize(400, 300) # 設定固定大小
        self.output_label.setStyleSheet("border: 1px solid black;") # 加上邊框

        # 選擇檔案按鈕
        self.open_button = QPushButton("選擇影像", self)
        self.open_button.clicked.connect(self.openImage)

        # 處理影像按鈕
        self.process_button = QPushButton("灰階轉換", self)
        self.process_button.clicked.connect(self.processImage)
        self.process_button.setEnabled(False) # 初始狀態停用

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
            self.image_path = filename # 儲存影像路徑
            self.img = cv2.imread(filename)
            self.displayImage(self.img, self.input_label)
            self.process_button.setEnabled(True) # 開啟處理按鈕

    def processImage(self):
        if hasattr(self, 'img'): # 檢查是否有讀取影像
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.displayImage(gray_img, self.output_label)

    def displayImage(self, img, label):
        # 將 OpenCV 影像轉換為 PyQt 影像
        if len(img.shape) == 3: # 彩色圖片
            h, w, c = img.shape
            bytes_per_line = 3 * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else: # 灰階圖片
            h, w = img.shape
            bytes_per_line = w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(label.width(), label.height()) # 縮放圖片以符合標籤大小
        label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    sys.exit(app.exec_())