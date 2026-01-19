# Simple 2D FFT Image Filtering Tool (LPF / HPF / BPF)
# PyQt5 + NumPy + OpenCV + Rasterio (GeoTIFF support)

import rasterio
import numpy as np
import sys
import numpy as np
import cv2
import rasterio

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QVBoxLayout, QSlider,
    QRadioButton, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class FFTFilterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D FFT Filters (LPF / HPF / BPF)")
        self.setGeometry(100, 100, 1200, 600)

        self.image = None      # float data (for FFT)
        self.image_disp = None # uint8 (for display)

        self.init_ui()

    def init_ui(self):
        main = QWidget()
        layout = QVBoxLayout()

        title = QLabel("2D FFT Image Filtering (GeoTIFF Supported)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:18px; font-weight:bold;")

        img_layout = QHBoxLayout()

        self.input_label = QLabel("Input Image")
        self.output_label = QLabel("Output Image")

        for lbl in (self.input_label, self.output_label):
            lbl.setFixedSize(500, 400)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border:1px solid gray;")

        # Controls
        control_box = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()

        upload_btn = QPushButton("Upload Image / GeoTIFF")
        upload_btn.clicked.connect(self.load_image)

        self.lpf_radio = QRadioButton("Low-Pass Filter")
        self.hpf_radio = QRadioButton("High-Pass Filter")
        self.bpf_radio = QRadioButton("Band-Pass Filter")
        self.lpf_radio.setChecked(True)

        self.lpf_radio.toggled.connect(self.update_slider_state)
        self.hpf_radio.toggled.connect(self.update_slider_state)
        self.bpf_radio.toggled.connect(self.update_slider_state)

        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(1, 200)
        self.cutoff_slider.setValue(40)
        self.cutoff_slider.valueChanged.connect(self.apply_filter)

        self.low_slider = QSlider(Qt.Horizontal)
        self.low_slider.setRange(1, 200)
        self.low_slider.setValue(20)
        self.low_slider.valueChanged.connect(self.apply_filter)

        self.high_slider = QSlider(Qt.Horizontal)
        self.high_slider.setRange(1, 200)
        self.high_slider.setValue(60)
        self.high_slider.valueChanged.connect(self.apply_filter)

        self.cutoff_slider.setEnabled(True)
        self.low_slider.setEnabled(False)
        self.high_slider.setEnabled(False)

        ctrl_layout.addWidget(upload_btn)
        ctrl_layout.addWidget(self.lpf_radio)
        ctrl_layout.addWidget(self.hpf_radio)
        ctrl_layout.addWidget(self.bpf_radio)

        ctrl_layout.addWidget(QLabel("Cutoff Frequency"))
        ctrl_layout.addWidget(self.cutoff_slider)

        ctrl_layout.addWidget(QLabel("Band-Pass Lower Cutoff"))
        ctrl_layout.addWidget(self.low_slider)
        ctrl_layout.addWidget(QLabel("Band-Pass Upper Cutoff"))
        ctrl_layout.addWidget(self.high_slider)
        ctrl_layout.addStretch()

        control_box.setLayout(ctrl_layout)

        img_layout.addWidget(self.input_label)
        img_layout.addWidget(control_box)
        img_layout.addWidget(self.output_label)

        layout.addWidget(title)
        layout.addLayout(img_layout)
        main.setLayout(layout)
        self.setCentralWidget(main)

    def update_slider_state(self):
        if self.bpf_radio.isChecked():
            self.cutoff_slider.setEnabled(False)
            self.low_slider.setEnabled(True)
            self.high_slider.setEnabled(True)
        else:
            self.cutoff_slider.setEnabled(True)
            self.low_slider.setEnabled(False)
            self.high_slider.setEnabled(False)

        self.apply_filter()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image / GeoTIFF",
            "",
            "Images (*.png *.jpg *.bmp *.tif *.tiff *.grd)"
        )

        if not path:
            return

        # ---- GeoTIFF (scientific raster) ----
        if path.lower().endswith((".tif", ".tiff")):
            try:
                with rasterio.open(path) as src:
                    data = src.read(1).astype(np.float32)
            except Exception as e:
                print("Failed to read GeoTIFF:", e)
                return

        # ---- Normal images ----
        else:
            data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if data is None:
                print("Failed to read image")
                return
            data = data.astype(np.float32)

        self.image = data

        # Normalize only for display
        self.image_disp = cv2.normalize(
            self.image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        self.display(self.image_disp, self.input_label)
        self.apply_filter()

    def apply_filter(self):
        if self.image is None:
            return

        rows, cols = self.image.shape
        crow, ccol = rows // 2, cols // 2

        f = np.fft.fftshift(np.fft.fft2(self.image))
        mask = np.zeros((rows, cols), np.uint8)

        if self.lpf_radio.isChecked():
            r = self.cutoff_slider.value()
            cv2.circle(mask, (ccol, crow), r, 1, -1)

        elif self.hpf_radio.isChecked():
            r = self.cutoff_slider.value()
            mask[:] = 1
            cv2.circle(mask, (ccol, crow), r, 0, -1)

        elif self.bpf_radio.isChecked():
            r1 = self.low_slider.value()
            r2 = self.high_slider.value()
            if r1 > r2:
                r1, r2 = r2, r1
            cv2.circle(mask, (ccol, crow), r2, 1, -1)
            cv2.circle(mask, (ccol, crow), r1, 0, -1)

        filtered = f * mask
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

        img_disp = cv2.normalize(
            img_back, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        self.display(img_disp, self.output_label)

    def display(self, img, label):
        qimg = QImage(
            img.data, img.shape[1], img.shape[0],
            img.strides[0], QImage.Format_Grayscale8
        )
        pix = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(), Qt.KeepAspectRatio
        )
        label.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FFTFilterApp()
    win.show()
    sys.exit(app.exec_())
