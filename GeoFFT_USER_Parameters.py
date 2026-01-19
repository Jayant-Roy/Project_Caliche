# 2D FFT Geoscience Filters (User-Configurable)

import sys
import numpy as np
import cv2
import rasterio

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QFormLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class FFTGeoscience(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D FFT Geoscience Filters")
        self.setGeometry(100, 100, 1550, 720)

        self.image = None
        self.last_output = None
        self.src_profile = None

        self.last_vmin = None
        self.last_vmax = None

        self.init_ui()

    def init_ui(self):

        main = QWidget()
        layout = QVBoxLayout()

        title = QLabel("2D FFT Continuous Geoscience Filters")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:18px; font-weight:bold;")

        img_layout = QHBoxLayout()

        self.input_label = QLabel("Input")
        self.output_label = QLabel("Output")

        for lbl in (self.input_label, self.output_label):
            lbl.setFixedSize(520, 420)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border:1px solid gray;")

        # Colorbar
        self.cbar_label = QLabel()
        self.cbar_label.setFixedSize(30, 420)

        self.cbar_text = QLabel()
        self.cbar_text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.cbar_text.setStyleSheet("font-size:10px;")

        cbar_layout = QHBoxLayout()
        cbar_layout.addWidget(self.cbar_label)
        cbar_layout.addWidget(self.cbar_text)

        # Controls
        control_box = QGroupBox("Controls")
        ctrl = QVBoxLayout()

        self.load_btn = QPushButton("Load Raster / GeoTIFF")
        self.load_btn.clicked.connect(self.load_image)

        self.filter_box = QComboBox()
        self.filter_box.addItems([
            "Low Pass Filter",
            "High Pass Filter",
            "Band Pass Filter",
            "Gaussian High Pass (Wavelength)",
            "Horizontal Derivative",
            "Vertical Derivative",
            "Reduction To Pole",
            "Tilt Derivative"
        ])
        self.filter_box.currentTextChanged.connect(self.update_param_visibility)

        self.param_box = QGroupBox("Filter Parameters")
        self.param_layout = QFormLayout()

        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.001, 0.5)
        self.cutoff.setValue(0.15)

        self.low_cut = QDoubleSpinBox()
        self.low_cut.setRange(0.001, 0.5)
        self.low_cut.setValue(0.05)

        self.high_cut = QDoubleSpinBox()
        self.high_cut.setRange(0.001, 0.5)
        self.high_cut.setValue(0.25)

        self.wavelength = QDoubleSpinBox()
        self.wavelength.setRange(1, 100000)
        self.wavelength.setValue(5000)

        self.deriv_order = QDoubleSpinBox()
        self.deriv_order.setRange(1, 2)
        self.deriv_order.setValue(1)

        self.z_order = QDoubleSpinBox()
        self.z_order.setRange(1, 2)
        self.z_order.setValue(1)

        self.inc = QDoubleSpinBox()
        self.inc.setRange(-90, 90)
        self.inc.setValue(90)

        self.dec = QDoubleSpinBox()
        self.dec.setRange(0, 360)
        self.dec.setValue(0)

        self.param_layout.addRow("Cutoff Frequency", self.cutoff)
        self.param_layout.addRow("Low Cutoff", self.low_cut)
        self.param_layout.addRow("High Cutoff", self.high_cut)
        self.param_layout.addRow("Gaussian Cutoff Wavelength (m)", self.wavelength)
        self.param_layout.addRow("Derivative Order", self.deriv_order)
        self.param_layout.addRow("Z-Derivative Order", self.z_order)
        self.param_layout.addRow("Inclination (°)", self.inc)
        self.param_layout.addRow("Declination (°)", self.dec)

        self.param_box.setLayout(self.param_layout)

        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.clicked.connect(self.apply_filter)
        self.apply_btn.setEnabled(False)

        self.save_btn = QPushButton("Save Output")
        self.save_btn.clicked.connect(self.save_output)
        self.save_btn.setEnabled(False)

        ctrl.addWidget(self.load_btn)
        ctrl.addWidget(self.filter_box)
        ctrl.addWidget(self.param_box)
        ctrl.addWidget(self.apply_btn)
        ctrl.addWidget(self.save_btn)
        ctrl.addStretch()

        control_box.setLayout(ctrl)

        img_layout.addWidget(self.input_label)
        img_layout.addWidget(control_box)
        img_layout.addWidget(self.output_label)
        img_layout.addLayout(cbar_layout)

        layout.addWidget(title)
        layout.addLayout(img_layout)
        main.setLayout(layout)
        self.setCentralWidget(main)

        self.update_param_visibility()

    def update_param_visibility(self):
        f = self.filter_box.currentText()

        for i in range(self.param_layout.rowCount()):
            self.param_layout.itemAt(i, QFormLayout.LabelRole).widget().setVisible(False)
            self.param_layout.itemAt(i, QFormLayout.FieldRole).widget().setVisible(False)

        def show(row):
            self.param_layout.itemAt(row, QFormLayout.LabelRole).widget().setVisible(True)
            self.param_layout.itemAt(row, QFormLayout.FieldRole).widget().setVisible(True)

        #f = self.filter_box.currentText()

        if f in ["Low Pass Filter", "High Pass Filter"]:
            show(0)
        elif f == "Band Pass Filter":
            show(1); show(2)
        elif f == "Gaussian High Pass (Wavelength)":
            show(3)
        elif f in ["Horizontal Derivative", "Vertical Derivative"]:
            show(4)
        elif f == "Reduction To Pole":
            show(6); show(7)
        elif f == "Tilt Derivative":
            show(5)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Raster", "", "Raster (*.tif *.tiff)"
        )
        if not path:
            return

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            self.src_profile = src.profile

        data = np.nan_to_num(data)
        data -= np.mean(data)

        self.image = data
        self.display(self.prepare_display(data), self.input_label)
        self.update_colorbar()
        self.apply_btn.setEnabled(True)

    def apply_filter(self):
        img = self.image.copy()
        rows, cols = img.shape

        img *= np.outer(np.hanning(rows), np.hanning(cols))
        F = np.fft.fftshift(np.fft.fft2(img))

        kx = np.fft.fftfreq(cols)
        ky = np.fft.fftfreq(rows)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        kmax = np.max(K)

        f = self.filter_box.currentText()

        if f == "Low Pass Filter":
            H = K <= self.cutoff.value() * kmax

        elif f == "High Pass Filter":
            H = K >= self.cutoff.value() * kmax

        elif f == "Band Pass Filter":
            H = (K >= self.low_cut.value() * kmax) & \
                (K <= self.high_cut.value() * kmax)

        elif f == "Gaussian High Pass (Wavelength)":
            kc = 1.0 / self.wavelength.value()
            H = np.exp(-(K**2) / (2 * kc**2))

        elif f == "Horizontal Derivative":
            n = int(self.deriv_order.value())
            H = (1j * 2 * np.pi * KX) ** n

        elif f == "Vertical Derivative":
            n = int(self.deriv_order.value())
            H = (2 * np.pi * K) ** n

        elif f == "Reduction To Pole":
            I = np.deg2rad(self.inc.value())
            D = np.deg2rad(self.dec.value())
            denom = np.sin(I)**2 + (np.cos(I)**2) * (KX*np.cos(D) + KY*np.sin(D))**2
            denom[denom == 0] = 1e-10
            H = (K**2) / denom

        elif f == "Tilt Derivative":
            n = int(self.z_order.value())

            Fx = (1j * 2 * np.pi * KX) * F
            Fy = (1j * 2 * np.pi * KY) * F
            Fz = (2 * np.pi * K) ** n * F

            dx = np.real(np.fft.ifft2(np.fft.ifftshift(Fx)))
            dy = np.real(np.fft.ifft2(np.fft.ifftshift(Fy)))
            dz = np.real(np.fft.ifft2(np.fft.ifftshift(Fz)))

            out = np.arctan(dz / (np.sqrt(dx**2 + dy**2) + 1e-10))
            self.finalize(out)
            return

        out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
        self.finalize(out)

    def finalize(self, out):
        self.last_output = out
        self.display(self.prepare_display(out), self.output_label)
        self.update_colorbar()
        self.save_btn.setEnabled(True)

    def prepare_display(self, img):
        vmin, vmax = np.percentile(img, [2, 98])
        self.last_vmin, self.last_vmax = vmin, vmax
        img = np.clip(img, vmin, vmax)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)

    def update_colorbar(self):
        grad = np.linspace(255, 0, 256).astype(np.uint8).reshape(256, 1)
        cbar = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
        cbar = cv2.resize(cbar, (20, 420))
        qimg = QImage(
            cbar.data,
            cbar.shape[1],
            cbar.shape[0],
            cbar.strides[0],
            QImage.Format_BGR888
        )
        self.cbar_label.setPixmap(QPixmap.fromImage(qimg))
        self.cbar_text.setText(
            f"{self.last_vmax:.2f}\n\n\n0\n\n\n{self.last_vmin:.2f}"
            )

    def save_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save", "filtered.tif", "GeoTIFF (*.tif)"
        )
        if not path:
            return

        profile = self.src_profile.copy()
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(self.last_output.astype(np.float32), 1)

    def display(self, img, label):
        qimg = QImage(
            img.data,
            img.shape[1],
            img.shape[0],
            img.strides[0],
            QImage.Format_BGR888
        )
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(), Qt.KeepAspectRatio))


#--------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FFTGeoscience()
    win.show()
    sys.exit(app.exec_())
