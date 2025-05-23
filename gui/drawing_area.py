from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QImage, QColor
from PyQt6.QtCore import Qt, QPoint
import numpy as np

class DrawingArea(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)

        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = QColor('black')
        self.pen_width = 3

    def set_color(self, color_name):
        self.pen_color = QColor(color_name)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            pen = QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def get_image(self):
        ptr = self.image.bits()
        ptr.setsize(self.image.bytesPerLine() * self.image.height())
        arr = np.array(ptr).reshape(self.image.height(), self.image.width(), 4)
        return arr[:, :, :3].copy()

