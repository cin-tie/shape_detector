from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import pyqtSignal

class ColorButtons(QWidget):
    signal_color_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.colors = ['black', 'red', 'green', 'blue']
        for color in self.colors:
            btn = QPushButton()
            btn.setStyleSheet(f"background-color: {color}")
            btn.setFixedSize(30,30)
            btn.clicked.connect(lambda checked, c=color: self.signal_color_changed.emit(c))
            layout.addWidget(btn)
        self.setLayout(layout)

class ButtonsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.color_buttons = ColorButtons()
        layout.addWidget(self.color_buttons)

        self.clear_button = QPushButton("Очистить")
        self.train_button = QPushButton("Обучить нейросеть")
        self.recognize_button = QPushButton("Распознать фигуру")

        layout.addWidget(self.clear_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.recognize_button)
        layout.addStretch()

        self.setLayout(layout)
