from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import QThread, pyqtSignal
from gui.drawing_area import DrawingArea
from gui.buttons_panel import ButtonsPanel
from data.generated_data import generate_dataset

class TrainThread(QThread):
    progress = pyqtSignal(str)  # сигнал для передачи логов из потока

    def __init__(self, nn_controller, epochs=5, batch_size=16, lr=0.001):
        super().__init__()
        self.nn_controller = nn_controller
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(self):
        for epoch in range(self.epochs):
            X, y_shape, y_color = generate_dataset(num_samples=1000, img_size=128)
            losses = []
            num_batches = len(X) // self.batch_size
            self.progress.emit(f"Epoch {epoch+1}/{self.epochs}")

            for i in range(0, len(X), self.batch_size):
                x_batch = X[i:i+self.batch_size]
                y_shape_batch = y_shape[i:i+self.batch_size]
                y_color_batch = y_color[i:i+self.batch_size]

                loss = self.nn_controller.model.train_batch(x_batch, y_shape_batch, y_color_batch, learning_rate=self.lr)
                losses.append(loss)

                if (i // self.batch_size) % 10 == 0 or i + self.batch_size >= len(X):
                    avg_loss = sum(losses) / len(losses)
                    self.progress.emit(f"  Batch {(i // self.batch_size)+1}/{num_batches}, Avg Loss: {avg_loss:.4f}")

            self.progress.emit(f"Epoch {epoch+1} finished, Avg Loss: {sum(losses)/len(losses):.4f}\n")
            self.nn_controller.save_model()


class MainWindow(QMainWindow):
    def __init__(self, nn_controller):
        super().__init__()
        self.nn_controller = nn_controller  # объект для обучения и распознавания

        self.setWindowTitle("Shape Detector")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.drawing_area = DrawingArea()
        self.buttons_panel = ButtonsPanel()

        self.status_label = QLabel("Рисуйте фигуру и выбирайте действие")

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.drawing_area)
        left_layout.addWidget(self.status_label)

        layout.addLayout(left_layout)
        layout.addWidget(self.buttons_panel)

        central_widget.setLayout(layout)

        # Связь кнопок с методами
        self.buttons_panel.clear_button.clicked.connect(self.drawing_area.clear)
        self.buttons_panel.recognize_button.clicked.connect(self.recognize_shape)
        self.buttons_panel.train_button.clicked.connect(self.train_network)
        self.buttons_panel.color_buttons.signal_color_changed.connect(self.drawing_area.set_color)

    def recognize_shape(self):
        img = self.drawing_area.get_image()
        result = self.nn_controller.predict(img)
        text = f"Фигура: {result['shape']}, Цвет: {result['color']}, Похожесть: {result['confidence']:.2f}%"
        self.status_label.setText(text)

    def train_network(self):
        self.train_thread = TrainThread(self.nn_controller, epochs=1)
        self.train_thread.progress.connect(self.on_train_progress)
        self.train_thread.start()

    def on_train_progress(self, message):
        print(message)
        # Здесь можно обновлять UI, например QTextEdit, QLabel, итд


    def on_training_finished(self):
        self.status_label.setText("Обучение завершено")
        self.buttons_panel.train_button.setEnabled(True)
