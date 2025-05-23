import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow
from nn.train import NNController

def main():
    app = QApplication(sys.argv)
    nn_controller = NNController()
    window = MainWindow(nn_controller)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
