from utils.logger import logger
from gui.main_window import MainWindow
from sweeper.hackrf_sweeper import HackRFSweeper
import matplotlib.pyplot as plt
import numpy as np

import cProfile
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QMessageBox


if __name__ == '__main__':
    logger.info(f'Starting application in thread {int(QThread.currentThread().currentThreadId())}')
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    result = app.exec_()
    sys.exit(result)

