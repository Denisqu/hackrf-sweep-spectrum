import sys
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication

from utils.logger import logger
from gui.main_window import MainWindow

if __name__ == '__main__':
    logger.info(f'Starting application in thread {int(QThread.currentThread().currentThreadId())}')
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setFixedSize(1600, 900)
    window.show()
    result = app.exec_()
    sys.exit(result)

