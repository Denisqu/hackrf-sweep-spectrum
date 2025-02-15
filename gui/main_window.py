import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, QObject

from utils.logger import logger

class SpectrogramWorker(QObject):
    # Сигнал для передачи подготовленных данных в основной поток
    plot_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # X, Y, data, freq_edges
    
    # Сигнал для запуска обработки данных (входной сигнал)
    process_requested = pyqtSignal(object)  # buffer

    def __init__(self):
        super().__init__()
        self.current_frequencies = None
        self.history_data = []
        self.max_history = 500

        # Подключаем сигнал process_requested к слоту process_data
        self.process_requested.connect(self.process_data)

    @pyqtSlot(object)
    def process_data(self, buffer):
        logger.info(f'Processing data in thread {int(QThread.currentThread().currentThreadId())}')
        try:
            # Обработка и подготовка данных
            all_frequencies, all_dbs = self._process_buffer(buffer)
            self._update_history(all_frequencies, all_dbs)
            
            # Подготовка данных для отрисовки
            if self.history_data:
                data_array = np.array(self.history_data).T
                freqs_mhz = self.current_frequencies / 1e6
                
                # Сортировка частот
                sorted_indices = np.argsort(freqs_mhz)
                freqs_mhz = freqs_mhz[sorted_indices]
                data_array = data_array[sorted_indices, :]
                
                # Расчет сетки
                time_edges = np.arange(data_array.shape[1] + 1)
                freq_edges = np.interp(
                    np.arange(len(freqs_mhz) + 1),
                    np.arange(len(freqs_mhz)),
                    freqs_mhz
                )
                
                X, Y = np.meshgrid(time_edges, freq_edges)
                self.plot_ready.emit(X, Y, data_array, freqs_mhz)
                
        except Exception as e:
            logger.error(f"Error processing data: {e}")

    def _process_buffer(self, buffer):
        all_frequencies = []
        all_dbs = []
        for entry in buffer:
            date_time_str, hz_low, hz_high, hz_bin_width, dbs_val = entry
            num_bins = len(dbs_val)
            frequencies = [hz_low + i * hz_bin_width for i in range(num_bins)]
            all_frequencies.extend(frequencies)
            all_dbs.extend(dbs_val)
        return np.array(all_frequencies), np.array(all_dbs)

    def _update_history(self, frequencies, dbs):
        if self.current_frequencies is None or not np.array_equal(self.current_frequencies, frequencies):
            self.current_frequencies = frequencies
            self.history_data = [dbs]
        else:
            self.history_data.append(dbs)
            if len(self.history_data) > self.max_history:
                self.history_data = self.history_data[-self.max_history:]

class SpectrogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)
        self.colorbar = None
        
        # Инициализация потока
        self.worker_thread = QThread()
        self.worker_thread.setObjectName = "SpectrogramWorkerThread"
        self.worker = SpectrogramWorker()
        self.worker.moveToThread(self.worker_thread)
        
        # Подключаем сигнал plot_ready к слоту update_plot
        self.worker.plot_ready.connect(self.update_plot)
        
        # Запускаем поток
        self.worker_thread.start()

    @pyqtSlot(object)
    def handle_data_ready(self, buffer):
        self.worker.process_requested.emit(buffer)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def update_plot(self, X, Y, data, freq_edges):
        
        logger.info(f'update_plot in thread {int(QThread.currentThread().currentThreadId())}')
        self.axes.clear()
        # Только отрисовка подготовленных данных
        mesh = self.axes.pcolormesh(X, Y, data, shading='auto', cmap='inferno', rasterized=True, vmin=-60, vmax=0)
        # Обновление цветовой шкалы
        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(mesh, ax=self.axes)
        # else:
        #     self.colorbar.update_normal(mesh)
            
        self.axes.set_xlabel('Time (samples)')
        self.axes.set_ylabel('Frequency (MHz)')
        self.canvas.draw()


    def closeEvent(self, event):
        self.worker_thread.quit()
        self.worker_thread.wait()
        super().closeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, sweeper):
        super().__init__()
        self.spectrogram_widget = SpectrogramWidget()
        self.setCentralWidget(self.spectrogram_widget)
        sweeper.data_ready_signal.connect(self.spectrogram_widget.worker.process_requested)

    def closeEvent(self, event):
        event.accept()