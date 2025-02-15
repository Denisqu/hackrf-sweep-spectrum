import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, QObject, QTimer

from utils.logger import logger
from sweeper.hackrf_sweeper import HackRFSweeper

class SpectrogramWorker(QObject):
    plot_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # X, Y, data, freq_edges
    process_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.current_frequencies = None
        self.history_data = []
        self.max_history = 500

        self.process_requested.connect(self.process_data)

    @pyqtSlot(object)
    def process_data(self, buffer):
        try:
            # Обработка и подготовка данных
            all_frequencies, all_dbs = self._process_buffer(buffer)
            self._update_history(all_frequencies, all_dbs)
            
            # Подготовка данных для отрисовки
            if self.history_data:
                # Оставляем данные в виде (время, частоты)
                data_array = np.array(self.history_data)  # shape: (time, freq)
                # Переворачиваем массив по оси времени, чтобы новые данные были сверху
                data_array = data_array[::-1, :]
                freqs_mhz = self.current_frequencies / 1e6
                
                # Сортируем частоты (ось 1) по возрастанию
                sorted_indices = np.argsort(freqs_mhz)
                freqs_mhz = freqs_mhz[sorted_indices]
                data_array = data_array[:, sorted_indices]
                
                # Расчет сетки: X - частоты, Y - время
                time_edges = np.arange(data_array.shape[0] + 1)
                freq_edges = np.interp(
                    np.arange(len(freqs_mhz) + 1),
                    np.arange(len(freqs_mhz)),
                    freqs_mhz
                )
                
                X, Y = np.meshgrid(freq_edges, time_edges)
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

        self.worker_thread = QThread()
        self.worker_thread.setObjectName("SpectrogramWorkerThread")
        self.worker = SpectrogramWorker()
        self.worker.moveToThread(self.worker_thread)

        self.worker.plot_ready.connect(self.update_plot)

        self.worker_thread.start()

        self.timer = QTimer(self)
        self.timer.setInterval(33)  # 33 мс ≈ 30 кадров в секунду
        self.timer.timeout.connect(self._on_timer_timeout)
        self.timer.start()

        # Буфер для накопления данных
        self.plot_data_buffer = []

    @pyqtSlot(object)
    def handle_data_ready(self, buffer):
        self.worker.process_requested.emit(buffer)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def update_plot(self, X, Y, data, freq_edges):
        self.plot_data_buffer.append((X, Y, data, freq_edges))

    def _on_timer_timeout(self):
        if self.plot_data_buffer:
            X, Y, data, freq_edges = self.plot_data_buffer[-1]
            self.plot_data_buffer.clear()  # Очищаем буфер

            self.axes.clear()
            mesh = self.axes.pcolormesh(X, Y, data, shading='auto', cmap='inferno', rasterized=True, vmin=-70, vmax=0)
            if self.colorbar is None:
                self.colorbar = self.figure.colorbar(mesh, ax=self.axes)
            else:
                self.colorbar.update_normal(mesh)

            self.axes.set_xlabel('Frequency (MHz)')
            self.axes.set_ylabel('Time (samples)')
            # Инвертируем ось Y, чтобы время шло сверху вниз
            self.axes.invert_yaxis()

            self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sweeper = HackRFSweeper()
        self.sweeper.sweeper_stopped_signal.connect(lambda: QApplication.instance().quit())
        self.spectrogram_widget = SpectrogramWidget()
        self.setCentralWidget(self.spectrogram_widget)
        self.sweeper.data_ready_signal.connect(self.spectrogram_widget.worker.process_requested)
        self.sweeper.start()

    def closeEvent(self, event):
        self.sweeper.stop()
        event.accept()
