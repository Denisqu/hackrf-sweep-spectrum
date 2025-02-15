import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSlot, QThread

from utils.logger import logger

class SpectrogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)
        self.data = []
        self.frequencies = None
        self.time_points = []
        self.colorbar = None

    @pyqtSlot(object)
    def handle_data_ready(self, buffer):
        logger.info(f'started handling data_ready_signal from thread {QThread.currentThread()}')
        all_frequencies = []
        all_dbs = []
        try:
            for entry in buffer:
                date_time_str, hz_low, hz_high, hz_bin_width, dbs_val = entry
                num_bins = len(dbs_val)
                step = hz_bin_width
                frequencies = [hz_low + i * step for i in range(num_bins)]
                all_frequencies.extend(frequencies)
                all_dbs.extend(dbs_val)
            
            all_frequencies = np.array(all_frequencies)
            all_dbs = np.array(all_dbs)

            if self.frequencies is None:
                self.frequencies = all_frequencies
                self.data = [all_dbs]
            else:
                if np.array_equal(self.frequencies, all_frequencies):
                    self.data.append(all_dbs)
                else:
                    print("Frequency bins changed, resetting data.")
                    self.frequencies = all_frequencies
                    self.data = [all_dbs]

            max_time_points = 10
            if len(self.data) > max_time_points:
                self.data = self.data[-max_time_points:]
            
            self.update_plot()
        except Exception as e:
            logger.error(f"Error processing data: {e}")
        logger.info(f'stopped handling data_ready_signal from thread {QThread.currentThread()}')


    def update_plot(self):
        logger.info(f'started handling update_plot from thread {QThread.currentThread()}')
        self.axes.clear()
        if not self.data:
            return
        
        data_array = np.array(self.data).T
        
        times = np.arange(data_array.shape[1])
        freqs_mhz = self.frequencies / 1e6
        
        # Убедимся, что частоты монотонно возрастают
        if not np.all(np.diff(freqs_mhz) > 0):
            freqs_mhz = np.sort(freqs_mhz)  # Сортируем частоты, если они не упорядочены
        
        # Создаем явные границы для ячеек
        time_edges = np.arange(data_array.shape[1] + 1)  # Границы по времени
        freq_edges = np.arange(len(freqs_mhz) + 1)  # Границы по частоте
        
        # Преобразуем частоты в границы
        freq_edges = np.interp(freq_edges, np.arange(len(freqs_mhz)), freqs_mhz)
        
        # Создаем сетку для границ
        X, Y = np.meshgrid(time_edges, freq_edges)
        
        # Рисуем спектрограмму с явными границами
        mesh = self.axes.pcolormesh(X, Y, data_array, shading='auto', cmap='inferno')
        
        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(mesh, ax=self.axes)
        else:
            self.colorbar.update_normal(mesh)
        
        self.axes.set_xlabel('Time (samples)')
        self.axes.set_ylabel('Frequency (MHz)')
        self.canvas.draw()
        logger.info(f'stopped handling update_plot from thread {QThread.currentThread()}')

class MainWindow(QMainWindow):
    def __init__(self, sweeper):
        super().__init__()
        self.spectrogram_widget = SpectrogramWidget()
        self.setCentralWidget(self.spectrogram_widget)
        sweeper.data_ready_signal.connect(self.spectrogram_widget.handle_data_ready)

    def closeEvent(self, event):
        event.accept()