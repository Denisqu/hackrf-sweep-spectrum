import sys
import os
import datetime
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QCheckBox, QPushButton, QLabel, QFileDialog, QSizePolicy, QSlider, QLineEdit
)
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, QObject, QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils.logger import logger
from sweeper.hackrf_sweeper import HackRFSweeper


class SpectrogramWorker(QObject):
    plot_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # X, Y, data, freq_edges
    process_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.current_frequencies = None
        self.history_data = []
        self.max_history = 250

        # Параметры для добавления шума по всему спектру
        self.add_noise = False
        self.noise_db = -10  # уровень шума (используется как параметр для генерации Гауссова шума)

        # Параметры фильтрации
        self.use_filters = False
        self.filter_specs = []  # список кортежей (min_freq, max_freq) в МГц

        self.process_requested.connect(self.process_data)

    @pyqtSlot(object)
    def process_data(self, buffer):
        try:
            # Обработка и подготовка данных
            all_frequencies, all_dbs = self._process_buffer(buffer)
            self._update_history(all_frequencies, dbs=all_dbs)

            # Подготовка данных для отрисовки
            if self.history_data:
                data_array = np.array(self.history_data)  # shape: (time, freq)
                data_array = data_array[::-1, :]  # новые данные сверху
                # Переводим частоты в МГц
                freqs_mhz = self.current_frequencies / 1e6

                # Применяем фильтрацию по заданным диапазонам
                if self.use_filters and self.filter_specs:
                    for (min_freq, max_freq) in self.filter_specs:
                        # Находим индексы частот, попадающие в диапазон (в МГц)
                        indices = np.where((freqs_mhz >= min_freq) & (freqs_mhz <= max_freq))[0]
                        if indices.size > 0:
                            # Генерируем гауссов шум с центром -67.5 дБ и std=1, затем обрезаем значения в диапазоне [-70, -65]
                            filtered_noise = np.random.normal(-68, 3.0, size=(data_array.shape[0], len(indices)))
                            data_array[:, indices] = filtered_noise

                # Применяем глобальный гауссов шум, если он включён
                if self.add_noise:
                    # Здесь noise_db интерпретируется как среднее значение шума, а стандартное отклонение фиксировано (например, 5)
                    noise = np.random.normal(self.noise_db, 5, size=data_array.shape)
                    data_array = data_array + noise

                # Сортировка по возрастанию частот
                sorted_indices = np.argsort(freqs_mhz)
                freqs_mhz = freqs_mhz[sorted_indices]
                data_array = data_array[:, sorted_indices]

                # Расчёт сетки: X - частоты, Y - время
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

    @pyqtSlot(bool, float)
    def set_noise_params(self, add_noise, noise_db):
        self.add_noise = add_noise
        self.noise_db = noise_db
        logger.info(f"Параметры шума обновлены: add_noise={add_noise}, noise_db={noise_db} дБ")

    @pyqtSlot(bool, str)
    def set_filter_params(self, use_filters, filter_str):
        """
        Обновление параметров фильтрации.
        :param use_filters: флаг использования фильтров
        :param filter_str: строка с диапазонами в формате "[min_freq0:max_freq0],[min_freq1:max_freq1]..."
                           Значения частот указываются в МГц.
        """
        self.use_filters = use_filters
        self.filter_specs = []
        filter_str = filter_str.strip()
        if filter_str:
            try:
                # Если строка начинается и заканчивается квадратными скобками, убираем их
                if filter_str.startswith("[") and filter_str.endswith("]"):
                    filter_str = filter_str[1:-1]
                # Разбиваем по разделителю "],["
                filters = filter_str.split("],[")
                for f in filters:
                    parts = f.split(":")
                    if len(parts) == 2:
                        min_freq = float(parts[0])
                        max_freq = float(parts[1])
                        self.filter_specs.append((min_freq, max_freq))
                logger.info(f"Обновлены диапазоны фильтрации: {self.filter_specs}")
            except Exception as e:
                logger.error(f"Ошибка при парсинге фильтров: {e}")
        else:
            logger.info("Диапазоны фильтрации не заданы (пустая строка).")


class SpectrogramWidget(QWidget):
    def __init__(self, controls_widget, parent=None):
        """
        :param controls_widget: Экземпляр ControlsWidget для доступа к состоянию записи и выбранной папке
        """
        super().__init__(parent)
        self.controls_widget = controls_widget

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)
        self.colorbar = None
        self.plot_data_buffer = []

        self.worker_thread = QThread()
        self.worker_thread.setObjectName("SpectrogramWorkerThread")
        self.worker = SpectrogramWorker()
        self.worker.plot_ready.connect(self.update_plot)

        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        self.timer = QTimer(self)
        self.timer.setInterval(500)  # обновление каждые 500 мс
        self.timer.timeout.connect(self._on_timer_timeout)
        self.timer.start()

    @pyqtSlot(object)
    def handle_data_ready(self, buffer):
        self.worker.process_requested.emit(buffer)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def update_plot(self, X, Y, data, freq_edges):
        self.plot_data_buffer.append((X, Y, data, freq_edges))

    def _on_timer_timeout(self):
        if self.plot_data_buffer:
            X, Y, data, freq_edges = self.plot_data_buffer[-1]
            self.plot_data_buffer.clear()  # очищаем буфер
            self.axes.clear()
            mesh = self.axes.pcolormesh(X, Y, data, shading='auto', cmap='inferno',
                                        rasterized=True, vmin=-70, vmax=0)
            if self.colorbar is None:
                self.colorbar = self.figure.colorbar(mesh, ax=self.axes)
            else:
                self.colorbar.update_normal(mesh)
            self.axes.set_xlabel('Frequency (MHz)')
            self.axes.set_ylabel('Time (samples)')
            self.axes.invert_yaxis()  # инвертируем ось Y
            self.canvas.draw()

            # Если запись включена, сохраняем только спектр (без осей и легенды)
            if self.controls_widget.recording and self.controls_widget.record_folder:
                temp_fig = Figure(figsize=self.figure.get_size_inches(), dpi=self.figure.get_dpi())
                temp_ax = temp_fig.add_axes([0, 0, 1, 1])
                temp_ax.pcolormesh(X, Y, data, shading='auto', cmap='inferno',
                                   rasterized=True, vmin=-70, vmax=0)
                temp_ax.set_axis_off()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"mesh_{timestamp}.png"
                filepath = os.path.join(self.controls_widget.record_folder, filename)
                try:
                    temp_fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
                    logger.info(f"Сохранён mesh в: {filepath}")
                    # Увеличиваем счётчик сохранённых спектрограмм и обновляем метку
                    self.controls_widget.record_count += 1
                    self.controls_widget.recording_count_label.setText(
                        f"Количество записей: {self.controls_widget.record_count}"
                    )
                except Exception as e:
                    logger.error(f"Ошибка сохранения mesh: {e}")
                temp_fig.clf()


class ControlsWidget(QWidget):
    # Сигналы для передачи параметров шума и фильтрации
    noise_params_changed = pyqtSignal(bool, float)
    filter_params_changed = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Создание основных элементов управления
        self.hackrf_checkbox = QCheckBox("Использовать HackRF")
        self.recorded_signal_checkbox = QCheckBox("Использовать записанный сигнал")
        self.hackrf_checkbox.setChecked(True)

        self.choose_recorded_folder_button = QPushButton("Выбрать папку с записанным сигналом")
        self.choose_record_folder_button = QPushButton("Выбрать папку для записи сигнала")
        self.recording_count_label = QLabel("Количество записей: 0")
        self.toggle_recording_button = QPushButton("Начать запись")

        # Новые элементы для гауссового шума
        self.noise_checkbox = QCheckBox("Добавить гауссов шум")
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(50)
        self.noise_slider.setValue(10)  # значение по умолчанию: 10 дБ
        self.noise_value_label = QLabel("10 дБ")

        # Новые элементы для фильтрации
        self.filter_checkbox = QCheckBox("Использовать фильтры")
        self.filter_lineedit = QLineEdit()
        self.filter_lineedit.setPlaceholderText("[min_freq0:max_freq0],[min_freq1:max_freq1]")

        # Локальные переменные для логики записи
        self.recording = False
        self.record_count = 0
        self.record_folder = None

        # Компоновка элементов управления в layout
        layout = QHBoxLayout()
        layout.addWidget(self.hackrf_checkbox)
        layout.addWidget(self.recorded_signal_checkbox)
        layout.addWidget(self.choose_recorded_folder_button)
        layout.addWidget(self.choose_record_folder_button)
        layout.addWidget(self.recording_count_label)
        layout.addWidget(self.toggle_recording_button)
        # Элементы для шума
        layout.addWidget(self.noise_checkbox)
        layout.addWidget(self.noise_slider)
        layout.addWidget(self.noise_value_label)
        # Элементы для фильтров
        layout.addWidget(self.filter_checkbox)
        layout.addWidget(self.filter_lineedit)
        self.setLayout(layout)

        # Подключение сигналов для исключительного выбора основных чекбоксов
        self.hackrf_checkbox.toggled.connect(self.on_hackrf_toggled)
        self.recorded_signal_checkbox.toggled.connect(self.on_recorded_toggled)

        # Подключение сигналов для кнопок
        self.choose_recorded_folder_button.clicked.connect(self.select_recorded_signal_folder)
        self.choose_record_folder_button.clicked.connect(self.select_record_folder)
        self.toggle_recording_button.clicked.connect(self.toggle_recording)

        # Подключение сигналов для управления шумом
        self.noise_checkbox.toggled.connect(self.on_noise_params_changed)
        self.noise_slider.valueChanged.connect(self.on_noise_slider_changed)

        # Подключение сигналов для фильтров
        self.filter_checkbox.toggled.connect(self.on_filter_params_changed)
        self.filter_lineedit.textChanged.connect(self.on_filter_params_changed)

    def on_hackrf_toggled(self, checked):
        if checked and self.recorded_signal_checkbox.isChecked():
            self.recorded_signal_checkbox.blockSignals(True)
            self.recorded_signal_checkbox.setChecked(False)
            self.recorded_signal_checkbox.blockSignals(False)

    def on_recorded_toggled(self, checked):
        if checked and self.hackrf_checkbox.isChecked():
            self.hackrf_checkbox.blockSignals(True)
            self.hackrf_checkbox.setChecked(False)
            self.hackrf_checkbox.blockSignals(False)

    def select_recorded_signal_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбрать папку с записанным сигналом")
        if folder:
            logger.info(f"Папка с записанным сигналом: {folder}")

    def select_record_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбрать папку для записи сигнала")
        if folder:
            self.record_folder = folder
            logger.info(f"Папка для записи сигнала: {folder}")

    def toggle_recording(self):
        if self.recording:
            self.recording = False
            self.toggle_recording_button.setText("Начать запись")
            logger.info("Запись остановлена")
        else:
            self.record_count = 0
            self.recording_count_label.setText("Количество записей: 0")
            self.recording = True
            self.toggle_recording_button.setText("Остановить запись")
            logger.info("Запись запущена")

    def on_noise_slider_changed(self, value):
        self.noise_value_label.setText(f"{value} дБ")
        self.emit_noise_params_changed()

    def on_noise_params_changed(self, checked=None):
        # Если аргумент не передан (например, при изменении слайдера), берём текущее состояние чекбокса
        self.emit_noise_params_changed()

    def emit_noise_params_changed(self):
        self.noise_params_changed.emit(self.noise_checkbox.isChecked(), float(self.noise_slider.value()))

    def on_filter_params_changed(self, checked=None):
        self.emit_filter_params_changed()

    def emit_filter_params_changed(self):
        self.filter_params_changed.emit(self.filter_checkbox.isChecked(), self.filter_lineedit.text())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.controls_widget = ControlsWidget()
        self.spectrogram_widget = SpectrogramWidget(self.controls_widget)
        self.spectrogram_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.controls_widget, 0)
        main_layout.addWidget(self.spectrogram_widget, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.sweeper = HackRFSweeper()
        self.sweeper.sweeper_stopped_signal.connect(lambda: QApplication.instance().quit())
        self.sweeper.data_ready_signal.connect(self.spectrogram_widget.worker.process_requested)
        self.sweeper.start()

        # Соединяем сигналы изменения параметров шума и фильтров с методами обновления в рабочем объекте
        self.controls_widget.noise_params_changed.connect(self.spectrogram_widget.worker.set_noise_params)
        self.controls_widget.filter_params_changed.connect(self.spectrogram_widget.worker.set_filter_params)

    def closeEvent(self, event):
        self.sweeper.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
