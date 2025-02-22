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
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from utils.logger import logger
from sweeper.hackrf_sweeper import HackRFSweeper

# Model-related functions from the provided ML code
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.vgg16(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    return model

def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = build_model(num_classes, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

# New InferenceWorker class for ML inference in a separate thread
class InferenceWorker(QObject):
    inference_result = pyqtSignal(list)  # Signal to emit inference probabilities

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the model (assumes the path exists)
        # C:/git-repos/hackrf-ml/best_model_initial_epoch_5.pth
        # C:/git-repos/hackrf-ml/best_model_finetune.pth
        self.model = load_model("C:/git-repos/hackrf-ml/best_model_initial_epoch_5.pth", num_classes=2, device=self.device)
        self.model.eval()
        # Define transformations matching the model's training setup
        self.transform = transforms.Compose([
            transforms.Resize((480, 1550)),  # Match img_height, img_width from ML code
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @pyqtSlot(bytes)
    def perform_inference(self, image_bytes):
        try:
            # Load and preprocess the image from bytes
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            image = self.transform(image)
            image = image.unsqueeze(0).to(self.device)  # Add batch dimension

            # Perform inference
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()  # Convert logits to probabilities

            # Emit the probabilities (list of 2 floats for 'noise' and 'drone')
            self.inference_result.emit(probs.tolist())
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            self.inference_result.emit([])  # Emit empty list on error

class SpectrogramWorker(QObject):
    plot_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # X, Y, data, freq_edges
    process_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.current_frequencies = None
        self.history_data = []
        self.max_history = 250
        self.add_noise = False
        self.noise_db = -10
        self.use_filters = False
        self.filter_specs = []
        self.process_requested.connect(self.process_data)

    @pyqtSlot(object)
    def process_data(self, buffer):
        try:
            all_frequencies, all_dbs = self._process_buffer(buffer)
            self._update_history(all_frequencies, dbs=all_dbs)
            if self.history_data:
                data_array = np.array(self.history_data)[::-1, :]
                freqs_mhz = self.current_frequencies / 1e6
                if self.use_filters and self.filter_specs:
                    for (min_freq, max_freq) in self.filter_specs:
                        indices = np.where((freqs_mhz >= min_freq) & (freqs_mhz <= max_freq))[0]
                        if indices.size > 0:
                            filtered_noise = np.random.normal(-68, 3.0, size=(data_array.shape[0], len(indices)))
                            data_array[:, indices] = filtered_noise
                if self.add_noise:
                    noise = np.random.normal(self.noise_db, 5, size=data_array.shape)
                    data_array = data_array + noise
                sorted_indices = np.argsort(freqs_mhz)
                freqs_mhz = freqs_mhz[sorted_indices]
                data_array = data_array[:, sorted_indices]
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
        self.use_filters = use_filters
        self.filter_specs = []
        filter_str = filter_str.strip()
        if filter_str and use_filters:
            try:
                if filter_str.startswith("[") and filter_str.endswith("]"):
                    filter_str = filter_str[1:-1]
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

class SpectrogramWidget(QWidget):
    inference_requested = pyqtSignal(bytes)  # Signal to request inference

    def __init__(self, controls_widget, parent=None):
        super().__init__(parent)
        self.controls_widget = controls_widget
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)
        self.colorbar = None
        self.plot_data_buffer = []
        self.inference_counter = 0  # Counter for inference triggering

        # Spectrogram worker setup (unchanged)
        self.worker_thread = QThread()
        self.worker_thread.setObjectName("SpectrogramWorkerThread")
        self.worker = SpectrogramWorker()
        self.worker.plot_ready.connect(self.update_plot)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        # Inference worker setup
        self.inference_thread = QThread()
        self.inference_thread.setObjectName("InferenceWorkerThread")
        self.inference_worker = InferenceWorker()
        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_thread.start()

        # Connect inference signals
        self.inference_requested.connect(self.inference_worker.perform_inference)
        self.inference_worker.inference_result.connect(self.controls_widget.update_ml_probabilities)

        # Timer setup (unchanged)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._on_timer_timeout)
        self.timer.start()

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def update_plot(self, X, Y, data, freq_edges):
        self.plot_data_buffer.append((X, Y, data, freq_edges))

    def _on_timer_timeout(self):
        if self.plot_data_buffer:
            X, Y, data, freq_edges = self.plot_data_buffer[-1]
            self.plot_data_buffer.clear()
            self.axes.clear()
            mesh = self.axes.pcolormesh(X, Y, data, shading='auto', cmap='inferno',
                                        rasterized=True, vmin=-70, vmax=0)
            if self.colorbar is None:
                self.colorbar = self.figure.colorbar(mesh, ax=self.axes)
            else:
                self.colorbar.update_normal(mesh)
            self.axes.set_xlabel('Frequency (MHz)')
            self.axes.set_ylabel('Time (samples)')
            self.axes.invert_yaxis()
            self.canvas.draw()

            # Always create temp_fig when there is data
            temp_fig = Figure(figsize=self.figure.get_size_inches(), dpi=self.figure.get_dpi())
            temp_ax = temp_fig.add_axes([0, 0, 1, 1])
            temp_ax.pcolormesh(X, Y, data, shading='auto', cmap='inferno',
                               rasterized=True, vmin=-70, vmax=0)
            temp_ax.set_axis_off()

            # Save the spectrogram if recording is enabled
            if self.controls_widget.recording and self.controls_widget.session_record_folder:
                filename = f"{self.controls_widget.record_count}.png"
                filepath = os.path.join(self.controls_widget.session_record_folder, filename)
                try:
                    temp_fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
                    logger.info(f"Сохранён mesh в: {filepath}")
                    self.controls_widget.record_count += 1
                    self.controls_widget.recording_count_label.setText(
                        f"Количество записей: {self.controls_widget.record_count}"
                    )
                except Exception as e:
                    logger.error(f"Ошибка сохранения mesh: {e}")

            # Trigger inference every fourth time
            self.inference_counter += 1
            if self.inference_counter % 4 == 0:
                buf = BytesIO()
                temp_fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                image_bytes = buf.getvalue()
                self.inference_requested.emit(image_bytes)

            temp_fig.clf()  # Clean up the temporary figure

class ControlsWidget(QWidget):
    noise_params_changed = pyqtSignal(bool, float)
    filter_params_changed = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hackrf_checkbox = QCheckBox("Использовать HackRF")
        self.recorded_signal_checkbox = QCheckBox("Использовать записанный сигнал")
        self.hackrf_checkbox.setChecked(True)
        self.choose_recorded_folder_button = QPushButton("Выбрать папку с записанным сигналом")
        self.choose_record_folder_button = QPushButton("Выбрать базовую папку для записи сигнала")
        self.recording_count_label = QLabel("Количество записей: 0")
        self.toggle_recording_button = QPushButton("Начать запись")
        self.noise_checkbox = QCheckBox("Добавить гауссов шум")
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(50)
        self.noise_slider.setValue(10)
        self.noise_value_label = QLabel("10 дБ")
        self.filter_checkbox = QCheckBox("Использовать фильтры")
        self.filter_lineedit = QLineEdit()
        self.filter_lineedit.setPlaceholderText("[min_freq0:max_freq0],[min_freq1:max_freq1]")

        # New label for ML probabilities
        self.ml_probabilities_label = QLabel("ML Probabilities: N/A")

        self.recording = False
        self.record_count = 0
        self.base_record_folder = None
        self.session_record_folder = None

        layout = QHBoxLayout()
        layout.addWidget(self.hackrf_checkbox)
        layout.addWidget(self.recorded_signal_checkbox)
        layout.addWidget(self.choose_recorded_folder_button)
        layout.addWidget(self.choose_record_folder_button)
        layout.addWidget(self.recording_count_label)
        layout.addWidget(self.toggle_recording_button)
        layout.addWidget(self.noise_checkbox)
        layout.addWidget(self.noise_slider)
        layout.addWidget(self.noise_value_label)
        layout.addWidget(self.filter_checkbox)
        layout.addWidget(self.filter_lineedit)
        layout.addWidget(self.ml_probabilities_label)  # Add the new label to the layout
        self.setLayout(layout)

        self.hackrf_checkbox.toggled.connect(self.on_hackrf_toggled)
        self.recorded_signal_checkbox.toggled.connect(self.on_recorded_toggled)
        self.choose_recorded_folder_button.clicked.connect(self.select_recorded_signal_folder)
        self.choose_record_folder_button.clicked.connect(self.select_record_folder)
        self.toggle_recording_button.clicked.connect(self.toggle_recording)
        self.noise_checkbox.toggled.connect(self.on_noise_params_changed)
        self.noise_slider.valueChanged.connect(self.on_noise_slider_changed)
        self.filter_checkbox.toggled.connect(self.on_filter_params_changed)
        self.filter_lineedit.textChanged.connect(self.on_filter_params_changed)

    @pyqtSlot(list)
    def update_ml_probabilities(self, probs):
        """Update the ML probabilities label with inference results."""
        if len(probs) == 2:
            text = f"ML Probabilities: Drone={probs[0]:.2f}, Noise={probs[1]:.2f}"
        else:
            text = "ML Probabilities: N/A"
        self.ml_probabilities_label.setText(text)

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
        folder = QFileDialog.getExistingDirectory(self, "Выбрать базовую папку для записи сигнала")
        if folder:
            self.base_record_folder = folder
            logger.info(f"Базовая папка для записи сигнала: {folder}")

    def toggle_recording(self):
        if self.recording:
            self.recording = False
            self.toggle_recording_button.setText("Начать запись")
            logger.info("Запись остановлена")
            self.session_record_folder = None
        else:
            if not self.base_record_folder:
                logger.error("Базовая папка для записи не выбрана!")
                return
            self.record_count = 0
            self.recording_count_label.setText("Количество записей: 0")
            self.recording = True
            self.toggle_recording_button.setText("Остановить запись")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            noise_enabled = "noiseOn" if self.noise_checkbox.isChecked() else "noiseOff"
            noise_power = str(self.noise_slider.value())
            filter_enabled = "filterOn" if self.filter_checkbox.isChecked() else "filterOff"
            filters_value = self.filter_lineedit.text().replace(" ", "") or "none"
            filters_value = filters_value.replace(":", "-").replace("/", "-").replace("\\", "-")
            session_folder_name = f"{timestamp}_{noise_enabled}_{noise_power}_{filter_enabled}_{filters_value}"
            self.session_record_folder = os.path.join(self.base_record_folder, session_folder_name)
            try:
                os.makedirs(self.session_record_folder, exist_ok=True)
                logger.info(f"Создана папка для сессии записи: {self.session_record_folder}")
            except Exception as e:
                logger.error(f"Ошибка при создании папки для сессии записи: {e}")
            logger.info("Запись запущена")

    def on_noise_slider_changed(self, value):
        self.noise_value_label.setText(f"{value} дБ")
        self.emit_noise_params_changed()

    def on_noise_params_changed(self, checked=None):
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