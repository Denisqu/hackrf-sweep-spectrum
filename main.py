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

# def plotter(sweep_data):
#     frequencies = []
#     dBs = []
#     time_groups = {}

#     for entry in sweep_data:
#         date_time, hz_low, hz_high, hz_bin_width, dB_values = entry
#         num_bins = int((hz_high - hz_low) / hz_bin_width)
#         if date_time not in time_groups:
#             time_groups[date_time] = []
#         time_groups[date_time].extend(dB_values)
    
#     # Генерируем центральные частоты для каждого бина
#     for i in range(len(dB_values)):
#         freq = hz_low + i * hz_bin_width + hz_bin_width / 2
#         frequencies.append(freq)
#         dBs.append(dB_values[i])

#     spectrogram_matrix = np.array([time_groups[time] for time in sorted(time_groups.keys())])
#     combined = sorted(zip(frequencies, dBs), key=lambda x: x[0])
#     frequencies_sorted, dBs_sorted = zip(*combined)

#     plt.figure(figsize=(15, 5))
#     plt.imshow(spectrogram_matrix, aspect='auto', cmap='viridis',
#             extent=[min(frequencies_sorted), max(frequencies_sorted),
#                     len(time_groups), 0])
#     plt.colorbar(label='dB')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Time Index')
#     plt.title('Spectrogram')
#     plt.show()

# class Worker(QObject):
#     finished = pyqtSignal()  # Signal emitted when the task is done
#     progress = pyqtSignal(int)  # Signal to report progress
#     result = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # Signal to send the spectrogram data

#     def run(self):
#         """Simulate a time-consuming task (e.g., generating a spectrogram)."""
#         fs = 1000  # Sampling frequency
#         t = np.linspace(0, 2, 2 * fs, endpoint=False)
#         signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

#         # Simulate progress updates
#         for i in range(101):
#             self.progress.emit(i)  # Emit progress
#             QThread.msleep(20)  # Simulate delay

#         # Compute the spectrogram
#         f, t, Sxx = plt.mlab.specgram(signal, Fs=fs, NFFT=1024, noverlap=512)

#         # Emit the result
#         self.result.emit(f, t, Sxx)
#         self.finished.emit()

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("Multithreading with PyQt")
#         self.setGeometry(100, 100, 800, 600)

#         # Create a figure and a canvas to display the spectrogram
#         self.figure, self.ax = plt.subplots()
#         self.canvas = FigureCanvas(self.figure)

#         # Create a button to start the task
#         self.button = QPushButton("Start Task")
#         self.button.clicked.connect(self.start_task)

#         # Set up the layout
#         layout = QVBoxLayout()
#         layout.addWidget(self.canvas)
#         layout.addWidget(self.button)

#         # Create a central widget and set the layout
#         central_widget = QWidget()
#         central_widget.setLayout(layout)
#         self.setCentralWidget(central_widget)

#         # Initialize worker and thread
#         self.worker = Worker()
#         self.thread = QThread()

#         # Move the worker to the thread
#         self.worker.moveToThread(self.thread)

#         # Connect signals and slots
#         self.thread.started.connect(self.worker.run)
#         self.worker.finished.connect(self.thread.quit)
#         self.worker.finished.connect(self.worker.deleteLater)
#         self.thread.finished.connect(self.thread.deleteLater)
#         self.worker.progress.connect(self.update_progress)
#         self.worker.result.connect(self.plot_spectrogram)

#     def start_task(self):
#         """Start the worker thread."""
#         self.button.setEnabled(False)  # Disable the button while the task is running
#         self.thread.start()

#     def update_progress(self, value):
#         """Update the progress of the task."""
#         self.setWindowTitle(f"Progress: {value}%")

#     def plot_spectrogram(self, f, t, Sxx):
#         """Plot the spectrogram."""
#         self.ax.clear()
#         self.ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
#         self.ax.set_ylabel('Frequency [Hz]')
#         self.ax.set_xlabel('Time [sec]')
#         self.ax.set_title('Spectrogram')
#         self.canvas.draw()

#         # Re-enable the button
#         self.button.setEnabled(True)

#         # Show a message box when done
#         QMessageBox.information(self, "Task Complete", "The spectrogram has been generated!")

def main():
    logger.error("test")
    app = QApplication(sys.argv)
    sweeper = HackRFSweeper()
    sweeper.sweeper_stopped_signal.connect(lambda: QApplication.instance().quit())
    window = MainWindow(sweeper)
    window.show()
    sweeper.start()
    # profiler = cProfile.Profile()
    # profiler.enable()
    result = app.exec_()
    # profiler.disable()
    # profiler.dump_stats('qt_app_profile.prof')
    sys.exit(result)


if __name__ == '__main__':
    # app = QApplication(sys.argv)

    # logger.info("Starting application...")
    # sweeper = HackRFSweeper()
    # sweeper.start()
    # # sweeper.data_ready_signal.connect(plotter, weak=False)
    # # sweeper.error_signal.connect(lambda err: logger.error(err), weak=False)

    # app.exec()
    # try:
    #     while (True):
    #         continue
    # finally:
    #     pass

    main()
    #test()

    # window = MainWindow()
    # window.show()
    # sys.exit(app.exec_())

