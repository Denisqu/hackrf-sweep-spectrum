from utils.logger import logger
import threading
from blinker import Signal
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal, QMetaObject, QObject, Qt, pyqtSlot

from time import sleep

########### PUBLIC ###########

class HackRFSweeper(QObject):
    data_ready_signal = pyqtSignal(object)
    _stop_signal      = pyqtSignal()
    sweeper_stopped_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._impl = _HackRFSweeperImpl()
        self._impl.data_ready_signal.connect(self.data_ready_signal)
        self._impl.stop_signal.connect(self.sweeper_stopped_signal)
        self.thread = QThread()
        self.thread.start()
        self._impl.moveToThread(self.thread)
    
    def start(self):
        QMetaObject.invokeMethod(
            self._impl, 
            "init",  
            Qt.ConnectionType.QueuedConnection, 
        )
        QMetaObject.invokeMethod(
            self._impl, 
            "parse_sweeps",
            Qt.ConnectionType.QueuedConnection,
        )

    def stop(self):
        self._stop_signal.emit()
        self.thread.wait()

########### PRIVATE ###########

class _HackRFSweeperImpl(QObject):
    data_ready_signal = pyqtSignal(object)
    stop_signal       = pyqtSignal()
    
    def __init__(self):
        self.i = 0
        super().__init__()
        self.process_died_signal = Signal()
        self.current_ranges_mhz = (2400, 2480)
        self.process = None
        self._current_buffer = []
        logger.info("init _HackRFSweeperImpl")  

    @pyqtSlot()
    def init(self):
        logger.info("init _HackRFSweeperImpl")
        self._init_process()
    
    @pyqtSlot()
    def parse_sweeps(self):
        while True:
            for line in self.process.stdout:
                line = line.strip()
                if "," in line:
                    # Разделяем строку на части
                    fields = line.split(", ")
                    if len(fields) > 6:  # Проверяем наличие необходимого количества полей
                        date_time = fields[0] + fields[1]
                        hz_low = fields[2].replace(',', '')  # Удаление запятых
                        hz_high = fields[3].replace(',', '')  # Удаление запятых
                        hz_bin_width = fields[4].replace(',', '')  # Удаление запятых
                        # Удаление запятых из каждого элемента dbs
                        dbs = [field.replace(',', '') for field in fields[6:]]
                        try:
                            # Преобразование строк в числа
                            hz_low_val = int(float(hz_low))
                            hz_high_val = int(float(hz_high))
                            hz_bin_width_val = float(hz_bin_width)
                            dbs_val = [float(db) for db in dbs]
                        except ValueError as ve:
                            logger.error(f"Ошибка преобразования чисел: {ve}")
                            continue

                        if hz_low_val == int(self.current_ranges_mhz[0]) * 1e6 and len(self._current_buffer) > 0:
                            logger.info(f'started emitting data_ready_signal from thread {int(QThread.currentThread().currentThreadId())}')
                            self.data_ready_signal.emit(self._current_buffer)
                            logger.info(f'ended emitting data_ready_signal from thread {int(QThread.currentThread().currentThreadId())}')                                
                            self._current_buffer = []
                        self._current_buffer.append((date_time, hz_low_val, hz_high_val, hz_bin_width_val, dbs_val))
        self.stop_signal.emit()

                    
    def stop_process(self):
        try:
            if self.process:
                self.process.terminate()
                logger.info("Процесс hackrf_sweep завершен.")
                self.process_died_signal.send()
        except Exception as e:
                logger.error(f"Ошибка при завершении процесса hackrf_sweep: {e}")

    def _init_process(self):
        logger.info("Starting hackrf_sweep proccess")
        command = ["hackrf_sweep", "-f", f"{int(self.current_ranges_mhz[0])}:{int(self.current_ranges_mhz[1])}"]
        try:
            # Запуск процесса
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
                logger.error(f"Произошла ошибка в парсере: {e}")
                self.process_died_signal.send()