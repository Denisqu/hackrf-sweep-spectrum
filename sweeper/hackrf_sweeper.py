from utils.logger import logger
import threading
from blinker import Signal
import subprocess

########### PUBLIC ###########

class HackRFSweeper:
    def __init__(self):
        self.data_ready_signal = Signal()
        self.error_signal = Signal()
        self.stop_event = threading.Event()

        self._impl = _HackRFSweeperImpl()
        self._impl.data_ready_signal.connect(lambda sweep: self.data_ready_signal.send(sweep), weak=False)
        self._impl.process_died_signal.connect(lambda: self.error_signal.send("HackRF Process died"), weak=False)

        self.thread = threading.Thread(target=self.run)
        self.thread.start()
    
    def run(self):
        while not self.stop_event.is_set():
            self._impl.parseSweeps()
        self._impl.stopProcess()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

########### PRIVATE ###########

class _HackRFSweeperImpl:
    def __init__(self):
        self.data_ready_signal = Signal()
        self.process_died_signal = Signal()
        self.current_ranges_mhz = (2400, 2480)
        self.process = None
        self._initProcess()
        
    def parseSweeps(self):
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

                        self.data_ready_signal.send((date_time, hz_low_val, hz_high_val, hz_bin_width_val, dbs_val))
                    except ValueError as ve:
                        logger.error(f"Ошибка преобразования чисел: {ve}")
                        continue
                    
    def stopProcess(self):
        try:
            if self.process:
                self.process.terminate()
                logger.info("Процесс hackrf_sweep завершен.")
                self.process_died_signal.send()
        except Exception as e:
                logger.error(f"Ошибка при завершении процесса hackrf_sweep: {e}")

    def _initProcess(self):
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