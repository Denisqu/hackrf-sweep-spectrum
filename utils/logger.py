import logging

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False  # To prevent re-initialization
        return cls._instance

    def __init__(self):
        if self._initialized:
            return        
        self.logger = logging.getLogger('HACKRF_SWEEP_SPECTRUM')
        self.logger.setLevel(logging.DEBUG)

        # Создание форматтера
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Создание обработчика для записи в файл
        self.file_handler = logging.FileHandler('HACKRF_SWEEP_SPECTRUM.log', encoding='utf-8')
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.formatter)

        # Создание обработчика для вывода в консоль
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(self.formatter)

        # Добавление обработчиков к логгеру
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
        self._initialized = True

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

logger = Logger()