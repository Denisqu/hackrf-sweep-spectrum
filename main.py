from utils.logger import logger
from sweeper.hackrf_sweeper import HackRFSweeper

if __name__ == '__main__':
    logger.info("Starting application...")
    sweeper = HackRFSweeper()
    sweeper.data_ready_signal.connect(lambda sweep: logger.info(sweep), weak=False)
    sweeper.error_signal.connect(lambda err: logger.error(err), weak=False)

    try:
        while (True):
            continue
    finally:
        sweeper.stop()