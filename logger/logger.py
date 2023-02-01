
import logging
from logging import handlers

class Logger():
    def __init__(self,filename,level="debug",fmt="%(asctime)s - %(levelname)s - %(message)s",when="D",backcount=3):
        self.setlevel = {"debug":logging.DEBUG,
                         "info":logging.INFO,
                         "warning":logging.WARNING,
                         "error":logging.WARNING}
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.setlevel[level])
        format_str = logging.Formatter(fmt,datefmt="%Y-%m-%d %H:%M:%S %p")

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)

        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,encoding="utf-8",backupCount=backcount)
        th.setFormatter(format_str)

        self.logger.addHandler(sh)
        self.logger.addHandler(th)

