import logging
from typing import Any

from banhxeo.utils import RuntimeEnv, get_runtime


class PrintLogger:  # fix Colab issues
    template: str = "[{level}]: {msg}"

    def info(self, msg: str):
        print(self.template.format(level="INFO", msg=msg))

    def debug(self, msg: str):
        print(self.template.format(level="DEBUG", msg=msg))

    def exception(self, msg: str):
        print(self.template.format(level="EXCEPTION", msg=msg))

    def error(self, msg: str):
        print(self.template.format(level="ERROR", msg=msg))

    def warning(self, msg: str):
        print(self.template.format(level="WARNING", msg=msg))


# Set up Logger
class Logger:
    def __init__(self):
        if get_runtime() == RuntimeEnv.COLAB:
            self.base = PrintLogger()
        else:
            self.base = logging.getLogger("banhxeo")

            logging_handler = []
            logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)
            logging_handler.append(consoleHandler)

            # set up config
            logging.basicConfig(
                level="NOTSET",
                format="%(message)s",
                datefmt="[%Y-%m-%d-%H:%M:%S%z]",
                handlers=logging_handler,
            )

    def info(self, msg: str):
        self.base.info(msg)

    def debug(self, msg: str):
        self.base.debug(msg)

    def exception(self, msg: str):
        self.base.exception(msg)

    def error(self, msg: str):
        self.base.error(msg)

    def warning(self, msg: str):
        self.base.warning(msg)

    def check_and_raise(self, msg: str, error_type: Exception, condition: Any):
        if not condition:
            self.error(msg)
            raise error_type


DEFAULT_LOGGER = Logger()
