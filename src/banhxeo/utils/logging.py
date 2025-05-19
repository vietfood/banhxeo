import logging
from typing import Any

logging_handler = []
try:
    from rich.logging import RichHandler

    logging_handler.append(RichHandler(rich_tracebacks=True, markup=True))
except ModuleNotFoundError:
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logging_handler.append(consoleHandler)


# Set up Logger
class Logger:
    base = logging.getLogger("banhxeo")

    def __init__(self):
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
