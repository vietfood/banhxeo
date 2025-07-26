import logging
from typing import Any

from banhxeo.utils import RuntimeEnv, get_runtime


# Fix Colab issue
class PrintLogger:  # noqa: D101
    template: str = "[{level}]: {msg}"

    def info(self, msg: str):
        """Logs an info-level message.

        Args:
            msg (str): The message to log.
        """
        print(self.template.format(level="INFO", msg=msg))

    def debug(self, msg: str):
        """Logs a debug-level message.

        Args:
            msg (str): The message to log.
        """
        print(self.template.format(level="DEBUG", msg=msg))

    def exception(self, msg: str):
        """Logs an exception-level message.

        Args:
            msg (str): The message to log.
        """
        print(self.template.format(level="EXCEPTION", msg=msg))

    def error(self, msg: str):
        """Logs an error-level message.

        Args:
            msg (str): The message to log.
        """
        print(self.template.format(level="ERROR", msg=msg))

    def warning(self, msg: str):
        """Logs a warning-level message.

        Args:
            msg (str): The message to log.
        """
        print(self.template.format(level="WARNING", msg=msg))


# Set up Logger
class Logger:  # noqa: D101
    def __init__(self):
        """Initializes the Logger instance.

        Chooses between PrintLogger (for Colab) and the standard logging.Logger.
        """
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
        """Logs an info-level message.

        Args:
            msg (str): The message to log.
        """
        self.base.info(msg)

    def debug(self, msg: str):
        """Logs a debug-level message.

        Args:
            msg (str): The message to log.
        """
        self.base.debug(msg)

    def exception(self, msg: str):
        """Logs an exception-level message.

        Args:
            msg (str): The message to log.
        """
        self.base.exception(msg)

    def error(self, msg: str):
        """Logs an error-level message.

        Args:
            msg (str): The message to log.
        """
        self.base.error(msg)

    def warning(self, msg: str):
        """Logs a warning-level message.

        Args:
            msg (str): The message to log.
        """
        self.base.warning(msg)

    def check_and_raise(self, msg: str, error_type: Exception, condition: Any):
        """Checks a condition and raises an error if the condition is not met.

        Args:
            msg (str): The error message to log and raise.
            error_type (Exception): The exception type to raise if the condition is False.
            condition (Any): The condition to check. If False, the error is raised.

        Raises:
            error_type: If the condition is False.
        """
        if not condition:
            self.error(msg)
            raise error_type


default_logger = Logger()
