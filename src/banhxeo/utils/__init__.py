import os
import sys
from enum import Enum, auto

from tqdm.auto import tqdm

import __main__


class RuntimeEnv(Enum):
    JUPYTER = auto()
    SHELL = auto()
    IPYTHON = auto()
    COLAB = auto()


def get_runtime() -> RuntimeEnv:
    if "google.colab" in sys.modules:
        return RuntimeEnv.COLAB
    elif "ipykernel" in sys.modules:
        return RuntimeEnv.JUPYTER
    elif "win32" in sys.platform:
        if "CMDEXTVERSION" in os.environ:
            return RuntimeEnv.SHELL
        else:
            return RuntimeEnv.SHELL
    elif "darwin" in sys.platform:
        return RuntimeEnv.SHELL
    else:
        if hasattr(__main__, "__file__"):
            return RuntimeEnv.SHELL
        else:
            return RuntimeEnv.IPYTHON


def validate_config(config_cls, **kwargs):
    from banhxeo.utils.logging import default_logger

    current_config = dict()
    for k, v in kwargs.items():
        if k in config_cls.model_fields:
            current_config[k] = v
        else:
            default_logger.warning(
                f"Ignoring unknown kwarg '{k}' during {config_cls.__name__} creation"
            )

    return config_cls(**current_config)


progress_bar = tqdm
