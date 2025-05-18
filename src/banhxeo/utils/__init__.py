import os
import sys
from enum import Enum, auto

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


def _progress_bar():
    if get_runtime() in [RuntimeEnv.COLAB, RuntimeEnv.JUPYTER]:
        from tqdm.notebook import tqdm

        return tqdm
    elif get_runtime() == RuntimeEnv.SHELL:
        from tqdm import tqdm

        return tqdm
    else:
        raise ValueError()


progress_bar = _progress_bar()
