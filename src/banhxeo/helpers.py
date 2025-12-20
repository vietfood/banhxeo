import functools
import os


@functools.lru_cache(maxsize=None)
def getenv(key, default=0):
    return type(default)(os.getenv(key, default))


DEBUG, WINO, IMAGE = getenv("DEBUG"), getenv("WINO"), 0
