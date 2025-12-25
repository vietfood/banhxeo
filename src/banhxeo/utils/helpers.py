import functools
import os
from typing import List, TypeVar

T = TypeVar("T")


@functools.lru_cache(maxsize=None)
def getenv(key, default=0):
    return type(default)(os.getenv(key, default))


def all_same(items: List[T]):
    return all(x == items[0] for x in items)


# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x):
    return type(x)(sorted(range(len(x)), key=x.__getitem__))


DEBUG, WINO, IMAGE = getenv("DEBUG"), getenv("WINO"), 0
