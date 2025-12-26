import functools
import os
from typing import List, TypeVar

T = TypeVar("T")


@functools.lru_cache(maxsize=None)
def getenv(key, default=0):
    return type(default)(os.getenv(key, default))


def all_same(items: List[T]):
    return all(x == items[0] for x in items)


def argfix(*x):
    return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x


# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x):
    return type(x)(sorted(range(len(x)), key=x.__getitem__))


def normalize_slice(s, dim_size):
    start, stop, step = s.indices(dim_size)
    # TODO: Handle negative strides (flip) later
    if step < 0:
        raise NotImplementedError("Negative strides not supported yet")
    return start, stop, step


DEBUG, WINO, IMAGE = getenv("DEBUG"), getenv("WINO"), 0
