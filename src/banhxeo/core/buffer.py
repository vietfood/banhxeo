from enum import Enum, auto
from typing import Any, List, Optional, Union

import torch


class UnaryOps(Enum):
    EXP2 = auto()
    LOG2 = auto()
    SIN = auto()
    SQRT = auto()


class BinaryOps(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()


class LoadOps(Enum):
    CONST = auto()
    FROM_CPU = auto()


type Ops = Union[LoadOps, UnaryOps, BinaryOps]


class LazyBuffer:
    def __init__(self, op: Ops, src: List["LazyBuffer"] = [], args: Any = None):
        self.op = op
        self.src = src
        self.args = args

        # If we computed this already, store the data here
        self.realized: Optional[torch.Tensor] = None

    def __repr__(self):
        return f"<LB {(self.op, self.realized, len(self.src), self.args)}>"

    def build(self, op, *others: "LazyBuffer"):
        # we build graph here
        if isinstance(op, BinaryOps):
            assert len(others) == 1
            return LazyBuffer(op, src=[self, others[0]])
        elif isinstance(op, UnaryOps):
            assert len(others) == 0
            return LazyBuffer(
                op,
                src=[self],
            )
        else:
            pass  # for LoadOps, it's a leaf
