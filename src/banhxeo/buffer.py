from enum import Enum, auto
from typing import Any, Optional, Tuple, Union

import torch

from banhxeo.view import View


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
    VIEW = auto()
    FROM_CPU = auto()


class MovementOps(Enum):
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()
    SLICE = auto()


type Op = Union[LoadOps, UnaryOps, BinaryOps, MovementOps]


class LazyBuffer:
    def __init__(
        self, op: Op, view: View, src: Tuple["LazyBuffer", ...] = (), args: Any = None
    ):
        self.op = op
        self.src = src
        self.args = args
        self.view = view

        # If we computed this already, store the data here
        self.realized: Optional[torch.Tensor] = None

    def __repr__(self):
        return f"<LazyBuffer (op={self.op}, realized={self.realized}, src={self.src}, args={self.args})>"

    def compute_ops(self, op, *others: "LazyBuffer"):
        if isinstance(op, BinaryOps):
            assert len(others) == 1
            return LazyBuffer(op, src=(self, others[0]), view=self.view)
        elif isinstance(op, UnaryOps):
            assert len(others) == 0
            return LazyBuffer(op, src=(self,), view=self.view)

    def movement_ops(self, op, *args):
        if not isinstance(op, MovementOps):
            raise ValueError("This function accepts MovementOps only")

        if op == MovementOps.PERMUTE:
            new_view = self.view.permute(args[0])
        elif op == MovementOps.SLICE:
            new_view = self.view.slice(args[0])
        elif op == MovementOps.EXPAND:
            new_view = self.view.broadcast_to(args[0])
        else:
            # current view
            new_view = self.view

        # if this is already a VIEW, just update its view and keep the same source
        if self.op == LoadOps.VIEW:
            return LazyBuffer(LoadOps.VIEW, src=self.src, view=new_view)

        # we treat the MovementOp as a new LoadOp
        # it loads the same pointer as its parent, but using its own view.
        return LazyBuffer(LoadOps.VIEW, src=(self,), view=new_view)
