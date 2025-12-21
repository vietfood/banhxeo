from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from banhxeo.device import DEFAULT_DEVICE
from banhxeo.view import View


class UnaryOp(Enum):
    EXP2 = auto()
    LOG2 = auto()
    SIN = auto()
    SQRT = auto()


class BinaryOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()


class LoadOp(Enum):
    CONST = auto()
    VIEW = auto()
    FROM_CPU = auto()


class MovementOp(Enum):
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()
    SLICE = auto()
    CONTIGUOUS = auto()


type Op = Union[LoadOp, UnaryOp, BinaryOp, MovementOp]


@dataclass
class RawBuffer:
    shape: Tuple[int, ...]  # We assume contiguous for raw buffer
    data: torch.Tensor
    device: str = DEFAULT_DEVICE
    dtype: torch.dtype = torch.float32

    def to_cpu(self):
        return self.data.cpu()

    @staticmethod
    def create(
        data: Optional[Union[List, Tuple, np.ndarray, torch.Tensor]],
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if data is None:
            assert shape is not None, "Cannot allocate empty buffer without shape"
            buf_data = torch.empty(shape, dtype=dtype)
        else:
            if isinstance(data, (List, Tuple)):
                # wrap torch Tensor
                buf_data = torch.tensor(data, dtype=dtype)
            elif isinstance(data, np.ndarray):
                # better method for numpy array
                buf_data = torch.from_numpy(data).to(dtype)
            else:
                # copy a new torch Tensor
                buf_data = data.clone().to(dtype)

            if shape is not None:
                buf_data = buf_data.view(shape)

            if device is not None and device.lower() != str(buf_data.device):
                buf_data = buf_data.to(device.lower())

        return RawBuffer(
            shape=buf_data.shape,
            data=buf_data,
            device=buf_data.device.type,
        )


class LazyBuffer:
    def __init__(
        self,
        op: Op,
        view: View,
        src: Tuple["LazyBuffer", ...] = (),
        args: Any = None,
        device: str = DEFAULT_DEVICE,
    ):
        self.op = op
        self.src = src
        self.args = args
        self.view = view
        self.device = device
        # If we computed this already, store the data here
        self.realized: Optional[RawBuffer] = None

    def __repr__(self):
        return f"<LazyBuffer (op={self.op}, realized={self.realized}, src={self.src}, args={self.args})>"

    def allocate(self):
        self.realized = RawBuffer.create(self.args, self.view.shape, self.device)

    def view_shape(self, shape: Tuple[int, ...]):
        if self.realized is None:
            raise ValueError("Current LazyBuffer isn't realized")
        return self.realized.data.view(shape)

    def compute_ops(self, op, *others: "LazyBuffer"):
        if isinstance(op, BinaryOp):
            assert len(others) == 1
            return LazyBuffer(op, src=(self, others[0]), view=self.view)
        elif isinstance(op, UnaryOp):
            assert len(others) == 0
            return LazyBuffer(op, src=(self,), view=self.view)

    def movement_ops(self, op, *args):
        if not isinstance(op, MovementOp):
            raise ValueError("This function accepts MovementOps only")

        if op == MovementOp.PERMUTE:
            new_view = self.view.permute(args[0])
        elif op == MovementOp.SLICE:
            new_view = self.view.slice(args[0])
        elif op == MovementOp.EXPAND:
            new_view = self.view.broadcast_to(args[0])
        else:
            # current view
            new_view = self.view

        # if this is already a VIEW, just update its view and keep the same source
        if self.op == LoadOp.VIEW:
            return LazyBuffer(LoadOp.VIEW, src=self.src, view=new_view)

        # we treat the MovementOp as a new LoadOp
        # it loads the same pointer as its parent, but using its own view.
        return LazyBuffer(LoadOp.VIEW, src=(self,), view=new_view)
