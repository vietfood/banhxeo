from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import torch

from banhxeo.core.device import DEFAULT_DEVICE
from banhxeo.core.view import View
from banhxeo.utils.helpers import DEBUG


class UnaryOp(Enum):
    EXP2 = auto()
    LOG2 = auto()
    SIN = auto()
    SQRT = auto()


class BinaryOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    MATMUL = auto()
    CMPLT = auto()  # compare less than


class ReduceOp(Enum):
    SUM = auto()
    MAX = auto()


class LoadOp(Enum):
    CONST = auto()
    VIEW = auto()
    FROM_CPU = auto()  # list, tuple, any iterable
    FROM_NUMPY = auto()  # we need to preserve shape and strides
    DEFAULT = auto()
    CONTIGUOUS = auto()


class MovementOp(Enum):
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()
    SLICE = auto()


class TernaryOp(Enum):
    WHERE = auto()


Op: TypeAlias = Union[LoadOp, UnaryOp, BinaryOp, MovementOp]


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
        device_str = device.lower() if device else "cpu"

        if data is None:
            assert shape is not None  # This should be handle by Tensor
            buf_data = torch.empty(shape, dtype=dtype, device=device_str)
        else:
            if isinstance(data, (List, Tuple)):
                # we assume 1D Tensor as default
                # note that convert to numpy then wrap torch is faster than raw list
                buf_data = torch.from_numpy(np.array(data)).to(
                    dtype=dtype, device=device_str
                )
            elif isinstance(data, np.ndarray):
                buf_data = torch.from_numpy(data).to(dtype=dtype, device=device_str)
            else:
                # copy a new torch Tensor
                buf_data = data.clone().to(dtype=dtype, device=device_str)

            if shape is not None:
                buf_data = buf_data.view(shape)

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
        return f"<LazyBuffer (op={self.op}, view={self.view}, realized={self.realized}, args={self.args}, src={self.src})>"

    @property
    def shape(self):
        return self.view.shape

    @property
    def strides(self):
        return self.view.strides

    @property
    def offset(self):
        return self.view.offset

    def allocate(self):
        if self.realized is not None:
            return
        self.realized = RawBuffer.create(self.args, self.view.shape, self.device)

    def view_as(self, shape: Tuple[int, ...]):
        if self.realized is None:
            raise ValueError("Current LazyBuffer isn't realized")
        return self.realized.data.view(shape)

    def compute_ops(self, op: Op, *others: "LazyBuffer"):
        if isinstance(op, BinaryOp):
            assert len(others) == 1, "BinaryOp requires 1 other buffers"
            if op == BinaryOp.MATMUL:
                view = View.create(shape=(self.shape[0], others[0].shape[1]))
            else:
                view = self.view
            return LazyBuffer(op, src=(self, others[0]), view=view, device=self.device)
        elif isinstance(op, UnaryOp):
            assert len(others) == 0
            return LazyBuffer(op, src=(self,), view=self.view, device=self.device)
        elif isinstance(op, TernaryOp):
            assert len(others) == 2, "TernaryOp requires 2 other buffers"
            return LazyBuffer(
                op, src=(self, others[0], others[1]), view=self.view, device=self.device
            )

    def movement_ops(self, op: Op, *args):
        if op == MovementOp.PERMUTE:
            new_view = self.view.permute(args[0])
        elif op == MovementOp.SLICE:
            new_view = self.view.slice(args[0])
        elif op == MovementOp.EXPAND:
            new_view = self.view.broadcast_to(args[0])
        elif op == MovementOp.RESHAPE:
            new_view = self.view.reshape(args[0])
        else:
            # current view
            new_view = self.view

        if DEBUG >= 2:
            print(
                f"Current view {self.view} => MovementOp={str(op)} with new view {new_view}"
            )

        # if this is already a VIEW, just update its view and keep the same source
        if self.op == LoadOp.VIEW:
            return LazyBuffer(
                LoadOp.VIEW, src=self.src, view=new_view, device=self.device
            )

        if self.op == LoadOp.CONST:  # we don't want to "view" const
            return self

        # we treat the MovementOp as a new LoadOp
        # it loads the same pointer as its parent, but using its own view.
        return LazyBuffer(LoadOp.VIEW, src=(self,), view=new_view, device=self.device)
