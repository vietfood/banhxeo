from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from banhxeo.buffer import BinaryOp, LazyBuffer, LoadOp, MovementOp, UnaryOp
from banhxeo.device import Device
from banhxeo.view import View


class Tensor:
    __slots__ = "lazydata", "requires_grad", "device"
    __deletable__ = ("_ctx",)
    training: ClassVar[bool] = False

    def __init__(
        self,
        data: Optional[
            Union[LazyBuffer, List, Tuple, np.ndarray, torch.Tensor, int, float]
        ] = None,
        device: str = "cpu",
    ):
        self.device = device.upper()

        if isinstance(data, LazyBuffer):
            self.lazydata = data
        elif isinstance(data, (int, float)):
            self.lazydata = LazyBuffer(
                LoadOp.CONST,
                view=View.create(shape=(1,)),
                args=[data],
                device=self.device,
            )
        elif isinstance(data, (List, Tuple, np.ndarray)):
            self.lazydata = LazyBuffer(
                LoadOp.FROM_CPU,
                view=View.create(shape=(len(data),)),
                args=[data],
                device=self.device,
            )

    def _broadcasted(self, other):
        if self.lazydata.view.shape == other.lazydata.view.shape:
            return self, other

        try:
            out_shape = tuple(
                max(s, o)
                for s, o in zip(self.lazydata.view.shape, other.lazydata.view.shape)
            )
            return self.expand(out_shape), other.expand(out_shape)
        except Exception:
            # This is naive. A real implementation handles (3, 1) + (3,) -> (3, 3)
            # For now, assume users are explicit or shapes match well enough
            # For example, (3, 1) and (1, 3)
            raise ValueError(
                f"Cannot broadcast {self.lazydata.view.shape} and {other.lazydata.view.shape}"
            )

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)
        return Tensor(x.lazydata.compute_ops(BinaryOp.ADD, y.lazydata))

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)
        return Tensor(x.lazydata.compute_ops(BinaryOp.MUL, y.lazydata))

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)
        return Tensor(x.lazydata.compute_ops(BinaryOp.SUB, y.lazydata))

    def log2(self):
        return Tensor(self.lazydata.compute_ops(UnaryOp.LOG2))

    def exp2(self):
        return Tensor(self.lazydata.compute_ops(UnaryOp.EXP2))

    def sin(self):
        return Tensor(self.lazydata.compute_ops(UnaryOp.SIN))

    def sqrt(self):
        return Tensor(self.lazydata.compute_ops(UnaryOp.SQRT))

    def permute(self, new_axis: Tuple[int, ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.PERMUTE, new_axis))

    def slice(self, args: Tuple[Tuple[int, ...], ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.SLICE, args))

    def expand(self, shape: Tuple[int, ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.EXPAND, shape))

    def realize(self):
        Device.get_backend(self.device)().exec(self.lazydata)
        return self.lazydata.view_shape(self.lazydata.view.shape)
