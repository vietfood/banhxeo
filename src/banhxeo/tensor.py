from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from banhxeo.buffer import BinaryOp, LazyBuffer, LoadOp, MovementOp, UnaryOp
from banhxeo.device import Device
from banhxeo.view import View
from banhxeo.helper import DEBUG


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
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self.device = device.upper()
        self.requires_grad = False

        if data is None:
            assert shape is not None, "Cannot allocate empty Tensor without shape"
            self.lazydata = LazyBuffer(
                LoadOp.DEFAULT, view=View.create(shape=shape), device=self.device
            )
        else:
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

    def contiguous(self):
        # If already contiguous, do nothing
        if self.lazydata.view.is_contiguous():
            return self
        # It's basically load the source using its complex view,
        # but write it out linearly
        return Tensor(LazyBuffer(op=LoadOp.CONTIGUOUS, 
                                 view=View.create(shape=self.lazydata.view.shape),
                                 src=(self.lazydata,),
                                 device=self.lazydata.device
                                ))

    def reshape(sel)f, new_shape: Tuple[int, ...]):
        if not self.lazydata.view.is_contiguous():
            if DEBUG >= 1:
                print("Trigerring contiguous copy!")
            # this is a naive approach that always forces a copy if
            # tensor isn't contiguous
            contiguous_tensor = self.contiguous()
            return Tensor(contiguous_tensor.lazydata.movement_ops(MovementOp.RESHAPE, new_shape))
        
        # reshape freely
        return Tensor(self.lazydata.movement_ops(MovementOp.RESHAPE, new_shape))

    def permute(self, new_axis: Tuple[int, ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.PERMUTE, new_axis))

    def slice(self, args: Tuple[Tuple[int, ...], ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.SLICE, args))

    def expand(self, shape: Tuple[int, ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.EXPAND, shape))

    def realize(self):
        Device.get_backend(self.device)().exec(self.lazydata)
        return self.lazydata.view_shape(self.lazydata.view.shape)
