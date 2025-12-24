from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from banhxeo.core.buffer import BinaryOp, LazyBuffer, LoadOp, MovementOp, UnaryOp
from banhxeo.core.device import DEFAULT_DEVICE, Device
from banhxeo.core.view import View


class Tensor:
    __slots__ = "lazydata", "requires_grad"
    __deletable__ = ("_ctx",)
    training: ClassVar[bool] = False

    def __init__(
        self,
        data: Optional[
            Union[LazyBuffer, List, Tuple, np.ndarray, torch.Tensor, int, float]
        ] = None,
        device: str = DEFAULT_DEVICE,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        device = device.upper()
        self.requires_grad = False

        if data is None:
            assert shape is not None, "Cannot allocate empty Tensor without shape"
            self.lazydata = LazyBuffer(
                LoadOp.DEFAULT, view=View.create(shape=shape), device=device
            )
        else:
            if isinstance(data, LazyBuffer):
                self.lazydata = data
            elif isinstance(data, (int, float)):
                self.lazydata = LazyBuffer(
                    LoadOp.CONST,
                    view=View.create(shape=(1,)),
                    args=[data],
                    device=device,
                )
            elif isinstance(data, (List, Tuple)):
                self.lazydata = LazyBuffer(
                    LoadOp.FROM_CPU,
                    view=View.create(shape=(len(data),)),
                    args=[data],
                    device=device,
                )
            elif isinstance(data, np.ndarray):
                self.lazydata = LazyBuffer(
                    LoadOp.FROM_NUMPY,
                    # for the numpy array it is a little bit problemtic
                    # here we assume the the tensor always continuous
                    # so we flatten numpy array first
                    view=View.create(shape=data.shape),
                    args=[data.flatten()],
                    device=device,
                )

    # ---------- Binary Ops ----------

    def add(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)
        return Tensor(x.lazydata.compute_ops(BinaryOp.ADD, y.lazydata))

    def mul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)
        return Tensor(x.lazydata.compute_ops(BinaryOp.MUL, y.lazydata))

    def sub(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)
        return Tensor(x.lazydata.compute_ops(BinaryOp.SUB, y.lazydata))

    def less(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)
        return Tensor(x.lazydata.compute_ops(BinaryOp.CMPLT, y.lazydata))

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # We need to check dimension and contiguous
        if self.lazydata.shape[1] != other.lazydata.shape[0]:
            raise ValueError(
                f"Incompatible dimensions between {self.lazydata.shape=} and {other.lazydata.shape=}"
            )

        if not self.lazydata.view.is_contiguous():
            print(
                "[WARNING] MatMul should be called with contiguous Tensor => Trigger contiguous copying"
            )
            self = self.contiguous()

        return Tensor(self.lazydata.compute_ops(BinaryOp.MATMUL, other.lazydata))

    # ---------- Unary Ops ----------

    def log2(self):
        return Tensor(self.lazydata.compute_ops(UnaryOp.LOG2))

    def exp2(self):
        return Tensor(self.lazydata.compute_ops(UnaryOp.EXP2))

    def sin(self):
        return Tensor(self.lazydata.compute_ops(UnaryOp.SIN))

    def sqrt(self):
        return Tensor(self.lazydata.compute_ops(UnaryOp.SQRT))

    # ---------- Load Ops ----------

    def contiguous(self):
        if self.lazydata.view.is_contiguous():
            return self
        # It's basically load the source using its complex view,
        # but write it out linearly
        return Tensor(
            LazyBuffer(
                op=LoadOp.CONTIGUOUS,
                view=View.create(shape=self.lazydata.view.shape),
                src=(self.lazydata,),
                device=self.lazydata.device,
            )
        )

    # ---------- Movement Ops ----------

    def _broadcasted(self, other):
        if self.lazydata.view.shape == other.lazydata.view.shape:
            return self, other

        # We don't broadcast on const
        if other.lazydata.op == LoadOp.CONST:
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

    def reshape(self, new_shape: Tuple[int, ...]):
        if not self.lazydata.view.is_contiguous():
            print("[WARNING] Trigerring contiguous copy!")
            # this is a naive approach that always forces a copy if
            # tensor isn't contiguous
            contiguous_tensor = self.contiguous()
            return Tensor(
                contiguous_tensor.lazydata.movement_ops(MovementOp.RESHAPE, new_shape)
            )

        # reshape freely
        return Tensor(self.lazydata.movement_ops(MovementOp.RESHAPE, new_shape))

    def permute(self, new_axis: Tuple[int, ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.PERMUTE, new_axis))

    def slice(self, args: Tuple[Tuple[int, ...], ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.SLICE, args))

    def expand(self, shape: Tuple[int, ...]):
        return Tensor(self.lazydata.movement_ops(MovementOp.EXPAND, shape))

    # ---------- Ops Wrapper ----------

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __sub__(self, other):
        return self.sub(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __lt__(self, other):
        return self.less(other)

    # ---------- Realize Method ----------

    def realize(self):
        Device.get_backend(self.lazydata.device)().exec(self.lazydata)
        return self.lazydata.view_as(self.lazydata.shape)
