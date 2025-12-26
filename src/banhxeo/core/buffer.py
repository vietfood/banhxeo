from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import torch

from banhxeo.core.device import DEFAULT_DEVICE
from banhxeo.core.dtype import DType, dtypes
from banhxeo.core.view import View
from banhxeo.utils.helpers import DEBUG


class UnaryOp(Enum):
    EXP = auto()
    LOG = auto()
    SIN = auto()
    SQRT = auto()
    NEG = auto()
    CAST = auto()


class BinaryOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    MATMUL = auto()
    CMPLT = auto()  # compare less than
    DIV = auto()
    MOD = auto()
    MAX = auto()


class ReduceOp(Enum):
    SUM = auto()
    MAX = auto()


class LoadOp(Enum):
    FROM_CONST = auto()
    FROM_PYTHON = auto()  # list, tuple, any iterable
    FROM_NUMPY = auto()  # we need to preserve shape and strides
    FROM_TORCH = auto()  # same as numpy but we need to take account of device
    FROM_NONE = auto()
    VIEW = auto()
    CONTIGUOUS = auto()
    RAND = auto()


class MovementOp(Enum):
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()  # TODO
    SHRINK = auto()  # TODO
    SLICE = auto()


class TernaryOp(Enum):
    WHERE = auto()


Op: TypeAlias = Union[LoadOp, UnaryOp, BinaryOp, MovementOp, TernaryOp, ReduceOp]


@dataclass
class RawBuffer:
    shape: Tuple[int, ...]  # We assume contiguous for raw buffer
    data: torch.Tensor
    device: str = DEFAULT_DEVICE
    dtype: torch.dtype = torch.float32

    def to(self, device: str):
        return self.data.to(device.lower())

    def numpy(self):
        return self.data.detach().to("cpu").numpy()

    @staticmethod
    def create(
        op: Op,
        args: Any,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        device_str = device.lower() if device else "cpu"
        if op == LoadOp.RAND:
            seed = args[0]
            if shape is None:
                raise ValueError("Shape cannot be None for random Buffer creation")

            if seed is not None:
                torch.manual_seed(seed)

            return RawBuffer(
                shape=shape, data=torch.rand(shape, dtype=dtype, device=device_str)
            )
        else:
            if args is None:
                if shape is None:
                    raise ValueError("Shape cannot be None for empty Buffer creation")
                buf_data = torch.empty(shape, dtype=dtype, device=device_str)
            else:
                data = args[0]
                if isinstance(data, (int, float)):
                    if shape is None:
                        raise ValueError("Scalar creation requires shape")
                    buf_data = torch.full(shape, data, dtype=dtype, device=device_str)
                elif isinstance(data, (List, Tuple)):
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
        dtype: DType = dtypes.float32,
        device: str = DEFAULT_DEVICE,
    ):
        self.op = op
        self.src = src
        self.args = args
        self.view = view
        self.device = device
        self.dtype = dtype

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
        self.realized = RawBuffer.create(
            self.op, self.args, self.view.shape, self.device
        )

    # ---------- Ops Methods ----------

    def compute_ops(self, op: Op, *others: "LazyBuffer", args=None):
        if isinstance(op, BinaryOp):
            assert len(others) == 1, "BinaryOp requires 1 other buffers"
            if op == BinaryOp.MATMUL:
                view = View.create(shape=(self.shape[0], others[0].shape[1]))
            else:
                view = self.view
            return LazyBuffer(
                op, src=(self, others[0]), view=view, device=self.device, args=args
            )
        elif isinstance(op, UnaryOp):
            assert len(others) == 0
            return LazyBuffer(
                op, src=(self,), view=self.view, device=self.device, args=args
            )
        elif isinstance(op, TernaryOp):
            assert len(others) == 2, "TernaryOp requires 2 other buffers"
            return LazyBuffer(
                op,
                src=(self, others[0], others[1]),
                view=self.view,
                device=self.device,
                args=args,
            )
        else:
            raise ValueError(f"This {op} isn't supported yet")

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

        if self.op == LoadOp.FROM_CONST:  # we don't want to "view" const
            return self

        # we treat the MovementOp as a new LoadOp
        # it loads the same pointer as its parent, but using its own view.
        return LazyBuffer(LoadOp.VIEW, src=(self,), view=new_view, device=self.device)

    def reduce_ops(self, op: Op, new_shape: Tuple[int, ...]):
        assert len(self.shape) == len(new_shape), (
            "[ERROR] Reduce shapes must have same dimensions"
        )
        return LazyBuffer(
            op,
            src=(self,),
            view=View.create(new_shape),
            device=self.device,
        )

    # ---------- Other Methods ----------

    def contiguous(self):
        if self.view.is_contiguous():
            return self
        # It's basically load the source using its complex view,
        # but write it out linearly
        return LazyBuffer(
            op=LoadOp.CONTIGUOUS,
            view=View.create(shape=self.view.shape),
            src=(self,),
            device=self.device,
        )

    def const(self, val: float):
        """
        We copy 'self.view' so this constant acts like a tensor of full shape (fill), allowing elementwise ops (like SUB) to infer the correct output shape.
        """
        return LazyBuffer(
            LoadOp.FROM_CONST,
            src=(),
            view=self.view,
            args=[val],
            device=self.device,
        )

    def broadcasted(self, other: "LazyBuffer"):
        if self.view.shape == other.view.shape:
            return self, other

        # We don't broadcast on const
        if other.op == LoadOp.FROM_CONST:
            return self, other

        try:
            out_shape = tuple(
                max(s, o) for s, o in zip(self.view.shape, other.view.shape)
            )
            return self.expand(out_shape), other.expand(out_shape)
        except Exception:
            # This is naive. A real implementation handles (3, 1) + (3,) -> (3, 3)
            # For now, assume users are explicit or shapes match well enough
            # For example, (3, 1) and (1, 3)
            raise ValueError(
                f"Cannot broadcast {self.view.shape} and {other.view.shape}"
            )

    def expand(self, new_shape: Tuple[int, ...]):
        return self.movement_ops(MovementOp.EXPAND, new_shape)

    def permute(self, new_axis: Tuple[int, ...]):
        return self.movement_ops(MovementOp.PERMUTE, new_axis)

    def slice(self, args: Tuple[Tuple[int, ...], ...]):
        return self.movement_ops(MovementOp.SLICE, args)

    def reshape(self, new_shape: Tuple[int, ...]):
        try:
            return self.movement_ops(MovementOp.RESHAPE, new_shape)
        except ValueError:
            # we force a physical copy (contiguous) and try again.
            if DEBUG >= 1:
                print(
                    f"[INFO] Reshape failed for {self.shape}->{new_shape}, triggering contiguous."
                )
            return self.contiguous().movement_ops(MovementOp.RESHAPE, new_shape)

    def matmul(self, other: "LazyBuffer"):
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Incompatible dimensions between {self.shape=} and {other.shape=}"
            )

        if not self.view.is_contiguous():
            print(
                "[WARNING] MatMul should be called with contiguous Tensor => Trigger contiguous copying"
            )
            new_buf = self.contiguous()
        else:
            new_buf = self

        return new_buf.compute_ops(BinaryOp.MATMUL, other)

    def t(self):
        return self.permute((1, 0))

    def where(self, input: "LazyBuffer", other: "LazyBuffer"):
        # https://github.com/tinygrad/teenygrad/blob/main/teenygrad/tensor.py#L719
        x_, y_ = self.broadcasted(input)
        x, z_ = x_.broadcasted(other)
        y, z = y_.broadcasted(z_)
        return x.compute_ops(TernaryOp.WHERE, y, z)

    def view_as(self, shape: Tuple[int, ...]):
        if self.realized is None:
            raise ValueError("Current LazyBuffer isn't realized")
        return self.realized.data.view(shape)
