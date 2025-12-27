import math
import time
from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from banhxeo.core.buffer import LazyBuffer, LoadOp
from banhxeo.core.device import DEFAULT_DEVICE, Device
from banhxeo.core.dtype import DType, dtypes
from banhxeo.core.view import View


class Tensor:
    __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
    __deletable__ = ("_ctx",)

    training: ClassVar[bool] = False
    no_grad: ClassVar[bool] = False
    default_type: ClassVar[DType] = dtypes.float32
    _seed: ClassVar[int] = int(time.time())

    def __init__(
        self,
        data: Optional[
            Union[LazyBuffer, List, Tuple, np.ndarray, torch.Tensor, int, float]
        ] = None,
        device: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[DType] = None,
        requires_grad: Optional[bool] = None,
    ):
        if device is None:
            device = DEFAULT_DEVICE
        device = device.upper()

        if dtype is None:
            dtype = Tensor.default_type

        if data is None:
            assert shape is not None, "Cannot allocate empty Tensor without shape"
            self.lazydata = LazyBuffer(
                LoadOp.FROM_NONE,
                view=View.create(shape=shape),
                device=device,
                dtype=dtype,
            )
        else:
            if isinstance(data, LazyBuffer):
                self.lazydata = data
            elif isinstance(data, (int, float)):
                self.lazydata = LazyBuffer(
                    LoadOp.FROM_CONST,
                    view=View.create(shape=(1,) if shape is None else shape),
                    args=[data],
                    device=device,
                    dtype=dtype,
                )
            elif isinstance(data, (List, Tuple)):
                self.lazydata = LazyBuffer(
                    LoadOp.FROM_PYTHON,
                    view=View.create(shape=(len(data),)),
                    args=[data],
                    device=device,
                    dtype=dtype,
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
                    dtype=dtype,
                )
            elif isinstance(data, torch.Tensor):
                self.lazydata = LazyBuffer(
                    LoadOp.FROM_TORCH,
                    view=View.create(shape=data.shape),
                    args=[
                        data.detach().flatten()
                    ],  # remove the autograd graph from current Tensor and flatten it
                    device=device,
                    dtype=dtype,
                )

        # gradient of this Tensor
        # or basically a reference to another Tensor in graph
        self.grad: Optional[Tensor] = None

        # NOTE: this can be in three states. False and None: no gradient, True: gradient
        # None (the default) will be updated to True if it's put in an optimizer
        self.requires_grad = requires_grad

        # internal variables used for autograd graph construction
        from banhxeo.core.function import Function

        self._ctx: Optional[Function] = None

    def __repr__(self):
        return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad else None)!r}>"

    def __str__(self):
        if self.lazydata.realized is None:
            print("[WARNING] Tensor isn't realized yet!")
            return self.__repr__()
        return str(self.lazydata.realized.data)

    def __hash__(self):
        return id(self)

    # ---------- Property ----------

    @property
    def device(self) -> str:
        return self.lazydata.device

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lazydata.shape

    @property
    def dtype(self) -> DType:
        return self.lazydata.dtype

    # ---------- Binary Ops ----------

    def add(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Add

        return Add.apply(x, y)

    def mul(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Mul

        return Mul.apply(x, y)

    def sub(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Sub

        return Sub.apply(x, y)

    def less(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Less

        return Less.apply(x, y)

    def matmul(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)

        from banhxeo.core.function import Matmul

        return Matmul.apply(self, other)

    def div(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Div

        return Div.apply(x, y)

    def maximum(self, x) -> "Tensor":
        from banhxeo.core.function import Maximum

        return Maximum.apply(x)

    def minimum(self, x) -> "Tensor":
        return -((-self).maximum(-x))

    # ---------- Unary Ops ----------

    def log(self) -> "Tensor":
        from banhxeo.core.function import Log

        return Log.apply(self)

    def exp(self) -> "Tensor":
        from banhxeo.core.function import Exp

        return Exp.apply(self)

    def sin(self) -> "Tensor":
        from banhxeo.core.function import Sin

        return Sin.apply(self)

    def sqrt(self) -> "Tensor":
        from banhxeo.core.function import Sqrt

        return Sqrt.apply(self)

    def neg(self) -> "Tensor":
        from banhxeo.core.function import Neg

        return Neg.apply(self)

    def cos(self) -> "Tensor":
        return ((math.pi / 2) - self).sin()

    def cast(self, dtype=None) -> "Tensor":
        from banhxeo.core.function import Cast

        return Cast.apply(self, dtype=dtype)

    # ---------- Ternary Ops ----------

    def _where(self, input, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        input = input if isinstance(input, Tensor) else Tensor(input)

        from banhxeo.core.function import Where

        return Where.apply(self, input, other)

    # ---------- Reduce Ops ----------

    def _reduce(self, reduce_fn, axis=None, keepdim=False) -> "Tensor":
        # 1. handle "Sum All": Flatten then sum
        if axis is None:
            # reshape(-1) flattens to 1D, then we sum that single dimension
            return self.reshape((-1,))._reduce(reduce_fn, axis=0, keepdim=False)

        # 2. handle Negative Axis
        if axis < 0:
            axis += len(self.shape)

        # 3. Handle Permutation (Move reduction axis to end)
        if axis != len(self.shape) - 1:
            permute_order = [i for i in range(len(self.shape)) if i != axis] + [axis]
            ret = self.permute(tuple(permute_order))._reduce(
                reduce_fn, axis=-1, keepdim=False
            )

            # If keepdim, we need to reshape it back to have a '1' in the original axis
            if keepdim:
                # e.g., (2, 3) sum(0) -> (3,) -> reshape(1, 3)
                shape = list(self.shape)
                shape[axis] = 1
                return ret.reshape(tuple(shape))
            return ret

        # 4. The Actual Reduction (Axis is now -1)
        # Calculate new shape: (10, 20, 30) -> (10, 20)
        new_shape = self.shape[:-1]

        ret = reduce_fn.apply(self, new_shape)

        # Handle keepdim for the simple case
        if keepdim:
            shape = list(self.shape)
            shape[axis] = 1
            return ret.reshape(tuple(shape))

        return ret

    def sum(self, axis=None, keepdim=False):
        from banhxeo.core.function import Sum

        return self._reduce(Sum, axis, keepdim)

    def max(self, axis=None, keepdim=False):
        from banhxeo.core.function import Max

        return self._reduce(Max, axis, keepdim)

    def min(self, axis=None, keepdim=False):
        # min(x) = -max(-x)
        return -((-self).max(axis=axis, keepdim=keepdim))

    def mean(self, axis=None, keepdim=False):
        # mean(x) = sum(x) / count
        out = self.sum(axis=axis, keepdim=keepdim)

        # Calculate divisor size
        if axis is None:
            div = math.prod(self.shape)
        elif isinstance(axis, int):
            div = self.shape[axis]
        else:
            # Handle tuple axis if you support it later
            div = math.prod(self.shape[i] for i in axis)

        return out.div(div)

    # ---------- Load Ops ----------

    def contiguous(self):
        from banhxeo.core.function import Contiguous

        return Contiguous.apply(self)

    def contiguous_backward(self):
        from banhxeo.core.function import ContiguousBackward

        return ContiguousBackward.apply(self)

    # ---------- Movement Ops ----------

    def _broadcasted(self, other: "Tensor") -> Tuple["Tensor", "Tensor"]:
        x, y = self.lazydata.broadcasted(other.lazydata)
        return Tensor(x), Tensor(y)

    def reshape(self, new_shape: Tuple[int, ...]):
        from banhxeo.core.function import Reshape

        return Reshape.apply(self, shape=new_shape)

    def permute(self, new_axis: Tuple[int, ...]):
        from banhxeo.core.function import Permute

        return Permute.apply(self, order=new_axis)

    def slice(self, args: Tuple[Tuple[int, ...], ...]):
        """
        TODO
        """

    def expand(self, shape: Tuple[int, ...]):
        from banhxeo.core.function import Expand

        return Expand.apply(self, shape=shape)

    def _transpose(self):
        assert self.shape == 2, (
            "Transpose only works with 2 dimension, please use Permute for more than 2 dimensions"
        )
        return self.permute((1, 0))

    def __getitem__(self, val):
        from banhxeo.utils.helpers import normalize_slice

        if not isinstance(val, tuple):
            val = (val,)

        # TODO: Handle ellipsis (...)

        new_shape = []
        new_strides = []
        new_offset = self.lazydata.view.offset

        current_dim = 0
        for v in val:
            dim_size = self.shape[current_dim]
            dim_stride = self.lazydata.view.strides[current_dim]

            if isinstance(v, int):
                # Integer Indexing: x[2]
                # 1. Normalize negative index
                if v < 0:
                    v += dim_size
                if not (0 <= v < dim_size):
                    raise IndexError(
                        f"Index {v} out of bounds for dim {current_dim} size {dim_size}"
                    )

                # 2. Update Offset: Move pointer forward
                new_offset += v * dim_stride

                # 3. Drop Dimension: We do NOT append to new_shape/new_strides

            elif isinstance(v, slice):
                # Slicing: x[1:5:2]
                start, stop, step = normalize_slice(v, dim_size)

                # 1. Update Offset: Move to start
                new_offset += start * dim_stride

                # 2. Update Shape: How many elements?
                # Formula: ceil((stop - start) / step)
                new_dim_size = math.ceil((stop - start) / step)
                new_shape.append(max(0, new_dim_size))

                # 3. Update Stride: Stride multiplies
                new_strides.append(dim_stride * step)

            else:
                raise NotImplementedError(f"Indexing with {type(v)} not supported")

            current_dim += 1

        # append remaining dimensions untouched
        while current_dim < len(self.shape):
            new_shape.append(self.shape[current_dim])
            new_strides.append(self.lazydata.view.strides[current_dim])
            current_dim += 1

        new_view = View(tuple(new_shape), tuple(new_strides), new_offset)

        return Tensor(
            LazyBuffer(
                LoadOp.VIEW,
                src=self.lazydata.src,
                view=new_view,
                device=self.device,
            )
        )

    # ---------- Ops Wrapper ----------

    # fmt: off
    def __add__(self, other): return self.add(other)
    def __radd__(self, other): return other.add(self)
    def __iadd__(self, other): return self.assign(self.add(other))

    def __mul__(self, other): return self.mul(other)
    def __rmul__(self, other): return other.mul(self)
    def __imul__(self, other): return self.assign(self.mul(other))

    def __sub__(self, other): return self.sub(other)
    def __isub__(self, other): return self.assign(self.add(other))
    def __rsub__(self, other): return other.sub(self)

    def __matmul__(self, other): return self.matmul(other)
    def __rmatmul__(self, other): return other.matmul(self)
    def __imatmul__(self, other): return self.assign(self.matmul(other))

    def __truediv__(self, other): return self.div(other)
    def __rtruediv__(self, other): return other.div(self)
    def __itruediv__(self, other): return self.assign(self.div(other))

    def __lt__(self, other): return self.less(other)
    def __gt__(self, other): return other.less(self)
    def __ge__(self, other): return 1.0 - (self < other)
    def __le__(self, other): return 1.0 - (self > other)
    def __eq__(self, other): return (self >= other) and (self <= other)
    def __ne__(self, other): return 1.0 - (self == other)

    def __neg__(self): return self.neg()

    def t(self): return self._transpose()
    # fmt: on

    # ---------- Static Method ----------

    @staticmethod
    def where(condition, input, other):
        condition = condition if isinstance(condition, Tensor) else Tensor(condition)
        return condition._where(input, other)

    # ---------- Creation Methods ----------

    @staticmethod
    def manual_seed(seed=0):
        Tensor._seed = seed

    @staticmethod
    def rand(shape: Tuple[int, ...], **kwargs):
        Tensor._seed += 1

        return Tensor(
            LazyBuffer(
                op=LoadOp.RAND,
                view=View.create(shape=shape),
                args=[Tensor._seed],
                device=kwargs.pop("device", DEFAULT_DEVICE),
            ),
            requires_grad=kwargs.pop("requires_grad", False),
            dtype=kwargs.pop("dtype", Tensor.default_type),
        )

    @staticmethod
    def full(shape: Tuple[int, ...], fill_value: Union[float, int], **kwargs):
        return Tensor(
            LazyBuffer(
                LoadOp.FROM_CONST,
                view=View.create(shape),
                args=[fill_value],
                device=kwargs.pop("device", DEFAULT_DEVICE),
            ),
            requires_grad=kwargs.pop("requires_grad", False),
            dtype=kwargs.pop("dtype", Tensor.default_type),
        )

    @staticmethod
    def zeros(shape: Tuple[int, ...], **kwargs):
        return Tensor.full(shape, 0.0, **kwargs)

    @staticmethod
    def ones(shape: Tuple[int, ...], **kwargs):
        return Tensor.full(shape, 1.0, **kwargs)

    def full_like(self, fill_value, **kwargs):
        return Tensor.full(
            self.shape,
            fill_value=fill_value,
            device=kwargs.pop("device", self.device),
            **kwargs,
        )

    def zeros_like(self, **kwargs):
        return self.full_like(0, **kwargs)

    def ones_like(self, **kwargs):
        return self.full_like(1, **kwargs)

    # ---------- Random Methods ----------

    @staticmethod
    def randn(shape: Tuple[int, ...], dtype: Optional[DType] = None, **kwargs):
        # create two uniform distribution
        src = Tensor.rand(shape=tuple([2] + list(shape)), **kwargs)
        # Then apply Muller transform trick (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
        # taken from: https://github.com/tinygrad/teenygrad/blob/main/teenygrad/tensor.py#L693
        lhs = (src[0] * (2 * math.pi)).cos()
        rhs = ((1 - src[1]).log() * (-2)).sqrt()
        return (lhs * rhs).cast(Tensor.default_type if dtype is None else dtype)

    @staticmethod
    def randint(low, high, shape: Tuple[int, ...], **kwargs):
        t = Tensor.rand(shape, **kwargs)
        t = t * (high - low)
        t = t + low
        return t.cast(dtypes.int32)

    @staticmethod
    def normal(shape: Tuple[int, ...], mean=0.0, std=1.0, **kwargs):
        return (std * Tensor.randn(shape, **kwargs)) + mean

    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs):
        dtype = kwargs.pop("dtype", Tensor.default_type)
        return ((high - low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

    # ---------- Other Methods ----------

    def backward(self, retain_graph: bool = False):
        if self._ctx is None:
            return

        if self.grad is None:
            self.grad = Tensor(1.0, device=self.device).expand(self.shape)

        # Topological Sort
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                if t._ctx:
                    for parent in t._ctx.parents:
                        build_topo(parent)
                topo.append(t)

        build_topo(self)

        # Backward pass
        for t in reversed(topo):
            if t.grad is None or t._ctx is None:
                continue

            grads = t._ctx.backward(t.grad.lazydata)
            if not isinstance(grads, tuple):
                grads = (grads,)

            # Accumulate gradients
            for parent, g in zip(t._ctx.parents, grads):
                if g is not None and parent.requires_grad:
                    g_tensor = Tensor(g, device=self.device)
                    if parent.grad is None:
                        parent.grad = g_tensor
                    else:
                        parent.grad = parent.grad + g_tensor

        # Clean up the computation graph
        if not retain_graph:
            for t in topo:
                if hasattr(t, "_ctx"):
                    delattr(t, "_ctx")

    def realize(self) -> "Tensor":
        Device.get_backend(self.lazydata.device)().exec(self.lazydata)
        return self

    def to(self, device: str):
        device = device.upper()
        if device == self.device:
            return self

        # Force realization. You can't move what doesn't exist yet.
        self.realize()

        assert self.lazydata.realized is not None
        ret = Tensor(
            self.lazydata.realized.to(device),
            device=device,
            requires_grad=self.requires_grad,
        )

        if self.grad:
            ret.grad = self.grad.to(device)

        return ret

    def detach(self):
        return Tensor(self.lazydata, device=self.device, requires_grad=False)

    def numpy(self):
        return (
            self.detach()
            .contiguous()
            .to("cpu")
            .realize()
            .lazydata.realized.numpy()  # type: ignore
            .reshape(self.shape)
        )

    def item(self) -> Union[float, int]:
        if len(self.shape) == 1 and self.shape[0] == 1:
            return self.numpy().item()
        raise ValueError(
            "item() method can be used only for Scalar Tensor (with shape=(1))"
        )

    def assign(self, x: Union["Tensor", float]) -> "Tensor":
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device, dtype=self.dtype)

        # Force Realization to break graph dependecy
        x.realize()

        # We take the raw realized buffer from 'x' and wrap it in a new
        # clean LazyBuffer with no parents (src=()).
        # This effectively "forgets" how x was calculated.
        if self.shape != x.shape:
            x = x.reshape(self.shape)

        self.lazydata = LazyBuffer(
            LoadOp.FROM_TORCH,  # note that Pytorch can implicitly handle both CPU and GPU devices mismatch
            src=(),
            view=x.lazydata.view,
            device=self.device,
        )
        self.lazydata.realized = x.lazydata.realized

        return self
