from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from banhxeo.core.buffer import (
    LazyBuffer,
    LoadOp,
)
from banhxeo.core.device import DEFAULT_DEVICE, Device
from banhxeo.core.view import View


class Tensor:
    __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
    __deletable__ = ("_ctx",)
    training: ClassVar[bool] = False
    no_grad: ClassVar[bool] = False

    def __init__(
        self,
        data: Optional[
            Union[LazyBuffer, List, Tuple, np.ndarray, torch.Tensor, int, float]
        ] = None,
        device: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        requires_grad: Optional[bool] = None,
    ):
        if device is None:
            device = DEFAULT_DEVICE
        device = device.upper()
        self.requires_grad = False

        if data is None:
            assert shape is not None, "Cannot allocate empty Tensor without shape"
            self.lazydata = LazyBuffer(
                LoadOp.FROM_NONE, view=View.create(shape=shape), device=device
            )
        else:
            if isinstance(data, LazyBuffer):
                self.lazydata = data
            elif isinstance(data, (int, float)):
                self.lazydata = LazyBuffer(
                    LoadOp.FROM_CONST,
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
    def device(self):
        return self.lazydata.device

    @property
    def shape(self):
        return self.lazydata.shape

    # ---------- Binary Ops ----------

    def add(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Add

        return Add.apply(x, y)

    def mul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Mul

        return Mul.apply(x, y)

    def sub(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Sub

        return Sub.apply(x, y)

    def less(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Less

        return Less.apply(x, y)

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        from banhxeo.core.function import Matmul

        return Matmul.apply(self, other)

    def div(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x, y = self._broadcasted(other)

        from banhxeo.core.function import Div

        return Div.apply(x, y)

    # ---------- Unary Ops ----------

    def log(self):
        from banhxeo.core.function import Log

        return Log.apply(self)

    def exp(self):
        from banhxeo.core.function import Exp

        return Exp.apply(self)

    def sin(self):
        from banhxeo.core.function import Sin

        return Sin.apply(self)

    def sqrt(self):
        from banhxeo.core.function import Sqrt

        return Sqrt.apply(self)

    def neg(self):
        from banhxeo.core.function import Neg

        return Neg.apply(self)

    # ---------- Ternary Ops ----------

    def _where(self, input, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        input = input if isinstance(input, Tensor) else Tensor(input)
        return Tensor(self.lazydata.where(input.lazydata, other.lazydata))

    # ---------- Load Ops ----------

    def contiguous(self):
        return Tensor(self.lazydata.contiguous())

    # ---------- Movement Ops ----------

    def _broadcasted(self, other: "Tensor") -> Tuple["Tensor", "Tensor"]:
        x, y = self.lazydata.broadcasted(other.lazydata)
        return Tensor(x), Tensor(y)

    def reshape(self, new_shape: Tuple[int, ...]):
        return Tensor(self.lazydata.reshape(new_shape))

    def permute(self, new_axis: Tuple[int, ...]):
        return Tensor(self.lazydata.permute(new_axis))

    def slice(self, args: Tuple[Tuple[int, ...], ...]):
        return Tensor(self.lazydata.slice(args))

    def expand(self, shape: Tuple[int, ...]):
        return Tensor(self.lazydata.expand(shape))

    def _transpose(self):
        assert self.shape == 2, (
            "Transpose only works with 2 dimension, please use Permute for more than 2 dimensions"
        )
        return self.permute((1, 0))

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

    def __neg__(self):
        return self.neg()

    def __div__(self, other):
        return self.div(other)

    def t(self):
        """
        Transpose 2D Tensor
        """
        return self._transpose()

    # ---------- Neural Network Method ----------

    def relu(self):
        # relu(x) = where(x < 0, 0, x)
        return Tensor.where(self < 0, 0, self)

    def leaky_relu(self, alpha):
        return Tensor.where(self < 0, alpha * self, self)

    # ---------- Static Method ----------

    @staticmethod
    def where(condition, input, other):
        condition = condition if isinstance(condition, Tensor) else Tensor(condition)
        return condition._where(input, other)

    # ---------- Other Methods ----------
    def backward(self):
        if self._ctx is None:
            return

        # initialize gradient at the root (1.0)
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

        # Then traverse the linear graph in reverse order
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

    def realize(self) -> "Tensor":
        Device.get_backend(self.lazydata.device)().exec(self.lazydata)
        return self

    def numpy(self):
        if self.lazydata.realized is None:
            self.realize()
        else:
            return self.lazydata.realized.to_cpu().numpy()
