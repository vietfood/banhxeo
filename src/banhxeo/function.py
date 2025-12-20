from __future__ import annotations

from typing import Tuple, Type, Union

from src.banhxeo import Tensor


class Function:
    def __init__(self, device: Union[str, Tuple[str, ...]], *tensors: Tensor):
        self.device = device
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = (
            True
            if any(self.needs_input_grad)
            else None
            if None in self.needs_input_grad
            else False
        )
        if self.requires_grad:
            self.parents = tensors

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"Forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise RuntimeError(f"Backward not implemented for {type(self)}")

    @classmethod
    def apply(func: Type[Function], *x: Tensor, **kwargs) -> Tensor:
        ctx = func(x[0].device, *x)
        ret = Tensor.__new__(Tensor)
        # Initially set gradient to None
        ret.lazydata, ret.requires_grad, ret.grad = (
            # Create lazy graph from ctx
            ctx.forward(*[t.lazydata for t in x], **kwargs),
            ctx.requires_grad,
            None,
        )
        ret._ctx = (
            ctx if ctx.requires_grad and not Tensor.no_grad else None
        )  # used by autograd engine
        return ret
