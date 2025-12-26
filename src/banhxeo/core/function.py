import math
from typing import Tuple, Type

from banhxeo.core.buffer import (
    BinaryOp,
    LazyBuffer,
    UnaryOp,
)
from banhxeo.tensor import Tensor


class Function:
    """
    Autograd engine
    * Inspiration from: https://github.com/tinygrad/teenygrad/blob/main/teenygrad/tensor.py#L16
    * A function basically records the context and current Tensor for automatic differentation later
    """

    def __init__(self, device: str, *tensors: Tensor):
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
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(cls: Type["Function"], *x: Tensor, **kwargs) -> Tensor:
        ctx = cls(x[0].device, *x)
        ret = Tensor(
            ctx.forward(*[t.lazydata for t in x], **kwargs),
            device=ctx.device,
            requires_grad=ctx.requires_grad,
        )
        if ctx.requires_grad and not Tensor.no_grad:
            ret._ctx = ctx  # used by autograd engine
        return ret


# ------------ Load Op ------------


class Contiguous(Function):
    def forward(self, x: LazyBuffer):
        return x.contiguous()

    def backward(self, grad_output: LazyBuffer):
        return grad_output


class ContiguousBackward(Function):
    def forward(self, x: LazyBuffer):
        return x

    def backward(self, grad_output: LazyBuffer):
        return grad_output.contiguous()


# ------------ Unary Op ------------


class Neg(Function):
    def forward(self, x: LazyBuffer):
        return x.compute_ops(UnaryOp.NEG)

    def backward(self, grad: LazyBuffer):
        return grad.compute_ops(UnaryOp.NEG)


class Sin(Function):
    def forward(self, x: LazyBuffer):
        self.x = x
        return x.compute_ops(UnaryOp.SIN)

    def backward(self, grad: LazyBuffer):
        return (
            self.x.const(math.pi / 2)
            .compute_ops(BinaryOp.SUB, self.x)
            .compute_ops(UnaryOp.SIN)
            .compute_ops(BinaryOp.MUL, grad)
        )


class Log(Function):
    def forward(self, x: LazyBuffer):
        self.x = x
        return x.compute_ops(UnaryOp.LOG)

    def backward(self, grad: LazyBuffer):
        return grad.compute_ops(BinaryOp.DIV, self.x)


class Exp(Function):
    def forward(self, x: LazyBuffer):
        self.ret = x.compute_ops(UnaryOp.EXP)
        return self.ret

    def backward(self, grad_output: LazyBuffer):
        return self.ret.compute_ops(BinaryOp.MUL, grad_output)


class Sqrt(Function):
    def forward(self, x: LazyBuffer):
        self.ret = x.compute_ops(UnaryOp.SQRT)
        return self.ret

    def backward(self, grad_output: LazyBuffer):
        return grad_output.compute_ops(
            BinaryOp.DIV, self.ret.compute_ops(BinaryOp.MUL, self.ret.const(2))
        )


class Cast(Function):
    def forward(self, x: LazyBuffer, dtype):
        self.input_dtype = x.dtype
        return x.compute_ops(UnaryOp.CAST, args=[dtype])

    def backward(self, grad_output: LazyBuffer):
        return grad_output.compute_ops(UnaryOp.CAST, args=[self.input_dtype])


# ------------ Binary Op ------------


class Less(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        return x.compute_ops(BinaryOp.CMPLT, y)


class Add(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        return x.compute_ops(BinaryOp.ADD, y)

    def backward(self, grad_output: LazyBuffer):
        return grad_output if self.needs_input_grad[
            0
        ] else None, grad_output if self.needs_input_grad[1] else None


class Sub(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        return x.compute_ops(BinaryOp.SUB, y)

    def backward(self, grad_output: LazyBuffer):
        return grad_output if self.needs_input_grad[
            0
        ] else None, grad_output.compute_ops(UnaryOp.NEG) if self.needs_input_grad[
            1
        ] else None


class Mul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        self.x, self.y = x, y
        return x.compute_ops(BinaryOp.MUL, y)

    def backward(self, grad_output: LazyBuffer):
        return self.y.compute_ops(BinaryOp.MUL, grad_output) if self.needs_input_grad[
            0
        ] else None, self.x.compute_ops(
            BinaryOp.MUL, grad_output
        ) if self.needs_input_grad[1] else None


class Div(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        self.x, self.y = x, y
        return x.compute_ops(BinaryOp.DIV, y)

    def backward(self, grad_output: LazyBuffer):
        return grad_output.compute_ops(BinaryOp.DIV, self.y) if self.needs_input_grad[
            0
        ] else None, grad_output.compute_ops(UnaryOp.NEG).compute_ops(
            BinaryOp.MUL, self.x
        ).compute_ops(
            BinaryOp.DIV, self.y.compute_ops(BinaryOp.MUL, self.y)
        ) if self.needs_input_grad[1] else None


class Matmul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        self.x, self.y = x, y
        return x.matmul(y)

    def backward(self, grad_output: LazyBuffer):
        return grad_output.matmul(self.y.t()) if self.needs_input_grad[
            0
        ] else None, self.x.t().matmul(grad_output) if self.needs_input_grad[
            1
        ] else None


# ------------ Ternary Op ------------


class Where(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer, z: LazyBuffer):
        self.x = x
        return x.where(y, z)

    def backward(self, grad_output: LazyBuffer):
        return (
            None,
            self.x.where(grad_output, grad_output.const(0))
            if self.needs_input_grad[1]
            else None,
            self.x.where(grad_output.const(0), grad_output)
            if self.needs_input_grad[2]
            else None,
        )


# ------------ Neural Network Op ------------


class Relu(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.compute_ops(BinaryOp.MAX, x.const(0))
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return (
            self.ret.const(0)
            .compute_ops(BinaryOp.CMPLT, self.ret)
            .compute_ops(BinaryOp.MUL, grad_output)
        )


class Sigmoid(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.const(1).compute_ops(
            BinaryOp.DIV,
            x.const(1).compute_ops(
                BinaryOp.ADD,
                x.compute_ops(UnaryOp.EXP),
            ),
        )
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.compute_ops(
            BinaryOp.MUL, self.ret.const(1).compute_ops(BinaryOp.SUB, self.ret)
        ).compute_ops(BinaryOp.MUL, grad_output)


# ------------ Movement Op ------------


class Reshape(Function):
    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]):
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output: LazyBuffer):
        return grad_output.reshape(self.input_shape)


class Permute(Function):
    def forward(self, x: LazyBuffer, order: Tuple[int, ...]):
        self.input_order = order
        return x.permute(order)

    def backward(self, grad_output: LazyBuffer):
        from banhxeo.utils.helpers import argsort

        return grad_output.permute(argsort(self.input_order))


# ------------ Reduce Op ------------

# TODO
