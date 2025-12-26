from banhxeo import nn
from banhxeo.nn import init


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias=True):
        super().__init__()

        self.weight = init.kaiming_uniform(shape=(out_dim, in_dim), requires_grad=True)

        if bias:
            self.bias = init.kaiming_uniform(shape=(out_dim,), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        # y = x @ W.T + b
        return x @ self.weight.t() + (self.bias if self.bias is not None else 0)


class SGD:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr

    def step(self):
        for t in self.params:
            if t.grad is not None:
                # In-place update: data = data - lr * grad
                # We assume t.grad is calculated
                t.lazydata.realized.data -= (
                    (t.grad * self.lr).realize().lazydata.realized.data
                )

    def zero_grad(self):
        for t in self.params:
            t.grad = None
