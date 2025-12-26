from typing import List

from banhxeo.tensor import Tensor


class Module:
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        # We wrap forward to add hooks later if needed
        return self.forward(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        # Recursively set training mode for sub-modules
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.train(mode)
            elif isinstance(attr, (list, tuple)):
                for x in attr:
                    if isinstance(x, Module):
                        x.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self) -> List[Tensor]:
        """
        Recursively find all Tensors that require gradients.
        """
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    params.append(value)
            elif isinstance(value, Module):
                # Recurse into sub-modules
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                # Handle lists of layers (like Sequential)
                for x in value:
                    if isinstance(x, Tensor) and x.requires_grad:
                        params.append(x)
                    elif isinstance(x, Module):
                        params.extend(x.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self, prefix=""):
        state = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                state[prefix + name] = value
            elif isinstance(value, Module):
                state.update(value.state_dict(prefix + name + "."))
        return state


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
