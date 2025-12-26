import numpy as np

from banhxeo.tensor import Tensor

# ------------ Activation ------------


def relu(input: Tensor) -> Tensor:
    return input.maximum(0.0)


def leaky_relu(input: Tensor, alpha: float = 0.5) -> Tensor:
    return Tensor.where(input < 0.0, alpha * input, input)


def sigmoid(input: Tensor) -> Tensor:
    return 1.0 / (1.0 + (-input).exp())


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    # Numerical stability trick: x - max(x)
    m = input.max(axis=dim, keepdim=True)
    e = (input - m).exp()
    s = e.sum(axis=dim, keepdim=True)
    return e.div(s)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    # log(exp(x - m) / sum(exp(x - m)))
    # = (x - m) - log(sum(exp(x - m)))
    m = input.max(axis=dim, keepdim=True)
    shifted = input - m
    shifted_exp_sum = shifted.exp().sum(axis=dim, keepdim=True)
    return shifted - shifted_exp_sum.log()


# --- Loss Functions ---


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = input - target
    sq_diff = diff * diff

    if reduction == "mean":
        return sq_diff.mean()
    elif reduction == "sum":
        return sq_diff.sum()
    return sq_diff


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    log_probs = log_softmax(input, dim=-1)

    # HACK: Since we lack a GATHER kernel, we use the One-Hot trick for now
    # This moves target to CPU to create mask, then back.
    # Efficient system would use a fused 'nll_loss' kernel.

    batch_size, num_classes = input.shape

    # Create one-hot mask (requires numpy bridge for now)
    y_idx = target.numpy().astype(int)
    mask_np = np.zeros((batch_size, num_classes), dtype=np.float32)
    mask_np[np.arange(batch_size), y_idx] = 1.0

    mask = Tensor(mask_np, device=input.device)

    # Select probabilities: sum(log_probs * mask)
    # The mask is 1 at the correct class, 0 elsewhere
    nll = -(log_probs * mask).sum(axis=-1)

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll
