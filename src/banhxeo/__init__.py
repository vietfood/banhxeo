import os
import warnings

try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "Banhxeo: Pytorch is not installed.\n"
        "Please go to https://pytorch.org/.\n to install Pytorch"
    )


# Set up device
def set_default_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        return None
    return device


GPU_DEVICE = set_default_gpu()
CPU_DEVICE = torch.device("cpu")

# Set up seed
DEFAULT_SEED = 1234

# Copy from: https://github.com/unslothai/unsloth/blob/main/unsloth/__init__.py
# Reduce VRAM usage by reducing fragmentation
# And optimize pinning of memory
if GPU_DEVICE == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
    )

# Install pretty traceback
try:
    from rich.traceback import install

    install(show_locals=True, max_frames=20)
except ModuleNotFoundError:
    warnings.warn(
        "Banhxeo: You can install `rich` library to get prettier console/notebook output"
    )

# Some other things
