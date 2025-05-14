import torch as t


def set_default_gpu():
    if t.cuda.is_available():
        device = t.device("cuda")
        print(f"Using CUDA: {t.cuda.get_device_name(device)}")
    elif t.backends.mps.is_available():
        device = t.device("mps")
        print("Using MPS device")
    else:
        return None
    return device


GPU_DEVICE = set_default_gpu()
CPU_DEVICE = t.device("cpu")

DEFAULT_SEED = 1234
