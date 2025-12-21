# We see Device as a singleton class
class Device:
    CPU = "CPU"
    CUDA = "CUDA"

    _backends = {}

    @staticmethod
    def register(name: str, backend_cls):
        Device._backends[name] = backend_cls

    @staticmethod
    def get_backend(device_name: str):
        return Device._backends[device_name]


def device_register_global():
    """
    just to avoid circular imports
    """
    from banhxeo.backend import (
        CPUBackend,
        CUDABackend,
    )

    Device.register(Device.CPU, CPUBackend)
    Device.register(Device.CUDA, CUDABackend)


DEFAULT_DEVICE = Device.CPU
