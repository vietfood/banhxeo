import logging
import os

import jax
import tensorflow as tf

# Set up seed
DEFAULT_SEED = 1234
USE_TORCH = True

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")
tf.random.set_seed(DEFAULT_SEED)

# Open 64bit mode
jax.config.update("jax_enable_x64", True)

# Enable debug NaN
jax.config.update("jax_debug_nans", True)

# Disable debug message
logger = logging.getLogger("jax._src.xla_bridge")
logger.setLevel(logging.ERROR)

# https://docs.jax.dev/en/latest/gpu_performance_tips.html
if jax.default_backend() == "gpu" and len(jax.devices("gpu")) > 0:
    print("GPU detected. Setting XLA_FLAGS for GPU performance.")
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )

# Copy from: https://github.com/unslothai/unsloth/blob/main/unsloth/__init__.py
# Hugging Face Hub faster downloads
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
pass

# XET is slower in Colab - investigate why
keynames = "\n" + "\n".join(os.environ.keys())
if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
pass
if "\nCOLAB_" in keynames:
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
pass
