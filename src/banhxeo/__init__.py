import os

import tensorflow as tf

# https://docs.jax.dev/en/latest/gpu_performance_tips.html
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True " "--xla_gpu_enable_latency_hiding_scheduler=true "
)

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")

# Set up seed
DEFAULT_SEED = 1234

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
