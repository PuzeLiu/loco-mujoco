from types import ModuleType

import numpy as np
import jax.numpy as jnp


def assert_backend_is_supported(module: ModuleType):
    """
    Check if the given module is supported.

    Args:
        module (ModuleType): The module to check (e.g., numpy or jax.numpy).

    Returns:
        bool: True if the module is supported, False otherwise.
    """
    is_supporter = module in {np, jnp}
    assert is_supporter, f"Unsupported backend module: {module.__name__}"

def resolve_dtype(dtype):
    if dtype is None:
        return jnp.float32
    if isinstance(dtype, str):
        name = dtype.lower()
        if name in ("bf16", "bfloat16"):
            return jnp.bfloat16
        if name in ("fp16", "float16"):
            return jnp.float16
        if name in ("fp32", "float32"):
            return jnp.float32
    return dtype