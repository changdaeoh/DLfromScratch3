import numpy as np
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
from dezero import Variable


def get_array_module(x):
    """Returns the array module for `x`.

    Args:
        x (dezero.Variable or numpy.ndarray or cupy.ndarray): Values to
            determine whether NumPy or CuPy should be used.

    Returns:
        module: `cupy` or `numpy` is returned based on the argument.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp
