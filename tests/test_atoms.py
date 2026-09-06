import cvxpy as cp
import numpy as np
import pytest

import dbcp

np.random.seed(10015)


def test_convolve():
    # Test the convolution atom to match np.convolve
    x0 = np.random.randn(10)
    y0 = np.random.randn(20)
    c0 = np.convolve(x0, y0)

    x = cp.Variable(10)
    y = cp.Variable(20)
    c = dbcp.convolve(x, y)

    x.value = x0
    y.value = y0
    assert np.allclose(c.value, c0)


def test_convolve_invalid_input():
    x = cp.Variable((3, 4))
    y = cp.Variable(5)
    with pytest.raises(ValueError):
        dbcp.convolve(x, y)

    x = cp.Variable(6)
    y = cp.Variable((2, 3))
    with pytest.raises(ValueError):
        dbcp.convolve(x, y)
