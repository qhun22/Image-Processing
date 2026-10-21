import numpy as np


def test_numpy_available():
    arr = np.array([[1, 2], [3, 4]])
    assert arr.shape == (2, 2)


def test_project_imports():
    import algorithms  # noqa: F401
    import core  # noqa: F401
