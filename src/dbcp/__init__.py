from .atoms import convolve
from .problem import BiconvexProblem, BiconvexRelaxProblem

__all__ = ["BiconvexProblem", "BiconvexRelaxProblem", "convolve", "__version__"]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("dbcp")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
