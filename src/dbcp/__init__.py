from .atoms import convolve
from .problem import BiconvexProblem
from .problem import BiconvexRelaxProblem as BiconvexRelaxProblem

__all__ = ["BiconvexProblem", "convolve", "__version__"]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("dbcp")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
