"""nbimplot public API."""

from importlib.metadata import PackageNotFoundError, version

from ._plot import AlignedPlots, LineHandle, Plot, Subplots

try:
    __version__ = version("nbimplot")
except PackageNotFoundError:
    __version__ = "0.1.9"

__all__ = ["Plot", "LineHandle", "Subplots", "AlignedPlots", "__version__"]
