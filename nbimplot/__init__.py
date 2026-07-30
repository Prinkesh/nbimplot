"""nbimplot public API."""

from importlib.metadata import PackageNotFoundError, version

from ._plot import AlignedPlots, Dashboard, LineHandle, Plot, Subplots

try:
    __version__ = version("nbimplot")
except PackageNotFoundError:
    __version__ = "0.1.13"

__all__ = ["Plot", "LineHandle", "Subplots", "AlignedPlots", "Dashboard", "__version__"]
