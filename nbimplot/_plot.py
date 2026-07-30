from __future__ import annotations

from dataclasses import dataclass
import inspect
import itertools
import json
import pathlib
from typing import Any, Callable

import anywidget
from IPython.display import display
import numpy as np
import traitlets

_ESM_PATH = pathlib.Path(__file__).with_name("_frontend.js")
_WASM_DIR = pathlib.Path(__file__).with_name("wasm")
_WASM_JS_PATH = _WASM_DIR / "nbimplot_wasm.js"
_WASM_BIN_PATH = _WASM_DIR / "nbimplot_wasm.wasm"
_WASM_ASSET_CACHE: tuple[str | None, bytes | None, str | None] | None = None

_PLOT_FLAG_NO_LEGEND = 1 << 0
_PLOT_FLAG_NO_MENUS = 1 << 1
_PLOT_FLAG_NO_BOX_SELECT = 1 << 2
_PLOT_FLAG_NO_MOUSE_POS = 1 << 3
_PLOT_FLAG_CROSSHAIRS = 1 << 4
_PLOT_FLAG_EQUAL = 1 << 5

_SUBPLOT_FLAG_NO_LEGEND = 1 << 0
_SUBPLOT_FLAG_NO_MENUS = 1 << 1
_SUBPLOT_FLAG_NO_RESIZE = 1 << 2
_SUBPLOT_FLAG_NO_ALIGN = 1 << 3
_SUBPLOT_FLAG_SHARE_ITEMS = 1 << 4
_SUBPLOT_FLAG_LINK_ROWS = 1 << 5
_SUBPLOT_FLAG_LINK_COLS = 1 << 6
_SUBPLOT_FLAG_LINK_ALL_X = 1 << 7
_SUBPLOT_FLAG_LINK_ALL_Y = 1 << 8
_SUBPLOT_FLAG_COL_MAJOR = 1 << 9

_SERIES_MARKER_NONE = -2
_SERIES_MARKER_AUTO = -1
_SERIES_MARKER_MAP: dict[str, int] = {
    "none": _SERIES_MARKER_NONE,
    "auto": _SERIES_MARKER_AUTO,
    "circle": 0,
    "square": 1,
    "diamond": 2,
    "up": 3,
    "down": 4,
    "left": 5,
    "right": 6,
    "cross": 7,
    "plus": 8,
    "asterisk": 9,
}


class _UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


_UNSET = _UnsetType()

_COLORMAP_CANONICAL_NAMES: dict[str, str] = {
    "deep": "Deep",
    "dark": "Dark",
    "pastel": "Pastel",
    "paired": "Paired",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "hot": "Hot",
    "cool": "Cool",
    "pink": "Pink",
    "jet": "Jet",
    "twilight": "Twilight",
    "rdbu": "RdBu",
    "brbg": "BrBG",
    "piyg": "PiYG",
    "spectral": "Spectral",
    "greys": "Greys",
}


@dataclass(slots=True)
class _SeriesMeta:
    name: str
    length: int
    dtype: str
    data: np.ndarray
    x_data: np.ndarray | None = None
    subplot_index: int = 0
    x_axis: int = 0
    y_axis: int = 3
    color: tuple[float, float, float, float] | None = None
    line_weight: float = 1.0
    marker: int = _SERIES_MARKER_NONE
    marker_size: float = 4.0
    hidden: bool = False
    stream_capacity: int | None = None
    stream_paused: bool = False
    stream_auto_render: bool = False
    stream_autoscale_y: bool = False


@dataclass(slots=True)
class _PrimitiveMeta:
    content: dict[str, Any]
    buffers: list[np.ndarray]


def _get_wasm_assets() -> tuple[str | None, bytes | None, str | None]:
    global _WASM_ASSET_CACHE
    if _WASM_ASSET_CACHE is not None:
        return _WASM_ASSET_CACHE

    try:
        js_source = _WASM_JS_PATH.read_text(encoding="utf-8")
        wasm_bytes = _WASM_BIN_PATH.read_bytes()
        if not js_source.strip():
            _WASM_ASSET_CACHE = (None, None, f"Empty WASM JS loader: {_WASM_JS_PATH}")
            return _WASM_ASSET_CACHE
        if len(wasm_bytes) == 0:
            _WASM_ASSET_CACHE = (None, None, f"Empty WASM binary: {_WASM_BIN_PATH}")
            return _WASM_ASSET_CACHE
        _WASM_ASSET_CACHE = (js_source, wasm_bytes, None)
        return _WASM_ASSET_CACHE
    except OSError as exc:
        _WASM_ASSET_CACHE = (None, None, f"Unable to load WASM assets: {exc!s}")
        return _WASM_ASSET_CACHE


def _mime_requested(mimetype: str, kwargs: dict[str, Any]) -> bool:
    include = kwargs.get("include")
    exclude = kwargs.get("exclude")
    if include is not None and mimetype not in include:
        return False
    if exclude is not None and mimetype in exclude:
        return False
    return True


def _with_text_plain(
    bundle: tuple[dict[str, Any], dict[str, Any]] | None,
    text: str,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if bundle is None:
        return None
    data, metadata = bundle
    data = dict(data)
    if _mime_requested("text/plain", kwargs):
        data["text/plain"] = text
    else:
        data.pop("text/plain", None)
    return data, metadata


def _is_dataframe_like(data: Any) -> bool:
    if data is _UNSET or isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
        return False
    return hasattr(data, "columns") and hasattr(data, "__getitem__")


def _is_column_selector(value: Any) -> bool:
    return isinstance(value, (str, int, np.integer))


def _column_names(data: Any) -> list[str]:
    try:
        columns = list(getattr(data, "columns"))
    except TypeError:
        return []
    return [str(column) for column in columns]


def _has_column(data: Any, key: Any) -> bool:
    try:
        return bool(key in getattr(data, "columns"))
    except TypeError:
        return False


def _dataframe_column(data: Any, key: Any, *, arg_name: str) -> Any:
    if not _is_dataframe_like(data):
        raise TypeError(f"{arg_name} column selection requires a dataframe-like positional argument.")
    if not _is_column_selector(key):
        raise TypeError(f"{arg_name} column selector must be a string or integer column name.")
    if not _has_column(data, key):
        available = ", ".join(_column_names(data)[:12])
        suffix = f" Available columns: {available}." if available else ""
        raise KeyError(f"Unknown {arg_name} column: {key!r}.{suffix}")
    try:
        return data[key]
    except Exception as exc:  # pragma: no cover - defensive for dataframe implementations.
        raise KeyError(f"Unable to read {arg_name} column {key!r}.") from exc


def _resolve_y_input(data: Any, y: Any = _UNSET, *, arg_name: str = "y") -> Any:
    if y is _UNSET:
        if data is _UNSET:
            raise TypeError(f"{arg_name} data is required.")
        if _is_dataframe_like(data):
            raise ValueError(f"{arg_name} column must be provided when the data argument is a dataframe.")
        return data
    if data is _UNSET:
        return y
    if not _is_dataframe_like(data):
        raise TypeError(f"{arg_name}= is only valid with a dataframe-like positional argument.")
    return _dataframe_column(data, y, arg_name=arg_name)


def _resolve_maybe_column(data: Any, value: Any, *, arg_name: str) -> Any:
    if value is _UNSET or value is None:
        return value
    if _is_dataframe_like(data) and _is_column_selector(value):
        return _dataframe_column(data, value, arg_name=arg_name)
    return value


def _resolve_xy_inputs(
    data: Any,
    *,
    x: Any | None = None,
    y: Any = _UNSET,
    y_arg_name: str = "y",
) -> tuple[Any | None, Any]:
    return (
        _resolve_maybe_column(data, x, arg_name="x") if x is not None else None,
        _resolve_y_input(data, y, arg_name=y_arg_name),
    )


def _to_float32_1d(data: Any, *, arg_name: str) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError(f"{arg_name} must be a 1D array, got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{arg_name} must not be empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{arg_name} must contain numeric values, got {arr.dtype!s}.")

    out = np.ascontiguousarray(arr, dtype=np.float32)
    if out.dtype.byteorder == ">":
        out = out.byteswap().view(out.dtype.newbyteorder("<"))
    return out


def _to_float32_2d(data: Any, *, arg_name: str) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"{arg_name} must be a 2D array, got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{arg_name} must not be empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{arg_name} must contain numeric values, got {arr.dtype!s}.")

    out = np.ascontiguousarray(arr, dtype=np.float32)
    if out.dtype.byteorder == ">":
        out = out.byteswap().view(out.dtype.newbyteorder("<"))
    return out


def _to_image_float32_flat(data: Any, *, arg_name: str) -> tuple[np.ndarray, int, int, int]:
    arr = np.asarray(data)
    if arr.ndim == 2:
        rows, cols = int(arr.shape[0]), int(arr.shape[1])
        channels = 1
    elif arr.ndim == 3 and int(arr.shape[2]) in {3, 4}:
        rows, cols = int(arr.shape[0]), int(arr.shape[1])
        channels = int(arr.shape[2])
    else:
        raise ValueError(
            f"{arg_name} must be a 2D array or a 3D array with 3/4 channels, got shape {arr.shape!r}."
        )
    if arr.size == 0:
        raise ValueError(f"{arg_name} must not be empty.")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{arg_name} has invalid shape {arr.shape!r}.")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{arg_name} must contain numeric values, got {arr.dtype!s}.")

    out = np.ascontiguousarray(arr, dtype=np.float32)
    if out.dtype.byteorder == ">":
        out = out.byteswap().view(out.dtype.newbyteorder("<"))
    return out.reshape(-1), rows, cols, channels


def _to_float2(value: Any, *, arg_name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError(f"{arg_name} must be a 2-length sequence.")
    if len(value) != 2:
        raise ValueError(f"{arg_name} must have exactly 2 elements.")
    x = float(value[0])
    y = float(value[1])
    if not np.isfinite([x, y]).all():
        raise ValueError(f"{arg_name} values must be finite.")
    return x, y


def _normalize_rgba(color: Any) -> tuple[float, float, float, float] | None:
    if color is None:
        return None
    if isinstance(color, str):
        text = color.strip()
        if not text:
            return None
        if text.startswith("#"):
            hexv = text[1:]
            if len(hexv) == 3:
                hexv = "".join(ch * 2 for ch in hexv) + "ff"
            elif len(hexv) == 4:
                hexv = "".join(ch * 2 for ch in hexv)
            elif len(hexv) == 6:
                hexv += "ff"
            elif len(hexv) != 8:
                raise ValueError("color hex must be #RGB, #RGBA, #RRGGBB, or #RRGGBBAA.")
            try:
                rgba = tuple(int(hexv[i : i + 2], 16) / 255.0 for i in range(0, 8, 2))
            except ValueError as exc:
                raise ValueError("color hex contains non-hex characters.") from exc
            return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
        raise ValueError("color string must be a hex value like '#3b82f6' or '#3b82f680'.")
    if isinstance(color, (tuple, list, np.ndarray)):
        vals = tuple(float(v) for v in color)
        if len(vals) == 3:
            vals = (vals[0], vals[1], vals[2], 1.0)
        if len(vals) != 4:
            raise ValueError("color sequence must have 3 or 4 values.")
        if not np.isfinite(vals).all():
            raise ValueError("color values must be finite.")
        if any(v < 0.0 or v > 1.0 for v in vals):
            raise ValueError("color values must be in [0, 1].")
        return vals
    raise TypeError("color must be None, hex string, or a 3/4-length float sequence.")


def _parse_marker(marker: str) -> int:
    key = str(marker).strip().lower()
    if key not in _SERIES_MARKER_MAP:
        allowed = ", ".join(sorted(_SERIES_MARKER_MAP))
        raise ValueError(f"marker must be one of: {allowed}.")
    return _SERIES_MARKER_MAP[key]


def _normalize_line_weight(value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError("line_weight must be finite and > 0.")
    return out


def _normalize_marker_size(value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError("marker_size must be finite and > 0.")
    return out


def _to_line_x(data: Any, *, y_len: int) -> np.ndarray:
    out = _to_float32_1d(data, arg_name="x")
    if int(out.size) != int(y_len):
        raise ValueError("x and y must have the same length.")
    if not np.isfinite(out).all():
        raise ValueError("x must contain only finite values.")
    if out.size > 1 and np.any(np.diff(out) < 0):
        raise ValueError("x must be sorted in non-decreasing order for line LOD.")
    return out


class LineHandle:
    """Mutable reference to an existing line series."""

    def __init__(
        self,
        plot: "Plot",
        series_id: str,
        name: str,
        *,
        stream_capacity: int | None = None,
        auto_render: bool = False,
        autoscale_y: bool = False,
    ) -> None:
        self._plot = plot
        self._series_id = series_id
        self._name = name
        self._stream_capacity = stream_capacity
        self._paused = False
        self._auto_render = bool(auto_render)
        self._autoscale_y = bool(autoscale_y)

    @property
    def series_id(self) -> str:
        return self._series_id

    @property
    def name(self) -> str:
        return self._name

    def set_data(self, data: Any = _UNSET, *, x: Any | None = None, y: Any = _UNSET) -> None:
        self._plot._set_series_data(self._series_id, data, x=x, y=y)

    def append(self, y: Any, *, x: Any | None = None, max_points: int | None = None) -> None:
        if self._paused:
            return None
        cap = self._stream_capacity if max_points is None else max_points
        self._plot._append_series_data(self._series_id, y, x=x, max_points=cap)
        if self._autoscale_y:
            self._plot.autoscale()
        elif self._auto_render:
            self._plot.render()
        return None

    def pause(self) -> None:
        self._paused = True
        self._plot._set_stream_paused(self._series_id, True)

    def resume(self) -> None:
        self._paused = False
        self._plot._set_stream_paused(self._series_id, False)

    def clear(self, *, x0: float = 0.0, y0: float = 0.0) -> None:
        self._plot._clear_series_data(self._series_id, x0=x0, y0=y0)

    def set_window(self, capacity: int) -> None:
        cap = int(capacity)
        if cap <= 0:
            raise ValueError("capacity must be > 0.")
        self._stream_capacity = cap
        self._plot._set_stream_capacity(self._series_id, cap)

    def set_stream_options(
        self,
        *,
        auto_render: bool | None = None,
        autoscale_y: bool | None = None,
    ) -> None:
        if auto_render is not None:
            self._auto_render = bool(auto_render)
        if autoscale_y is not None:
            self._autoscale_y = bool(autoscale_y)
        self._plot._set_stream_options(
            self._series_id,
            auto_render=self._auto_render,
            autoscale_y=self._autoscale_y,
        )

    def set_style(
        self,
        *,
        color: Any = _UNSET,
        line_weight: float | None = None,
        marker: str | None = None,
        marker_size: float | None = None,
    ) -> None:
        self._plot._set_series_style(
            self._series_id,
            color=color,
            line_weight=line_weight,
            marker=marker,
            marker_size=marker_size,
        )


class Plot(anywidget.AnyWidget):
    """Notebook-only plot widget with binary line uploads."""

    _esm = _ESM_PATH

    width = traitlets.Int(900).tag(sync=True)
    height = traitlets.Int(450).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    prefer_implot = traitlets.Bool(True).tag(sync=True)
    strict_wasm = traitlets.Bool(True).tag(sync=True)

    def __init__(
        self,
        *,
        width: int = 900,
        height: int = 450,
        title: str = "",
        colormap: str | None = None,
        prefer_implot: bool = True,
        strict_wasm: bool = True,
        no_legend: bool = False,
        no_menus: bool = False,
        no_box_select: bool = False,
        no_mouse_pos: bool = False,
        crosshairs: bool = False,
        equal: bool = False,
        axis_scale_x: str = "linear",
        axis_scale_y: str = "linear",
        subplot_rows: int = 1,
        subplot_cols: int = 1,
        subplot_flags: int = 0,
        **kwargs: Any,
    ) -> None:
        if not prefer_implot:
            raise ValueError("prefer_implot=False is not supported; ImPlot is always on.")
        if not strict_wasm:
            raise ValueError("strict_wasm=False is not supported; JS fallback is disabled.")
        super().__init__(
            width=width,
            height=height,
            title=title,
            prefer_implot=True,
            strict_wasm=True,
            **kwargs,
        )
        self._series: dict[str, _SeriesMeta] = {}
        self._series_counter = itertools.count(1)
        self._primitives: dict[str, _PrimitiveMeta] = {}
        self._primitive_counter = itertools.count(1)
        self._view_callbacks: list[Callable[[Plot, dict[str, float]], None]] = []
        self._perf_callbacks: list[Callable[[Plot, dict[str, float]], None]] = []
        self._tool_callbacks: list[Callable[[Plot, dict[str, Any]], None]] = []
        self._selection_callbacks: list[Callable[[Plot, dict[str, Any]], None]] = []
        self._hover_callbacks: list[Callable[[Plot, dict[str, Any]], None]] = []
        self._click_callbacks: list[Callable[[Plot, dict[str, Any]], None]] = []
        self._perf_reporting_enabled = False
        self._perf_interval_ms = 500
        self._plot_flags = 0
        self._axis_scale_x = 0
        self._axis_scale_y = 0
        self._colormap_name = ""
        self._axis_state: dict[int, tuple[bool, int]] = {
            0: (True, 0),
            1: (False, 0),
            2: (False, 0),
            3: (True, 0),
            4: (False, 0),
            5: (False, 0),
        }
        self._axis_labels: dict[int, str] = {}
        self._axis_formats: dict[int, str] = {}
        self._axis_ticks: dict[int, tuple[np.ndarray, list[str], bool]] = {}
        self._axis_limits_constraints: dict[int, tuple[bool, float, float]] = {}
        self._axis_zoom_constraints: dict[int, tuple[bool, float, float]] = {}
        self._axis_links: dict[int, int] = {}
        self._subplot_rows = 1
        self._subplot_cols = 1
        self._subplot_flags = 0
        self._aligned_group_id = ""
        self._aligned_vertical = True
        self._aligned_enabled = False
        self._linked_crosshair: dict[str, Any] = {
            "enabled": False,
            "group_id": "",
            "axis": "x",
        }
        self._theme_name = ""
        self._last_view: dict[str, float] | None = None
        self._hide_next_item = False
        self._closed = False
        self.set_plot_flags(
            no_legend=no_legend,
            no_menus=no_menus,
            no_box_select=no_box_select,
            no_mouse_pos=no_mouse_pos,
            crosshairs=crosshairs,
            equal=equal,
        )
        self.set_axis_scale(x=axis_scale_x, y=axis_scale_y)
        self.set_subplots_config(
            rows=int(subplot_rows),
            cols=int(subplot_cols),
            flags=int(subplot_flags),
        )
        self.set_colormap(colormap)
        self.on_msg(self._handle_frontend_message)

    def line(
        self,
        name: str,
        data: Any = _UNSET,
        *,
        x: Any | None = None,
        y: Any = _UNSET,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
        color: Any = None,
        line_weight: float = 1.0,
        marker: str = "none",
        marker_size: float = 4.0,
        max_points: int | None = None,
    ) -> LineHandle:
        self._ensure_open()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        subplot_idx = self._validate_subplot_index(subplot_index)
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)

        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        data = _to_float32_1d(y_values, arg_name="y")
        x_data = _to_line_x(x_values, y_len=int(data.size)) if x_values is not None else None
        capacity: int | None = None
        if max_points is not None:
            capacity_i = int(max_points)
            if capacity_i <= 0:
                raise ValueError("max_points must be > 0 when provided.")
            capacity = capacity_i
            if data.size > capacity_i:
                data = data[-capacity_i:].copy()
                if x_data is not None:
                    x_data = x_data[-capacity_i:].copy()
        hidden = bool(self._consume_hide_next_item())
        color_rgba = _normalize_rgba(color)
        line_weight_v = _normalize_line_weight(line_weight)
        marker_v = _parse_marker(marker)
        marker_size_v = _normalize_marker_size(marker_size)
        series_id = f"s{next(self._series_counter)}"
        self._series[series_id] = _SeriesMeta(
            name=name,
            length=int(data.size),
            dtype="float32",
            data=data,
            x_data=x_data,
            subplot_index=subplot_idx,
            x_axis=int(x_axis_code),
            y_axis=int(y_axis_code),
            color=color_rgba,
            line_weight=line_weight_v,
            marker=marker_v,
            marker_size=marker_size_v,
            hidden=hidden,
            stream_capacity=capacity,
        )
        color_r, color_g, color_b, color_a = (
            color_rgba if color_rgba is not None else (0.0, 0.0, 0.0, 0.0)
        )
        self.send(
            {
                "type": "line",
                "series_id": series_id,
                "name": name,
                "subplot_index": subplot_idx,
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
                "dtype": "float32",
                "length": int(data.size),
                "has_x": bool(x_data is not None),
                "has_color": bool(color_rgba is not None),
                "color_r": float(color_r),
                "color_g": float(color_g),
                "color_b": float(color_b),
                "color_a": float(color_a),
                "line_weight": float(line_weight_v),
                "marker": int(marker_v),
                "marker_size": float(marker_size_v),
                "hidden": bool(hidden),
                "max_points": 0 if capacity is None else int(capacity),
            },
            buffers=[memoryview(x_data), memoryview(data)] if x_data is not None else [memoryview(data)],
        )
        return LineHandle(
            self,
            series_id=series_id,
            name=name,
            stream_capacity=capacity,
        )

    def stream_line(
        self,
        name: str,
        *,
        capacity: int,
        initial: Any | None = None,
        initial_x: Any | None = None,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
        color: Any = None,
        line_weight: float = 1.0,
        marker: str = "none",
        marker_size: float = 4.0,
        auto_render: bool = False,
        autoscale_y: bool = False,
    ) -> LineHandle:
        cap = int(capacity)
        if cap <= 0:
            raise ValueError("capacity must be > 0.")
        if initial is None:
            initial_data = np.zeros(1, dtype=np.float32)
        else:
            initial_data = _to_float32_1d(initial, arg_name="initial")
        handle = self.line(
            name,
            initial_data,
            x=initial_x,
            subplot_index=subplot_index,
            x_axis=x_axis,
            y_axis=y_axis,
            color=color,
            line_weight=line_weight,
            marker=marker,
            marker_size=marker_size,
            max_points=cap,
        )
        handle.set_stream_options(auto_render=auto_render, autoscale_y=autoscale_y)
        return handle

    def scatter(
        self,
        name: str,
        data: Any = _UNSET,
        *,
        x: Any | None = None,
        y: Any = _UNSET,
        size: float = 2.0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        self._send_xy_primitive(
            "scatter",
            name=name,
            y=y_values,
            x=x_values,
            size=float(size),
            subplot_index=subplot_index,
            x_axis=x_axis,
            y_axis=y_axis,
        )

    def bubbles(
        self,
        name: str,
        data: Any = _UNSET,
        sizes: Any = _UNSET,
        *,
        x: Any | None = None,
        y: Any = _UNSET,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        if sizes is _UNSET:
            raise TypeError("sizes data is required.")
        size_values = _resolve_maybe_column(data, sizes, arg_name="sizes")
        values_y = _to_float32_1d(y_values, arg_name="y")
        values_sizes = _to_float32_1d(size_values, arg_name="sizes")
        if values_y.size != values_sizes.size:
            raise ValueError("y and sizes must have the same length.")
        has_x = x_values is not None
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        buffers: list[np.ndarray] = [values_y, values_sizes]
        if has_x:
            values_x = _to_float32_1d(x_values, arg_name="x")
            if values_x.size != values_y.size:
                raise ValueError("x, y, and sizes must have the same length.")
            buffers.insert(0, values_x)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "bubbles",
                "name": name,
                "length": int(values_y.size),
                "has_x": bool(has_x),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            buffers,
        )

    def stairs(
        self,
        name: str,
        data: Any = _UNSET,
        *,
        x: Any | None = None,
        y: Any = _UNSET,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        self._send_xy_primitive(
            "stairs",
            name=name,
            y=y_values,
            x=x_values,
            subplot_index=subplot_index,
            x_axis=x_axis,
            y_axis=y_axis,
        )

    def stems(
        self,
        name: str,
        data: Any = _UNSET,
        *,
        x: Any | None = None,
        y: Any = _UNSET,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        self._send_xy_primitive(
            "stems",
            name=name,
            y=y_values,
            x=x_values,
            subplot_index=subplot_index,
            x_axis=x_axis,
            y_axis=y_axis,
        )

    def digital(
        self,
        name: str,
        data: Any = _UNSET,
        *,
        x: Any | None = None,
        y: Any = _UNSET,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        self._send_xy_primitive(
            "digital",
            name=name,
            y=y_values,
            x=x_values,
            subplot_index=subplot_index,
            x_axis=x_axis,
            y_axis=y_axis,
        )

    def bars(
        self,
        name: str,
        data: Any = _UNSET,
        *,
        x: Any | None = None,
        y: Any = _UNSET,
        bar_width: float = 0.67,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        self._send_xy_primitive(
            "bars",
            name=name,
            y=y_values,
            x=x_values,
            bar_width=float(bar_width),
            subplot_index=subplot_index,
            x_axis=x_axis,
            y_axis=y_axis,
        )

    def bar_groups(
        self,
        labels: list[str] | tuple[str, ...],
        values: Any,
        *,
        group_size: float = 0.67,
        shift: float = 0.0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        if not labels:
            raise ValueError("labels must not be empty.")
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        values_2d = _to_float32_2d(values, arg_name="values")
        item_count, group_count = int(values_2d.shape[0]), int(values_2d.shape[1])
        if len(labels) != item_count:
            raise ValueError("labels length must equal values.shape[0].")
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "bar_groups",
                "labels": [str(s) for s in labels],
                "item_count": item_count,
                "group_count": group_count,
                "group_size": float(group_size),
                "shift": float(shift),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [values_2d.reshape(-1)],
        )

    def bars_h(
        self,
        name: str,
        data: Any = _UNSET,
        *,
        x: Any = _UNSET,
        y: Any | None = None,
        bar_height: float = 0.67,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        if x is _UNSET:
            if data is _UNSET:
                raise TypeError("x data is required.")
            if _is_dataframe_like(data):
                raise ValueError("x column must be provided when the data argument is a dataframe.")
            x_values = data
        else:
            x_values = _resolve_maybe_column(data, x, arg_name="x")
        y_values = _resolve_maybe_column(data, y, arg_name="y") if y is not None else None
        values_x = _to_float32_1d(x_values, arg_name="x")
        values_y = (
            _to_float32_1d(y_values, arg_name="y")
            if y_values is not None
            else np.arange(values_x.size, dtype=np.float32)
        )
        if values_x.size != values_y.size:
            raise ValueError("x and y must have the same length.")
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "bars_h",
                "name": name,
                "length": int(values_x.size),
                "bar_height": float(bar_height),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [values_x, values_y],
        )

    def shaded(
        self,
        name: str,
        data: Any = _UNSET,
        y2: Any = _UNSET,
        *,
        x: Any | None = None,
        y1: Any = _UNSET,
        alpha: float = 0.2,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        if y1 is _UNSET:
            if data is _UNSET:
                raise TypeError("y1 data is required.")
            if _is_dataframe_like(data):
                raise ValueError("y1 column must be provided when the data argument is a dataframe.")
            y1_values = data
        else:
            y1_values = _resolve_maybe_column(data, y1, arg_name="y1")
        if y2 is _UNSET:
            raise TypeError("y2 data is required.")
        y2_values = _resolve_maybe_column(data, y2, arg_name="y2")
        x_values = _resolve_maybe_column(data, x, arg_name="x") if x is not None else None
        values_y1 = _to_float32_1d(y1_values, arg_name="y1")
        values_y2 = _to_float32_1d(y2_values, arg_name="y2")
        if values_y1.size != values_y2.size:
            raise ValueError("y1 and y2 must have the same length.")
        buffers: list[np.ndarray] = [values_y1, values_y2]
        has_x = x_values is not None
        if has_x:
            values_x = _to_float32_1d(x_values, arg_name="x")
            if values_x.size != values_y1.size:
                raise ValueError("x and y arrays must have the same length.")
            buffers.insert(0, values_x)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "shaded",
                "name": name,
                "length": int(values_y1.size),
                "has_x": bool(has_x),
                "alpha": float(alpha),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            buffers,
        )

    def error_bars(
        self,
        name: str,
        data: Any = _UNSET,
        err: Any | None = None,
        *,
        y: Any = _UNSET,
        err_neg: Any | None = None,
        err_pos: Any | None = None,
        x: Any | None = None,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        values_y = _to_float32_1d(y_values, arg_name="y")
        asymmetric = err_neg is not None or err_pos is not None
        if asymmetric:
            if err is not None:
                raise ValueError("Use either err OR (err_neg, err_pos), not both.")
            if err_neg is None or err_pos is None:
                raise ValueError("Both err_neg and err_pos are required for asymmetric error bars.")
            values_err_neg = _to_float32_1d(
                _resolve_maybe_column(data, err_neg, arg_name="err_neg"),
                arg_name="err_neg",
            )
            values_err_pos = _to_float32_1d(
                _resolve_maybe_column(data, err_pos, arg_name="err_pos"),
                arg_name="err_pos",
            )
            if values_y.size != values_err_neg.size or values_y.size != values_err_pos.size:
                raise ValueError("y, err_neg, and err_pos must have the same length.")
            err_interleaved = np.empty(values_y.size * 2, dtype=np.float32)
            err_interleaved[0::2] = values_err_neg
            err_interleaved[1::2] = values_err_pos
            buffers: list[np.ndarray] = [values_y, err_interleaved]
        else:
            if err is None:
                raise ValueError("err is required for symmetric error bars.")
            values_err = _to_float32_1d(_resolve_maybe_column(data, err, arg_name="err"), arg_name="err")
            if values_y.size != values_err.size:
                raise ValueError("y and err must have the same length.")
            buffers = [values_y, values_err]
        has_x = x_values is not None
        if has_x:
            values_x = _to_float32_1d(x_values, arg_name="x")
            if values_x.size != values_y.size:
                raise ValueError("x and y arrays must have the same length.")
            buffers.insert(0, values_x)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "error_bars",
                "name": name,
                "length": int(values_y.size),
                "has_x": bool(has_x),
                "asymmetric": bool(asymmetric),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            buffers,
        )

    def error_bars_h(
        self,
        name: str,
        data: Any = _UNSET,
        err: Any | None = None,
        *,
        x: Any = _UNSET,
        err_neg: Any | None = None,
        err_pos: Any | None = None,
        y: Any | None = None,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        if x is _UNSET:
            if data is _UNSET:
                raise TypeError("x data is required.")
            if _is_dataframe_like(data):
                raise ValueError("x column must be provided when the data argument is a dataframe.")
            x_values = data
        else:
            x_values = _resolve_maybe_column(data, x, arg_name="x")
        values_x = _to_float32_1d(x_values, arg_name="x")
        asymmetric = err_neg is not None or err_pos is not None
        if asymmetric:
            if err is not None:
                raise ValueError("Use either err OR (err_neg, err_pos), not both.")
            if err_neg is None or err_pos is None:
                raise ValueError("Both err_neg and err_pos are required for asymmetric error bars.")
            values_err_neg = _to_float32_1d(
                _resolve_maybe_column(data, err_neg, arg_name="err_neg"),
                arg_name="err_neg",
            )
            values_err_pos = _to_float32_1d(
                _resolve_maybe_column(data, err_pos, arg_name="err_pos"),
                arg_name="err_pos",
            )
            if values_x.size != values_err_neg.size or values_x.size != values_err_pos.size:
                raise ValueError("x, err_neg, and err_pos must have the same length.")
            err_interleaved = np.empty(values_x.size * 2, dtype=np.float32)
            err_interleaved[0::2] = values_err_neg
            err_interleaved[1::2] = values_err_pos
            values_err = err_interleaved
        else:
            if err is None:
                raise ValueError("err is required for symmetric horizontal error bars.")
            values_err = _to_float32_1d(_resolve_maybe_column(data, err, arg_name="err"), arg_name="err")
            if values_x.size != values_err.size:
                raise ValueError("x and err must have the same length.")
        y_values = _resolve_maybe_column(data, y, arg_name="y") if y is not None else None
        values_y = (
            _to_float32_1d(y_values, arg_name="y")
            if y_values is not None
            else np.arange(values_x.size, dtype=np.float32)
        )
        if values_x.size != values_y.size:
            raise ValueError("x and y must have the same length.")
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "error_bars_h",
                "name": name,
                "length": int(values_x.size),
                "asymmetric": bool(asymmetric),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [values_x, values_err, values_y],
        )

    def inf_lines(
        self,
        name: str,
        values: Any,
        *,
        axis: str = "x",
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        vals = _to_float32_1d(values, arg_name="values")
        axis_norm = axis.lower()
        if axis_norm not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'.")
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "inf_lines",
                "name": name,
                "axis": axis_norm,
                "length": int(vals.size),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [vals],
        )

    def vlines(
        self,
        name: str,
        values: Any,
        *,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        self.inf_lines(
            name,
            values,
            axis="x",
            subplot_index=subplot_index,
            x_axis=x_axis,
            y_axis=y_axis,
        )

    def hlines(
        self,
        name: str,
        values: Any,
        *,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        self.inf_lines(
            name,
            values,
            axis="y",
            subplot_index=subplot_index,
            x_axis=x_axis,
            y_axis=y_axis,
        )

    def histogram(
        self,
        name: str,
        data: Any = _UNSET,
        *,
        y: Any = _UNSET,
        bins: int = 50,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        values = _to_float32_1d(_resolve_y_input(data, y), arg_name="y")
        bin_count = int(bins)
        if bin_count <= 0:
            raise ValueError("bins must be > 0.")
        counts, edges = np.histogram(values, bins=bin_count)
        edges_f32 = np.asarray(edges, dtype=np.float32)
        counts_f32 = np.asarray(counts, dtype=np.float32)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "histogram",
                "name": name,
                "bins": int(bin_count),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [edges_f32, counts_f32],
        )

    def histogram2d(
        self,
        name: str,
        data: Any = _UNSET,
        y: Any = _UNSET,
        *,
        x: Any = _UNSET,
        x_bins: int = 64,
        y_bins: int = 64,
        label_fmt: str | None = "%.0f",
        scale_min: float | None = None,
        scale_max: float | None = None,
        heatmap_flags: int = 0,
        show_colorbar: bool = False,
        colorbar_label: str | None = "",
        colorbar_format: str | None = "%g",
        colorbar_flags: int = 0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        if x is _UNSET:
            if data is _UNSET:
                raise TypeError("x data is required.")
            if _is_dataframe_like(data):
                raise ValueError("x column must be provided when the data argument is a dataframe.")
            x_values = data
        else:
            x_values = _resolve_maybe_column(data, x, arg_name="x")
        if y is _UNSET:
            raise TypeError("y data is required.")
        y_values = _resolve_maybe_column(data, y, arg_name="y")
        values_x = _to_float32_1d(x_values, arg_name="x")
        values_y = _to_float32_1d(y_values, arg_name="y")
        if values_x.size != values_y.size:
            raise ValueError("x and y must have the same length.")
        x_bins_i = int(x_bins)
        y_bins_i = int(y_bins)
        if x_bins_i <= 0 or y_bins_i <= 0:
            raise ValueError("x_bins and y_bins must be > 0.")
        if scale_min is not None and not np.isfinite(float(scale_min)):
            raise ValueError("scale_min must be finite when provided.")
        if scale_max is not None and not np.isfinite(float(scale_max)):
            raise ValueError("scale_max must be finite when provided.")
        heatmap_flags_i = int(heatmap_flags)
        if heatmap_flags_i < 0:
            raise ValueError("heatmap_flags must be >= 0.")
        colorbar_flags_i = int(colorbar_flags)
        if colorbar_flags_i < 0:
            raise ValueError("colorbar_flags must be >= 0.")
        hist, x_edges, y_edges = np.histogram2d(values_x, values_y, bins=(x_bins_i, y_bins_i))
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "histogram2d",
                "name": name,
                "rows": int(hist.shape[0]),
                "cols": int(hist.shape[1]),
                "label_fmt": None if label_fmt is None else str(label_fmt),
                "scale_min": None if scale_min is None else float(scale_min),
                "scale_max": None if scale_max is None else float(scale_max),
                "heatmap_flags": int(heatmap_flags_i),
                "show_colorbar": bool(show_colorbar),
                "colorbar_label": "" if colorbar_label is None else str(colorbar_label),
                "colorbar_format": "%g" if colorbar_format is None else str(colorbar_format),
                "colorbar_flags": colorbar_flags_i,
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [
                np.asarray(x_edges, dtype=np.float32),
                np.asarray(y_edges, dtype=np.float32),
                np.asarray(hist, dtype=np.float32).reshape(-1),
            ],
        )

    def heatmap(
        self,
        name: str,
        z: Any,
        *,
        label_fmt: str | None = "%.2f",
        scale_min: float | None = None,
        scale_max: float | None = None,
        heatmap_flags: int = 0,
        show_colorbar: bool = False,
        colorbar_label: str | None = "",
        colorbar_format: str | None = "%g",
        colorbar_flags: int = 0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        if scale_min is not None and not np.isfinite(float(scale_min)):
            raise ValueError("scale_min must be finite when provided.")
        if scale_max is not None and not np.isfinite(float(scale_max)):
            raise ValueError("scale_max must be finite when provided.")
        heatmap_flags_i = int(heatmap_flags)
        if heatmap_flags_i < 0:
            raise ValueError("heatmap_flags must be >= 0.")
        colorbar_flags_i = int(colorbar_flags)
        if colorbar_flags_i < 0:
            raise ValueError("colorbar_flags must be >= 0.")
        arr = _to_float32_2d(z, arg_name="z")
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "heatmap",
                "name": name,
                "rows": int(arr.shape[0]),
                "cols": int(arr.shape[1]),
                "label_fmt": None if label_fmt is None else str(label_fmt),
                "scale_min": None if scale_min is None else float(scale_min),
                "scale_max": None if scale_max is None else float(scale_max),
                "heatmap_flags": int(heatmap_flags_i),
                "show_colorbar": bool(show_colorbar),
                "colorbar_label": "" if colorbar_label is None else str(colorbar_label),
                "colorbar_format": "%g" if colorbar_format is None else str(colorbar_format),
                "colorbar_flags": colorbar_flags_i,
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [arr.reshape(-1)],
        )

    def image(
        self,
        name: str,
        z: Any,
        *,
        bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        uv0: tuple[float, float] = (0.0, 0.0),
        uv1: tuple[float, float] = (1.0, 1.0),
        tint: Any = (1.0, 1.0, 1.0, 1.0),
        image_flags: int = 0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        flat, rows, cols, channels = _to_image_float32_flat(z, arg_name="z")
        uv0_x, uv0_y = _to_float2(uv0, arg_name="uv0")
        uv1_x, uv1_y = _to_float2(uv1, arg_name="uv1")
        tint_rgba = _normalize_rgba(tint)
        if tint_rgba is None:
            tint_rgba = (1.0, 1.0, 1.0, 1.0)
        image_flags_i = int(image_flags)
        if image_flags_i < 0:
            raise ValueError("image_flags must be >= 0.")
        if bounds is None:
            x_min, y_min = 0.0, 0.0
            x_max, y_max = float(cols), float(rows)
        else:
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError("bounds must be ((x_min, y_min), (x_max, y_max)).")
            x_min, y_min = _to_float2(bounds[0], arg_name="bounds[0]")
            x_max, y_max = _to_float2(bounds[1], arg_name="bounds[1]")
            if x_max == x_min or y_max == y_min:
                raise ValueError("bounds min/max must span a non-zero range.")
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "image",
                "name": name,
                "rows": rows,
                "cols": cols,
                "channels": channels,
                "bounds_x_min": float(x_min),
                "bounds_x_max": float(x_max),
                "bounds_y_min": float(y_min),
                "bounds_y_max": float(y_max),
                "uv0_x": float(uv0_x),
                "uv0_y": float(uv0_y),
                "uv1_x": float(uv1_x),
                "uv1_y": float(uv1_y),
                "image_flags": int(image_flags_i),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [flat, np.asarray(tint_rgba, dtype=np.float32)],
        )

    def pie_chart(
        self,
        name: str,
        values: Any,
        *,
        labels: list[str] | tuple[str, ...] | None = None,
        x: float = 0.0,
        y: float = 0.0,
        radius: float = 1.0,
        angle0: float = 90.0,
        label_fmt: str = "%.1f",
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        vals = _to_float32_1d(values, arg_name="values")
        if labels is None:
            labels = tuple(str(i) for i in range(int(vals.size)))
        if len(labels) != int(vals.size):
            raise ValueError("labels length must match values length.")
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "pie_chart",
                "name": name,
                "labels": [str(s) for s in labels],
                "x": float(x),
                "y": float(y),
                "radius": float(radius),
                "angle0": float(angle0),
                "label_fmt": str(label_fmt),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [vals],
        )

    def text(
        self,
        label: str,
        x: float,
        y: float,
        *,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "text",
                "label": str(label),
                "x": float(x),
                "y": float(y),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def annotation(
        self,
        label: str,
        x: float,
        y: float,
        *,
        offset_x: float = 8.0,
        offset_y: float = -8.0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "annotation",
                "label": str(label),
                "x": float(x),
                "y": float(y),
                "offset_x": float(offset_x),
                "offset_y": float(offset_y),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def dummy(
        self,
        name: str,
        *,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "dummy",
                "name": str(name),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def tag_x(
        self,
        value: float,
        *,
        label_fmt: str | None = "%g",
        round_value: bool = False,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "tag_x",
                "name": "",
                "value": float(value),
                "label_fmt": None if label_fmt is None else str(label_fmt),
                "round_value": bool(round_value),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def tag_y(
        self,
        value: float,
        *,
        label_fmt: str | None = "%g",
        round_value: bool = False,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "tag_y",
                "name": "",
                "value": float(value),
                "label_fmt": None if label_fmt is None else str(label_fmt),
                "round_value": bool(round_value),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def colormap_slider(
        self,
        *,
        label: str = "Colormap",
        t: float = 0.5,
        fmt: str = "",
        subplot_index: int = 0,
    ) -> None:
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "colormap_slider",
                "label": str(label),
                "value": float(t),
                "label_fmt": str(fmt),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": 0,
                "y_axis": 3,
            },
            [],
        )

    def colormap_button(
        self,
        *,
        label: str = "Colormap",
        width: float = 0.0,
        height: float = 0.0,
        subplot_index: int = 0,
    ) -> None:
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "colormap_button",
                "label": str(label),
                "x": float(width),
                "y": float(height),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": 0,
                "y_axis": 3,
            },
            [],
        )

    def colormap_selector(
        self,
        *,
        label: str = "Colormap",
        subplot_index: int = 0,
    ) -> None:
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "colormap_selector",
                "label": str(label),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": 0,
                "y_axis": 3,
            },
            [],
        )

    def drag_drop_plot(
        self,
        *,
        source: bool = True,
        target: bool = True,
        subplot_index: int = 0,
    ) -> None:
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "drag_drop_plot",
                "has_x": bool(source),
                "axis": "plot" if target else "none",
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": 0,
                "y_axis": 3,
            },
            [],
        )

    def drag_drop_axis(
        self,
        axis: str,
        *,
        source: bool = True,
        target: bool = True,
        subplot_index: int = 0,
    ) -> None:
        axis_code = self._axis_code(axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "drag_drop_axis",
                "has_x": bool(source),
                "axis": str(axis),
                "length": int(axis_code),
                "value": 1.0 if target else 0.0,
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": 0,
                "y_axis": 3,
            },
            [],
        )

    def drag_drop_legend(
        self,
        *,
        target: bool = True,
        subplot_index: int = 0,
    ) -> None:
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "drag_drop_legend",
                "value": 1.0 if target else 0.0,
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": 0,
                "y_axis": 3,
            },
            [],
        )

    def drag_line_x(
        self,
        name: str,
        value: float,
        *,
        thickness: float = 1.0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "drag_line_x",
                "name": str(name),
                "value": float(value),
                "thickness": float(thickness),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def drag_line_y(
        self,
        name: str,
        value: float,
        *,
        thickness: float = 1.0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "drag_line_y",
                "name": str(name),
                "value": float(value),
                "thickness": float(thickness),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def drag_point(
        self,
        name: str,
        x: float,
        y: float,
        *,
        size: float = 4.0,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "drag_point",
                "name": str(name),
                "x": float(x),
                "y": float(y),
                "size": float(size),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def drag_rect(
        self,
        name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
    ) -> None:
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        self._send_primitive(
            {
                "type": "primitive_add",
                "kind": "drag_rect",
                "name": str(name),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "subplot_index": self._validate_subplot_index(subplot_index),
                "x_axis": int(x_axis_code),
                "y_axis": int(y_axis_code),
            },
            [],
        )

    def set_view(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        x_min_v = float(x_min)
        x_max_v = float(x_max)
        y_min_v = float(y_min)
        y_max_v = float(y_max)
        if not np.isfinite([x_min_v, x_max_v, y_min_v, y_max_v]).all():
            raise ValueError("view values must be finite.")
        if x_max_v <= x_min_v:
            raise ValueError("x_max must be greater than x_min.")
        if y_max_v <= y_min_v:
            raise ValueError("y_max must be greater than y_min.")
        self._last_view = {
            "x_min": x_min_v,
            "x_max": x_max_v,
            "y_min": y_min_v,
            "y_max": y_max_v,
        }
        self.send(
            {
                "type": "set_view",
                "x_min": x_min_v,
                "x_max": x_max_v,
                "y_min": y_min_v,
                "y_max": y_max_v,
            }
        )

    def hide_next_item(self, hidden: bool = True) -> None:
        self._hide_next_item = bool(hidden)

    def autoscale(self) -> None:
        self.send({"type": "autoscale"})

    def set_axis_scale(self, *, x: str = "linear", y: str = "linear") -> None:
        self._axis_scale_x = self._axis_scale_code(x, axis_name="x")
        self._axis_scale_y = self._axis_scale_code(y, axis_name="y")
        self._axis_state[0] = (True, self._axis_scale_x)
        self._axis_state[3] = (True, self._axis_scale_y)
        self.send(
            {
                "type": "plot_options",
                "plot_flags": int(self._plot_flags),
                "axis_scale_x": int(self._axis_scale_x),
                "axis_scale_y": int(self._axis_scale_y),
            }
        )
        self.send({"type": "axis_state", "axis": 0, "enabled": True, "scale": int(self._axis_scale_x)})
        self.send({"type": "axis_state", "axis": 3, "enabled": True, "scale": int(self._axis_scale_y)})

    def set_axis_label(self, axis: str, label: str | None = None) -> None:
        axis_code = self._axis_code(axis)
        value = "" if label is None else str(label)
        self._axis_labels[axis_code] = value
        self.send({"type": "axis_label", "axis": int(axis_code), "label": value})

    def set_axis_format(self, axis: str, fmt: str | None = None) -> None:
        axis_code = self._axis_code(axis)
        value = "" if fmt is None else str(fmt)
        self._axis_formats[axis_code] = value
        self.send({"type": "axis_format", "axis": int(axis_code), "format": value})

    def set_axis_ticks(
        self,
        axis: str,
        values: Any,
        *,
        labels: list[str] | tuple[str, ...] | None = None,
        keep_default: bool = False,
    ) -> None:
        axis_code = self._axis_code(axis)
        vals = _to_float32_1d(values, arg_name="values")
        labels_list = [str(v) for v in labels] if labels is not None else []
        if labels and len(labels_list) != int(vals.size):
            raise ValueError("labels length must match ticks values length.")
        self._axis_ticks[axis_code] = (vals, labels_list, bool(keep_default))
        self.send(
            {
                "type": "axis_ticks",
                "axis": int(axis_code),
                "count": int(vals.size),
                "labels": labels_list,
                "keep_default": bool(keep_default),
            },
            buffers=[memoryview(vals)],
        )

    def clear_axis_ticks(self, axis: str) -> None:
        axis_code = self._axis_code(axis)
        self._axis_ticks.pop(axis_code, None)
        self.send({"type": "axis_ticks_clear", "axis": int(axis_code)})

    def set_axis_limits_constraints(
        self,
        axis: str,
        min_value: float | None,
        max_value: float | None,
        *,
        enabled: bool = True,
    ) -> None:
        axis_code = self._axis_code(axis)
        if not enabled or min_value is None or max_value is None:
            self._axis_limits_constraints[axis_code] = (False, 0.0, 0.0)
            self.send(
                {
                    "type": "axis_limits_constraints",
                    "axis": int(axis_code),
                    "enabled": False,
                    "min": 0.0,
                    "max": 0.0,
                }
            )
            return
        vmin = float(min_value)
        vmax = float(max_value)
        if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
            raise ValueError("Axis limits constraints require finite min/max with max > min.")
        self._axis_limits_constraints[axis_code] = (True, vmin, vmax)
        self.send(
            {
                "type": "axis_limits_constraints",
                "axis": int(axis_code),
                "enabled": True,
                "min": vmin,
                "max": vmax,
            }
        )

    def set_axis_zoom_constraints(
        self,
        axis: str,
        min_zoom: float | None,
        max_zoom: float | None,
        *,
        enabled: bool = True,
    ) -> None:
        axis_code = self._axis_code(axis)
        if not enabled or min_zoom is None or max_zoom is None:
            self._axis_zoom_constraints[axis_code] = (False, 0.0, 0.0)
            self.send(
                {
                    "type": "axis_zoom_constraints",
                    "axis": int(axis_code),
                    "enabled": False,
                    "min": 0.0,
                    "max": 0.0,
                }
            )
            return
        zmin = float(min_zoom)
        zmax = float(max_zoom)
        if not np.isfinite([zmin, zmax]).all() or zmax <= zmin:
            raise ValueError("Axis zoom constraints require finite min/max with max > min.")
        self._axis_zoom_constraints[axis_code] = (True, zmin, zmax)
        self.send(
            {
                "type": "axis_zoom_constraints",
                "axis": int(axis_code),
                "enabled": True,
                "min": zmin,
                "max": zmax,
            }
        )

    def set_axis_link(self, axis: str, target_axis: str | None = None) -> None:
        axis_code = self._axis_code(axis)
        if target_axis is None:
            self._axis_links.pop(axis_code, None)
            self.send({"type": "axis_link", "axis": int(axis_code), "target_axis": -1})
            return
        target_code = self._axis_code(target_axis)
        if axis_code == target_code:
            self._axis_links.pop(axis_code, None)
            self.send({"type": "axis_link", "axis": int(axis_code), "target_axis": -1})
            return
        if (axis_code <= 2) != (target_code <= 2):
            raise ValueError("axis and target_axis must both be x-axes or both be y-axes.")
        self._axis_links[axis_code] = target_code
        self.send(
            {
                "type": "axis_link",
                "axis": int(axis_code),
                "target_axis": int(target_code),
            }
        )

    def set_axis_state(self, axis: str, *, enabled: bool, scale: str = "linear") -> None:
        axis_code = self._axis_code(axis)
        scale_code = self._axis_scale_code(scale, axis_name=axis)
        if axis_code in (0, 3):
            enabled = True
        self._axis_state[axis_code] = (bool(enabled), int(scale_code))
        if axis_code == 0:
            self._axis_scale_x = scale_code
        elif axis_code == 3:
            self._axis_scale_y = scale_code
        self.send(
            {
                "type": "axis_state",
                "axis": int(axis_code),
                "enabled": bool(enabled),
                "scale": int(scale_code),
            }
        )
        if axis_code in (0, 3):
            self.send(
                {
                    "type": "plot_options",
                    "plot_flags": int(self._plot_flags),
                    "axis_scale_x": int(self._axis_scale_x),
                    "axis_scale_y": int(self._axis_scale_y),
                }
            )

    def set_secondary_axes(
        self, *, x2: bool = False, x3: bool = False, y2: bool = False, y3: bool = False
    ) -> None:
        self.set_axis_state("x2", enabled=x2, scale=self._scale_name(self._axis_state[1][1]))
        self.set_axis_state("x3", enabled=x3, scale=self._scale_name(self._axis_state[2][1]))
        self.set_axis_state("y2", enabled=y2, scale=self._scale_name(self._axis_state[4][1]))
        self.set_axis_state("y3", enabled=y3, scale=self._scale_name(self._axis_state[5][1]))

    def set_time_axis(self, axis: str = "x1") -> None:
        self.set_axis_state(axis, enabled=True, scale="time")

    def set_plot_flags(
        self,
        *,
        no_legend: bool = False,
        no_menus: bool = False,
        no_box_select: bool = False,
        no_mouse_pos: bool = False,
        crosshairs: bool = False,
        equal: bool = False,
    ) -> None:
        flags = 0
        if no_legend:
            flags |= _PLOT_FLAG_NO_LEGEND
        if no_menus:
            flags |= _PLOT_FLAG_NO_MENUS
        if no_box_select:
            flags |= _PLOT_FLAG_NO_BOX_SELECT
        if no_mouse_pos:
            flags |= _PLOT_FLAG_NO_MOUSE_POS
        if crosshairs:
            flags |= _PLOT_FLAG_CROSSHAIRS
        if equal:
            flags |= _PLOT_FLAG_EQUAL
        self._plot_flags = flags
        self.send(
            {
                "type": "plot_options",
                "plot_flags": int(self._plot_flags),
                "axis_scale_x": int(self._axis_scale_x),
                "axis_scale_y": int(self._axis_scale_y),
            }
        )

    def set_subplots_config(self, *, rows: int, cols: int, flags: int = 0) -> None:
        rows_i = max(1, int(rows))
        cols_i = max(1, int(cols))
        self._subplot_rows = rows_i
        self._subplot_cols = cols_i
        self._subplot_flags = max(0, int(flags))
        self.send(
            {
                "type": "subplots_config",
                "rows": rows_i,
                "cols": cols_i,
                "flags": int(self._subplot_flags),
            }
        )

    def set_aligned_group(
        self,
        group_id: str,
        *,
        enabled: bool = True,
        vertical: bool = True,
    ) -> None:
        self._aligned_group_id = str(group_id)
        self._aligned_enabled = bool(enabled) and bool(self._aligned_group_id)
        self._aligned_vertical = bool(vertical)
        self.send(
            {
                "type": "aligned_group",
                "group_id": self._aligned_group_id,
                "enabled": bool(self._aligned_enabled),
                "vertical": bool(self._aligned_vertical),
            }
        )

    def set_colormap(self, name: str | None = None) -> None:
        if name is None:
            self._colormap_name = ""
        else:
            raw = str(name).strip()
            self._colormap_name = _COLORMAP_CANONICAL_NAMES.get(raw.lower(), raw)
        self.send({"type": "colormap", "name": self._colormap_name})

    def on_view_change(self, callback: Callable[[Plot, dict[str, float]], None]) -> None:
        self._view_callbacks.append(callback)

    def on_tool_change(self, callback: Callable[[Plot, dict[str, Any]], None]) -> None:
        self._tool_callbacks.append(callback)

    def on_selection_change(self, callback: Callable[[Plot, dict[str, Any]], None]) -> None:
        self._selection_callbacks.append(callback)

    def on_select(self, callback: Callable[[Plot, dict[str, Any]], None]) -> None:
        self.on_selection_change(callback)

    def on_hover(self, callback: Callable[[Plot, dict[str, Any]], None]) -> None:
        self._hover_callbacks.append(callback)

    def on_click(self, callback: Callable[[Plot, dict[str, Any]], None]) -> None:
        self._click_callbacks.append(callback)

    def on_perf_stats(
        self, callback: Callable[[Plot, dict[str, float]], None], *, interval_ms: int = 500
    ) -> None:
        self._perf_callbacks.append(callback)
        interval_i = max(100, int(interval_ms))
        if self._perf_reporting_enabled and interval_i == self._perf_interval_ms:
            return
        self._perf_reporting_enabled = True
        self._perf_interval_ms = interval_i
        self.send(
            {
                "type": "set_perf_reporting",
                "enabled": True,
                "interval_ms": self._perf_interval_ms,
            }
        )

    def indices_for_selection(
        self,
        selection: dict[str, Any],
        series: str | LineHandle | None = None,
    ) -> dict[str, np.ndarray] | np.ndarray:
        """Return exact raw point indices inside a selection rectangle.

        The callback payload includes fast WASM-computed x-index ranges. This
        helper applies the y bounds in Python only when the caller explicitly
        asks for exact indices.
        """
        if not isinstance(selection, dict):
            raise TypeError("selection must be the dict passed to on_select/on_selection_change.")
        try:
            x_min = float(selection["x_min"])
            x_max = float(selection["x_max"])
            y_min = float(selection["y_min"])
            y_max = float(selection["y_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("selection must contain finite x_min/x_max/y_min/y_max values.") from exc
        if not np.isfinite([x_min, x_max, y_min, y_max]).all():
            raise ValueError("selection bounds must be finite.")
        x0, x1 = sorted((x_min, x_max))
        y0, y1 = sorted((y_min, y_max))

        def _indices_for_meta(meta: _SeriesMeta) -> np.ndarray:
            y = meta.data
            if y.size == 0:
                return np.empty(0, dtype=np.int64)
            if meta.x_data is not None:
                x = meta.x_data
                start = int(np.searchsorted(x, x0, side="left"))
                stop = int(np.searchsorted(x, x1, side="right"))
                if stop <= start:
                    return np.empty(0, dtype=np.int64)
                window_y = y[start:stop]
                window_x = x[start:stop]
                mask = (
                    np.isfinite(window_x)
                    & np.isfinite(window_y)
                    & (window_x >= x0)
                    & (window_x <= x1)
                    & (window_y >= y0)
                    & (window_y <= y1)
                )
                return (np.nonzero(mask)[0] + start).astype(np.int64, copy=False)

            start = max(0, int(np.ceil(x0)))
            stop = min(int(y.size), int(np.floor(x1)) + 1)
            if stop <= start:
                return np.empty(0, dtype=np.int64)
            window_y = y[start:stop]
            mask = np.isfinite(window_y) & (window_y >= y0) & (window_y <= y1)
            return (np.nonzero(mask)[0] + start).astype(np.int64, copy=False)

        if series is not None:
            return _indices_for_meta(self._series[self._resolve_series_id(series)])
        return {sid: _indices_for_meta(meta) for sid, meta in self._series.items()}

    selection_indices = indices_for_selection

    def selection_bounds(self, selection: dict[str, Any]) -> tuple[float, float, float, float]:
        if not isinstance(selection, dict):
            raise TypeError("selection must be the dict passed to on_select/on_selection_change.")
        try:
            x0, x1 = sorted((float(selection["x_min"]), float(selection["x_max"])))
            y0, y1 = sorted((float(selection["y_min"]), float(selection["y_max"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("selection must contain finite x_min/x_max/y_min/y_max values.") from exc
        if not np.isfinite([x0, x1, y0, y1]).all():
            raise ValueError("selection bounds must be finite.")
        return x0, x1, y0, y1

    def highlight_selection(
        self,
        selection: dict[str, Any],
        series: str | LineHandle | None = None,
        *,
        name: str = "selection",
        points: bool = True,
        rect: bool = True,
    ) -> None:
        x0, x1, y0, y1 = self.selection_bounds(selection)
        subplot_index = int(selection.get("subplot_index", 0))
        if rect:
            self.drag_rect(name, x0, y0, x1, y1, subplot_index=subplot_index)
        if not points:
            return
        selected = self.indices_for_selection(selection, series)
        items = selected.items() if isinstance(selected, dict) else [(series, selected)]
        for key, indices in items:
            if indices.size == 0:
                continue
            if isinstance(key, LineHandle):
                series_id = key.series_id
            elif isinstance(key, str):
                series_id = self._resolve_series_id(key)
            else:
                series_id = ""
                if series is not None:
                    series_id = self._resolve_series_id(series)
            meta = self._series.get(series_id)
            if meta is None:
                continue
            idx = np.asarray(indices, dtype=np.int64)
            y_vals = np.ascontiguousarray(meta.data[idx], dtype=np.float32)
            x_vals = (
                np.ascontiguousarray(meta.x_data[idx], dtype=np.float32)
                if meta.x_data is not None
                else np.ascontiguousarray(idx.astype(np.float32), dtype=np.float32)
            )
            label = f"{name}:{meta.name}"
            self.scatter(label, y_vals, x=x_vals, size=5.0, subplot_index=meta.subplot_index)

    def export_csv_selection(
        self,
        selection: dict[str, Any],
        series: str | LineHandle | None = None,
        *,
        filename: str | pathlib.Path | None = None,
    ) -> str:
        selected = self.indices_for_selection(selection, series)
        rows = ["series_id,series_name,index,x,y"]
        items = selected.items() if isinstance(selected, dict) else [(series, selected)]
        for key, indices in items:
            if isinstance(key, LineHandle):
                series_id = key.series_id
            elif isinstance(key, str):
                series_id = self._resolve_series_id(key)
            elif isinstance(series, LineHandle):
                series_id = series.series_id
            elif isinstance(series, str):
                series_id = self._resolve_series_id(series)
            else:
                series_id = str(key or "")
            meta = self._series.get(series_id)
            if meta is None:
                continue
            for raw_idx in np.asarray(indices, dtype=np.int64):
                idx = int(raw_idx)
                x_val = float(meta.x_data[idx]) if meta.x_data is not None else float(idx)
                y_val = float(meta.data[idx])
                rows.append(f"{series_id},{meta.name},{idx},{x_val:.9g},{y_val:.9g}")
        csv_text = "\n".join(rows) + "\n"
        if filename is not None:
            pathlib.Path(filename).write_text(csv_text, encoding="utf-8")
        return csv_text

    def get_state(
        self,
        key: Any = _UNSET,
        drop_defaults: bool = False,
        *,
        include_data: bool = False,
    ) -> dict[str, Any]:
        if (
            key is not _UNSET
            or drop_defaults
            or not hasattr(self, "_plot_flags")
            or self._called_from_widget_state()
        ):
            kwargs: dict[str, Any] = {"drop_defaults": drop_defaults}
            if key is not _UNSET:
                kwargs["key"] = key
            return super().get_state(**kwargs)
        state: dict[str, Any] = {
            "version": 1,
            "width": int(self.width),
            "height": int(self.height),
            "title": str(self.title),
            "plot_flags": int(self._plot_flags),
            "axis_scale_x": self._scale_name(self._axis_scale_x),
            "axis_scale_y": self._scale_name(self._axis_scale_y),
            "colormap": self._colormap_name,
            "theme": self._theme_name,
            "view": dict(self._last_view) if self._last_view is not None else None,
            "subplots": {
                "rows": int(self._subplot_rows),
                "cols": int(self._subplot_cols),
                "flags": int(self._subplot_flags),
            },
            "aligned_group": {
                "group_id": self._aligned_group_id,
                "enabled": bool(self._aligned_enabled),
                "vertical": bool(self._aligned_vertical),
            },
            "linked_crosshair": dict(self._linked_crosshair),
            "axis_state": {
                str(axis): {"enabled": bool(enabled), "scale": self._scale_name(scale)}
                for axis, (enabled, scale) in self._axis_state.items()
            },
            "axis_labels": {str(k): v for k, v in self._axis_labels.items()},
            "axis_formats": {str(k): v for k, v in self._axis_formats.items()},
            "axis_links": {str(k): v for k, v in self._axis_links.items()},
            "series": [],
        }
        for series_id, meta in self._series.items():
            item: dict[str, Any] = {
                "series_id": series_id,
                "name": meta.name,
                "subplot_index": int(meta.subplot_index),
                "x_axis": int(meta.x_axis),
                "y_axis": int(meta.y_axis),
                "style": self._series_style_payload(meta),
                "hidden": bool(meta.hidden),
                "stream_capacity": meta.stream_capacity,
                "stream_paused": bool(meta.stream_paused),
                "stream_auto_render": bool(meta.stream_auto_render),
                "stream_autoscale_y": bool(meta.stream_autoscale_y),
                "has_x": meta.x_data is not None,
            }
            if include_data:
                item["y"] = meta.data.tolist()
                if meta.x_data is not None:
                    item["x"] = meta.x_data.tolist()
            state["series"].append(item)
        return state

    @staticmethod
    def _called_from_widget_state() -> bool:
        for frame in inspect.stack(context=0)[1:6]:
            if "ipywidgets" in frame.filename and frame.function in {
                "open",
                "send_state",
                "get_manager_state",
            }:
                return True
        return False

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("state must be a dict returned by get_state().")
        if "width" in state:
            self.width = int(state["width"])
        if "height" in state:
            self.height = int(state["height"])
        if "title" in state:
            self.title = str(state["title"])
        subplots = state.get("subplots")
        if isinstance(subplots, dict):
            self.set_subplots_config(
                rows=int(subplots.get("rows", self._subplot_rows)),
                cols=int(subplots.get("cols", self._subplot_cols)),
                flags=int(subplots.get("flags", self._subplot_flags)),
            )
        if "plot_flags" in state:
            self._plot_flags = int(state["plot_flags"])
            self.send(
                {
                    "type": "plot_options",
                    "plot_flags": int(self._plot_flags),
                    "axis_scale_x": int(self._axis_scale_x),
                    "axis_scale_y": int(self._axis_scale_y),
                }
            )
        if "axis_scale_x" in state or "axis_scale_y" in state:
            self.set_axis_scale(
                x=str(state.get("axis_scale_x", self._scale_name(self._axis_scale_x))),
                y=str(state.get("axis_scale_y", self._scale_name(self._axis_scale_y))),
            )
        for axis, cfg in (state.get("axis_state") or {}).items():
            if isinstance(cfg, dict):
                self.set_axis_state(
                    self._axis_name(int(axis)),
                    enabled=bool(cfg.get("enabled", True)),
                    scale=str(cfg.get("scale", "linear")),
                )
        for axis, label in (state.get("axis_labels") or {}).items():
            self.set_axis_label(self._axis_name(int(axis)), str(label))
        for axis, fmt in (state.get("axis_formats") or {}).items():
            self.set_axis_format(self._axis_name(int(axis)), str(fmt))
        for axis, target in (state.get("axis_links") or {}).items():
            self.set_axis_link(self._axis_name(int(axis)), self._axis_name(int(target)))
        aligned = state.get("aligned_group")
        if isinstance(aligned, dict):
            self.set_aligned_group(
                str(aligned.get("group_id", "")),
                enabled=bool(aligned.get("enabled", False)),
                vertical=bool(aligned.get("vertical", True)),
            )
        if state.get("theme"):
            self.set_theme(str(state.get("theme")))
        if "colormap" in state:
            self.set_colormap(state.get("colormap"))
        linked = state.get("linked_crosshair")
        if isinstance(linked, dict):
            self.set_linked_crosshair(
                str(linked.get("group_id", "default")),
                enabled=bool(linked.get("enabled", False)),
                axis=str(linked.get("axis", "x")),
            )
        view = state.get("view")
        if isinstance(view, dict):
            self.set_view(float(view["x_min"]), float(view["x_max"]), float(view["y_min"]), float(view["y_max"]))
        for item in state.get("series", []):
            if not isinstance(item, dict) or "y" not in item:
                continue
            style = item.get("style") or {}
            handle = self.line(
                str(item.get("name", "series")),
                item["y"],
                x=item.get("x"),
                subplot_index=int(item.get("subplot_index", 0)),
                x_axis=self._axis_name(int(item.get("x_axis", 0))),
                y_axis=self._axis_name(int(item.get("y_axis", 3))),
                color=(
                    style.get("color_r", 0.0),
                    style.get("color_g", 0.0),
                    style.get("color_b", 0.0),
                    style.get("color_a", 0.0),
                )
                if style.get("has_color")
                else None,
                line_weight=float(style.get("line_weight", 1.0)),
                marker=self._marker_name(int(style.get("marker", _SERIES_MARKER_NONE))),
                marker_size=float(style.get("marker_size", 4.0)),
                max_points=item.get("stream_capacity"),
            )
            handle.set_stream_options(
                auto_render=bool(item.get("stream_auto_render", False)),
                autoscale_y=bool(item.get("stream_autoscale_y", False)),
            )
            if item.get("stream_paused"):
                handle.pause()

    def export_json_state(
        self,
        filename: str | pathlib.Path | None = None,
        *,
        include_data: bool = False,
    ) -> str:
        text = json.dumps(self.get_state(include_data=include_data), indent=2)
        if filename is not None:
            pathlib.Path(filename).write_text(text, encoding="utf-8")
        return text

    def set_linked_crosshair(
        self,
        group_id: str = "default",
        *,
        enabled: bool = True,
        axis: str = "x",
    ) -> None:
        axis_norm = str(axis).strip().lower()
        if axis_norm not in {"x", "y", "xy"}:
            raise ValueError("axis must be 'x', 'y', or 'xy'.")
        self._linked_crosshair = {
            "enabled": bool(enabled),
            "group_id": str(group_id or "default"),
            "axis": axis_norm,
        }
        self.send({"type": "linked_crosshair", **self._linked_crosshair})

    def set_theme(self, name: str = "nbimplot") -> None:
        theme = str(name or "").strip() or "nbimplot"
        self._theme_name = theme
        self.send({"type": "theme", "name": theme})

    def copy_png_to_clipboard(self) -> None:
        self._ensure_open()
        self.send({"type": "copy_png_to_clipboard"})

    def render(self) -> None:
        self._ensure_open()
        self.send({"type": "render"})

    def export_png(self, filename: str = "nbimplot.png") -> None:
        """Download the current frontend canvas as a PNG from the notebook view."""
        self._ensure_open()
        self.send({"type": "export_png", "filename": str(filename or "nbimplot.png")})

    def show(self) -> None:
        display(self)
        return None

    def __repr__(self) -> str:
        title = f", title={self.title!r}" if self.title else ""
        return (
            f"nbimplot.Plot(width={self.width}, height={self.height}{title}, "
            f"series={len(self._series)}, primitives={len(self._primitives)}, renderer='wasm-implot')"
        )

    def _repr_mimebundle_(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
        return _with_text_plain(super()._repr_mimebundle_(**kwargs), repr(self), kwargs)

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            self.send({"type": "dispose"})
            self._series.clear()
            self._closed = True
        super().close()

    def _set_series_data(
        self,
        series_id: str,
        data: Any = _UNSET,
        *,
        x: Any | None = None,
        y: Any = _UNSET,
    ) -> None:
        self._ensure_open()
        meta = self._series.get(series_id)
        if meta is None:
            raise KeyError(f"Unknown series_id: {series_id}")

        x_values, y_values = _resolve_xy_inputs(data, x=x, y=y)
        data = _to_float32_1d(y_values, arg_name="y")
        if x_values is not None:
            x_data = _to_line_x(x_values, y_len=int(data.size))
        elif meta.x_data is not None:
            if int(data.size) != int(meta.x_data.size):
                raise ValueError("x must be provided when resizing a custom-x line.")
            x_data = meta.x_data
        else:
            x_data = None
        if meta.stream_capacity is not None and data.size > int(meta.stream_capacity):
            data = data[-int(meta.stream_capacity) :].copy()
            if x_data is not None:
                x_data = x_data[-int(meta.stream_capacity) :].copy()
        reuse_allocation = (
            data.size == meta.length
            and meta.dtype == "float32"
            and ((meta.x_data is None and x_data is None) or (meta.x_data is not None and x_data is not None))
        )

        meta.length = int(data.size)
        meta.dtype = "float32"
        meta.data = data
        meta.x_data = x_data

        self.send(
            {
                "type": "set_data",
                "series_id": series_id,
                "dtype": "float32",
                "length": int(data.size),
                "has_x": bool(x_data is not None),
                "reuse_allocation": reuse_allocation,
            },
            buffers=[memoryview(x_data), memoryview(data)] if x_data is not None else [memoryview(data)],
        )

    def _resolve_series_id(self, value: str | LineHandle) -> str:
        if isinstance(value, LineHandle):
            return value.series_id
        raw = str(value)
        if raw in self._series:
            return raw
        for series_id, meta in self._series.items():
            if meta.name == raw:
                return series_id
        raise KeyError(f"Unknown series: {raw!r}")

    def _append_series_data(
        self,
        series_id: str,
        y: Any,
        *,
        x: Any | None = None,
        max_points: int | None = None,
    ) -> None:
        self._ensure_open()
        meta = self._series.get(series_id)
        if meta is None:
            raise KeyError(f"Unknown series_id: {series_id}")
        append_data = _to_float32_1d(y, arg_name="y")
        append_x = _to_line_x(x, y_len=int(append_data.size)) if x is not None else None
        if meta.x_data is not None and append_x is None:
            raise ValueError("x must be provided when appending to a custom-x line series.")
        if meta.x_data is None and append_x is not None:
            existing_x: np.ndarray | None = np.arange(meta.data.size, dtype=np.float32)
        else:
            existing_x = meta.x_data
        if existing_x is not None and append_x is not None and existing_x.size > 0 and append_x.size > 0:
            if float(append_x[0]) < float(existing_x[-1]):
                raise ValueError("appended x values must continue the non-decreasing x order.")
        if max_points is None:
            cap = meta.stream_capacity
        else:
            cap_i = int(max_points)
            if cap_i <= 0:
                raise ValueError("max_points must be > 0 when provided.")
            cap = cap_i
            meta.stream_capacity = cap_i
        if append_data.size == 0:
            return
        if existing_x is not None and append_x is not None:
            if cap is not None and append_data.size >= int(cap):
                new_data = append_data[-int(cap) :].copy()
                new_x = append_x[-int(cap) :].copy()
            else:
                new_data = np.concatenate([meta.data, append_data]).astype(np.float32, copy=False)
                new_x = np.concatenate([existing_x, append_x]).astype(np.float32, copy=False)
                if cap is not None and new_data.size > int(cap):
                    new_data = new_data[-int(cap) :].copy()
                    new_x = new_x[-int(cap) :].copy()
            self._set_series_data(series_id, new_data, x=new_x)
            return
        if cap is not None and append_data.size >= int(cap):
            new_data = append_data[-int(cap) :].copy()
        else:
            new_data = np.concatenate([meta.data, append_data]).astype(np.float32, copy=False)
            if cap is not None and new_data.size > int(cap):
                new_data = new_data[-int(cap) :].copy()
        meta.data = np.ascontiguousarray(new_data, dtype=np.float32)
        meta.length = int(meta.data.size)
        meta.dtype = "float32"
        self.send(
            {
                "type": "append_data",
                "series_id": series_id,
                "dtype": "float32",
                "append_length": int(append_data.size),
                "max_points": 0 if cap is None else int(cap),
                "length": int(meta.length),
            },
            buffers=[memoryview(append_data)],
        )

    def _clear_series_data(self, series_id: str, *, x0: float = 0.0, y0: float = 0.0) -> None:
        meta = self._series.get(series_id)
        if meta is None:
            raise KeyError(f"Unknown series_id: {series_id}")
        data = np.array([float(y0)], dtype=np.float32)
        if meta.x_data is not None:
            self._set_series_data(series_id, data, x=np.array([float(x0)], dtype=np.float32))
        else:
            self._set_series_data(series_id, data)

    def _set_stream_capacity(self, series_id: str, capacity: int) -> None:
        meta = self._series.get(series_id)
        if meta is None:
            raise KeyError(f"Unknown series_id: {series_id}")
        meta.stream_capacity = int(capacity)
        if meta.data.size > int(capacity):
            y = meta.data[-int(capacity) :].copy()
            x = meta.x_data[-int(capacity) :].copy() if meta.x_data is not None else None
            self._set_series_data(series_id, y, x=x)

    def _set_stream_paused(self, series_id: str, paused: bool) -> None:
        meta = self._series.get(series_id)
        if meta is None:
            raise KeyError(f"Unknown series_id: {series_id}")
        meta.stream_paused = bool(paused)

    def _set_stream_options(self, series_id: str, *, auto_render: bool, autoscale_y: bool) -> None:
        meta = self._series.get(series_id)
        if meta is None:
            raise KeyError(f"Unknown series_id: {series_id}")
        meta.stream_auto_render = bool(auto_render)
        meta.stream_autoscale_y = bool(autoscale_y)

    @staticmethod
    def _series_style_payload(meta: _SeriesMeta) -> dict[str, Any]:
        color = meta.color if meta.color is not None else (0.0, 0.0, 0.0, 0.0)
        return {
            "has_color": bool(meta.color is not None),
            "color_r": float(color[0]),
            "color_g": float(color[1]),
            "color_b": float(color[2]),
            "color_a": float(color[3]),
            "line_weight": float(meta.line_weight),
            "marker": int(meta.marker),
            "marker_size": float(meta.marker_size),
        }

    def _set_series_style(
        self,
        series_id: str,
        *,
        color: Any = _UNSET,
        line_weight: float | None = None,
        marker: str | None = None,
        marker_size: float | None = None,
    ) -> None:
        self._ensure_open()
        meta = self._series.get(series_id)
        if meta is None:
            raise KeyError(f"Unknown series_id: {series_id}")

        changed = False
        if color is not _UNSET:
            next_color = _normalize_rgba(color)
            if next_color != meta.color:
                meta.color = next_color
                changed = True
        if line_weight is not None:
            next_weight = _normalize_line_weight(line_weight)
            if next_weight != meta.line_weight:
                meta.line_weight = next_weight
                changed = True
        if marker is not None:
            next_marker = _parse_marker(marker)
            if next_marker != meta.marker:
                meta.marker = next_marker
                changed = True
        if marker_size is not None:
            next_marker_size = _normalize_marker_size(marker_size)
            if next_marker_size != meta.marker_size:
                meta.marker_size = next_marker_size
                changed = True

        if not changed:
            return

        payload = {"type": "series_style", "series_id": series_id}
        payload.update(self._series_style_payload(meta))
        self.send(payload)

    @staticmethod
    def _axis_scale_code(value: str, *, axis_name: str) -> int:
        norm = str(value).strip().lower()
        if norm in {"linear", "lin"}:
            return 0
        if norm in {"log", "log10"}:
            return 1
        if norm in {"time", "datetime", "date"}:
            return 2
        raise ValueError(f"{axis_name} axis scale must be 'linear', 'log', or 'time'.")

    @staticmethod
    def _scale_name(code: int) -> str:
        if int(code) == 1:
            return "log"
        if int(code) == 2:
            return "time"
        return "linear"

    @staticmethod
    def _axis_name(code: int) -> str:
        mapping = {
            0: "x1",
            1: "x2",
            2: "x3",
            3: "y1",
            4: "y2",
            5: "y3",
        }
        return mapping.get(int(code), "x1")

    @staticmethod
    def _marker_name(code: int) -> str:
        for name, value in _SERIES_MARKER_MAP.items():
            if int(value) == int(code):
                return name
        return "none"

    @staticmethod
    def _axis_code(value: str) -> int:
        norm = str(value).strip().lower()
        mapping = {
            "x1": 0,
            "x2": 1,
            "x3": 2,
            "y1": 3,
            "y2": 4,
            "y3": 5,
        }
        if norm not in mapping:
            raise ValueError("axis must be one of: x1, x2, x3, y1, y2, y3.")
        return mapping[norm]

    @classmethod
    def _axes_codes(cls, x_axis: str, y_axis: str) -> tuple[int, int]:
        x_code = cls._axis_code(x_axis)
        y_code = cls._axis_code(y_axis)
        if x_code > 2 or y_code < 3:
            raise ValueError("x_axis must be x1/x2/x3 and y_axis must be y1/y2/y3.")
        return x_code, y_code

    def _validate_subplot_index(self, subplot_index: int) -> int:
        idx = int(subplot_index)
        if idx < 0:
            raise ValueError("subplot_index must be >= 0.")
        max_idx = self._subplot_rows * self._subplot_cols - 1
        if idx > max_idx:
            raise ValueError(
                f"subplot_index={idx} is out of range for {self._subplot_rows}x{self._subplot_cols} subplots."
            )
        return idx

    def _consume_hide_next_item(self) -> bool:
        hidden = bool(self._hide_next_item)
        self._hide_next_item = False
        return hidden

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Plot is closed.")

    def _handle_frontend_message(
        self, _: anywidget.AnyWidget, content: Any, buffers: Any
    ) -> None:
        del buffers
        if self._closed or not isinstance(content, dict):
            return
        msg_type = content.get("type")
        if msg_type == "frontend_ready":
            include_wasm_assets = bool(content.get("need_wasm_assets", True))
            self._replay_state_to_frontend(include_wasm_assets=include_wasm_assets)
            return
        if msg_type == "perf_stats":
            if not self._perf_callbacks:
                return
            try:
                stats = {
                    "fps": float(content.get("fps", 0.0)),
                    "lod_ms": float(content.get("lod_ms", 0.0)),
                    "segment_build_ms": float(content.get("segment_build_ms", 0.0)),
                    "render_ms": float(content.get("render_ms", 0.0)),
                    "frame_ms": float(content.get("frame_ms", 0.0)),
                    "draw_points": float(content.get("draw_points", 0.0)),
                    "draw_segments": float(content.get("draw_segments", 0.0)),
                    "primitive_count": float(content.get("primitive_count", 0.0)),
                    "pixel_width": float(content.get("pixel_width", 0.0)),
                }
            except (TypeError, ValueError):
                return
            for callback in self._perf_callbacks:
                callback(self, stats)
            return
        if msg_type == "interaction_update":
            tools = content.get("tools", [])
            if isinstance(tools, list) and self._tool_callbacks:
                for tool in tools:
                    if isinstance(tool, dict):
                        payload = dict(tool)
                        for callback in self._tool_callbacks:
                            callback(self, payload)
            selection = content.get("selection")
            if isinstance(selection, dict) and self._selection_callbacks:
                try:
                    series_payload: list[dict[str, Any]] = []
                    raw_series = selection.get("series", [])
                    if isinstance(raw_series, list):
                        for item in raw_series:
                            if not isinstance(item, dict):
                                continue
                            series_payload.append(
                                {
                                    "series_id": str(item.get("series_id", "")),
                                    "series_name": str(item.get("series_name", "")),
                                    "series_token": int(item.get("series_token", 0)),
                                    "subplot_index": int(item.get("subplot_index", 0)),
                                    "index_min": int(item.get("index_min", -1)),
                                    "index_max": int(item.get("index_max", -1)),
                                    "count": int(item.get("count", 0)),
                                    "has_x": bool(item.get("has_x", False)),
                                }
                            )
                    selection_payload = {
                        "subplot_index": int(selection.get("subplot_index", 0)),
                        "x_min": float(selection.get("x_min", 0.0)),
                        "x_max": float(selection.get("x_max", 0.0)),
                        "y_min": float(selection.get("y_min", 0.0)),
                        "y_max": float(selection.get("y_max", 0.0)),
                        "series": series_payload,
                    }
                except (TypeError, ValueError):
                    return
                for callback in self._selection_callbacks:
                    callback(self, selection_payload)
            hover = content.get("hover")
            if isinstance(hover, dict) and self._hover_callbacks:
                try:
                    hover_payload = {
                        "series_id": str(hover.get("series_id", "")),
                        "series_name": str(hover.get("series_name", "")),
                        "series_token": int(hover.get("series_token", 0)),
                        "subplot_index": int(hover.get("subplot_index", 0)),
                        "active": bool(hover.get("active", True)),
                        "x": float(hover.get("x", 0.0)),
                        "y": float(hover.get("y", 0.0)),
                        "index": int(hover.get("index", -1)),
                        "distance_px": float(hover.get("distance_px", 0.0)),
                    }
                except (TypeError, ValueError):
                    return
                for callback in self._hover_callbacks:
                    callback(self, hover_payload)
            click = content.get("click")
            if isinstance(click, dict) and self._click_callbacks:
                try:
                    click_payload = {
                        "series_id": str(click.get("series_id", "")),
                        "series_name": str(click.get("series_name", "")),
                        "series_token": int(click.get("series_token", 0)),
                        "subplot_index": int(click.get("subplot_index", 0)),
                        "active": bool(click.get("active", True)),
                        "x": float(click.get("x", 0.0)),
                        "y": float(click.get("y", 0.0)),
                        "button": int(click.get("button", 0)),
                        "index": int(click.get("index", -1)),
                    }
                except (TypeError, ValueError):
                    return
                for callback in self._click_callbacks:
                    callback(self, click_payload)
            return
        if msg_type != "view_change":
            return
        try:
            view = {
                "x_min": float(content["x_min"]),
                "x_max": float(content["x_max"]),
                "y_min": float(content["y_min"]),
                "y_max": float(content["y_max"]),
            }
        except (KeyError, TypeError, ValueError):
            return
        self._last_view = dict(view)
        for callback in self._view_callbacks:
            callback(self, view)

    def _replay_state_to_frontend(self, *, include_wasm_assets: bool = True) -> None:
        if include_wasm_assets:
            wasm_js, wasm_bin, reason = _get_wasm_assets()
            if wasm_js is None or wasm_bin is None:
                self.send(
                    {"type": "wasm_assets_missing", "reason": reason or "WASM assets missing."}
                )
            else:
                self.send(
                    {"type": "wasm_assets", "wasm_js_source": wasm_js},
                    buffers=[memoryview(wasm_bin)],
                )

        self.send(
            {
                "type": "plot_options",
                "plot_flags": int(self._plot_flags),
                "axis_scale_x": int(self._axis_scale_x),
                "axis_scale_y": int(self._axis_scale_y),
            }
        )
        self.send(
            {
                "type": "subplots_config",
                "rows": int(self._subplot_rows),
                "cols": int(self._subplot_cols),
                "flags": int(self._subplot_flags),
            }
        )
        self.send(
            {
                "type": "aligned_group",
                "group_id": self._aligned_group_id,
                "enabled": bool(self._aligned_enabled),
                "vertical": bool(self._aligned_vertical),
            }
        )
        self.send({"type": "colormap", "name": self._colormap_name})
        if self._theme_name:
            self.send({"type": "theme", "name": self._theme_name})
        self.send({"type": "linked_crosshair", **self._linked_crosshair})
        for axis_idx in range(6):
            enabled, scale = self._axis_state.get(axis_idx, (axis_idx in (0, 3), 0))
            self.send(
                {
                    "type": "axis_state",
                    "axis": int(axis_idx),
                    "enabled": bool(enabled),
                    "scale": int(scale),
                }
            )
        for axis_idx, label in self._axis_labels.items():
            self.send({"type": "axis_label", "axis": int(axis_idx), "label": str(label)})
        for axis_idx, fmt in self._axis_formats.items():
            self.send({"type": "axis_format", "axis": int(axis_idx), "format": str(fmt)})
        for axis_idx, (ticks, labels, keep_default) in self._axis_ticks.items():
            self.send(
                {
                    "type": "axis_ticks",
                    "axis": int(axis_idx),
                    "count": int(ticks.size),
                    "labels": [str(s) for s in labels],
                    "keep_default": bool(keep_default),
                },
                buffers=[memoryview(ticks)],
            )
        for axis_idx, (enabled, vmin, vmax) in self._axis_limits_constraints.items():
            self.send(
                {
                    "type": "axis_limits_constraints",
                    "axis": int(axis_idx),
                    "enabled": bool(enabled),
                    "min": float(vmin),
                    "max": float(vmax),
                }
            )
        for axis_idx, (enabled, zmin, zmax) in self._axis_zoom_constraints.items():
            self.send(
                {
                    "type": "axis_zoom_constraints",
                    "axis": int(axis_idx),
                    "enabled": bool(enabled),
                    "min": float(zmin),
                    "max": float(zmax),
                }
            )
        for axis_idx, target in self._axis_links.items():
            self.send(
                {
                    "type": "axis_link",
                    "axis": int(axis_idx),
                    "target_axis": int(target),
                }
            )

        if self._perf_reporting_enabled:
            self.send(
                {
                    "type": "set_perf_reporting",
                    "enabled": True,
                    "interval_ms": self._perf_interval_ms,
                }
            )

        for series_id, meta in self._series.items():
            self.send(
                {
                    "type": "line",
                    "series_id": series_id,
                    "name": meta.name,
                    "subplot_index": int(meta.subplot_index),
                    "x_axis": int(meta.x_axis),
                    "y_axis": int(meta.y_axis),
                    "dtype": meta.dtype,
                    "length": meta.length,
                    "has_x": bool(meta.x_data is not None),
                    "hidden": bool(meta.hidden),
                    "max_points": 0 if meta.stream_capacity is None else int(meta.stream_capacity),
                    **self._series_style_payload(meta),
                },
                buffers=[memoryview(meta.x_data), memoryview(meta.data)]
                if meta.x_data is not None
                else [memoryview(meta.data)],
            )
        for meta in self._primitives.values():
            self.send(
                meta.content,
                buffers=[memoryview(b) for b in meta.buffers],
            )

    def _send_xy_primitive(
        self,
        kind: str,
        *,
        name: str,
        y: Any,
        x: Any | None = None,
        subplot_index: int = 0,
        x_axis: str = "x1",
        y_axis: str = "y1",
        **extra: Any,
    ) -> None:
        values_y = _to_float32_1d(y, arg_name="y")
        has_x = x is not None
        x_axis_code, y_axis_code = self._axes_codes(x_axis, y_axis)
        buffers: list[np.ndarray] = [values_y]
        if has_x:
            values_x = _to_float32_1d(x, arg_name="x")
            if values_x.size != values_y.size:
                raise ValueError("x and y arrays must have the same length.")
            buffers.insert(0, values_x)
        content: dict[str, Any] = {
            "type": "primitive_add",
            "kind": kind,
            "name": name,
            "length": int(values_y.size),
            "has_x": bool(has_x),
            "subplot_index": self._validate_subplot_index(subplot_index),
            "x_axis": int(x_axis_code),
            "y_axis": int(y_axis_code),
        }
        content.update(extra)
        self._send_primitive(content, buffers)

    def _send_primitive(self, content: dict[str, Any], buffers: list[np.ndarray]) -> None:
        self._ensure_open()
        payload = dict(content)
        if "hidden" not in payload:
            payload["hidden"] = bool(self._consume_hide_next_item())
        else:
            payload["hidden"] = bool(payload["hidden"])
        primitive_id = f"p{next(self._primitive_counter)}"
        payload["primitive_id"] = primitive_id
        self._primitives[primitive_id] = _PrimitiveMeta(
            content=payload,
            buffers=[np.ascontiguousarray(b, dtype=np.float32) for b in buffers],
        )
        self.send(payload, buffers=[memoryview(b) for b in self._primitives[primitive_id].buffers])


class _SubplotProxy:
    """Proxy object that routes plotting calls to one native ImPlot subplot cell."""

    _ROUTED_METHODS = {
        "line",
        "stream_line",
        "scatter",
        "bubbles",
        "stairs",
        "stems",
        "digital",
        "bars",
        "bar_groups",
        "bars_h",
        "shaded",
        "error_bars",
        "error_bars_h",
        "inf_lines",
        "vlines",
        "hlines",
        "histogram",
        "histogram2d",
        "heatmap",
        "image",
        "pie_chart",
        "text",
        "annotation",
        "dummy",
        "tag_x",
        "tag_y",
        "colormap_slider",
        "colormap_button",
        "colormap_selector",
        "drag_drop_plot",
        "drag_drop_axis",
        "drag_drop_legend",
        "drag_line_x",
        "drag_line_y",
        "drag_point",
        "drag_rect",
    }

    def __init__(self, plot: Plot, subplot_index: int) -> None:
        self._plot = plot
        self._subplot_index = int(subplot_index)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._plot, name)
        if not callable(attr) or name not in self._ROUTED_METHODS:
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("subplot_index", self._subplot_index)
            return attr(*args, **kwargs)

        return _wrapped


class Subplots:
    """Native ImPlot subplots rendered in a single Plot widget."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        title: str = "",
        width: int = 900,
        height: int = 450,
        link_rows: bool = False,
        link_cols: bool = False,
        link_all_x: bool = False,
        link_all_y: bool = False,
        share_items: bool = False,
        no_legend: bool = False,
        no_menus: bool = False,
        no_resize: bool = False,
        no_align: bool = False,
        col_major: bool = False,
        **plot_kwargs: Any,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be > 0.")
        self.title = str(title)
        self.rows = int(rows)
        self.cols = int(cols)
        self.link_rows = bool(link_rows)
        self.link_cols = bool(link_cols)
        self.link_all_x = bool(link_all_x)
        self.link_all_y = bool(link_all_y)
        self.share_items = bool(share_items)
        self.no_legend = bool(no_legend)
        self.no_menus = bool(no_menus)
        self.no_resize = bool(no_resize)
        self.no_align = bool(no_align)
        self.col_major = bool(col_major)

        subplot_flags = 0
        if self.no_legend:
            subplot_flags |= _SUBPLOT_FLAG_NO_LEGEND
        if self.no_menus:
            subplot_flags |= _SUBPLOT_FLAG_NO_MENUS
        if self.no_resize:
            subplot_flags |= _SUBPLOT_FLAG_NO_RESIZE
        if self.no_align:
            subplot_flags |= _SUBPLOT_FLAG_NO_ALIGN
        if self.share_items:
            subplot_flags |= _SUBPLOT_FLAG_SHARE_ITEMS
        if self.link_rows:
            subplot_flags |= _SUBPLOT_FLAG_LINK_ROWS
        if self.link_cols:
            subplot_flags |= _SUBPLOT_FLAG_LINK_COLS
        if self.link_all_x:
            subplot_flags |= _SUBPLOT_FLAG_LINK_ALL_X
        if self.link_all_y:
            subplot_flags |= _SUBPLOT_FLAG_LINK_ALL_Y
        if self.col_major:
            subplot_flags |= _SUBPLOT_FLAG_COL_MAJOR

        self._plot = Plot(width=width, height=height, title=self.title, **plot_kwargs)
        self._plot.set_subplots_config(rows=self.rows, cols=self.cols, flags=subplot_flags)

        self._cells: dict[tuple[int, int], _SubplotProxy] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                subplot_index = c * self.rows + r if self.col_major else r * self.cols + c
                self._cells[(r, c)] = _SubplotProxy(self._plot, subplot_index=subplot_index)

    def subplot(self, row: int, col: int) -> _SubplotProxy:
        key = (int(row), int(col))
        if key not in self._cells:
            raise IndexError(f"subplot index out of range: ({row}, {col})")
        return self._cells[key]

    def show(self) -> None:
        self._plot.show()
        return None

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return (
            f"nbimplot.{cls_name}(rows={self.rows}, cols={self.cols}, title={self.title!r}, "
            f"renderer='wasm-implot')"
        )

    def _repr_mimebundle_(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
        return _with_text_plain(self._plot._repr_mimebundle_(**kwargs), repr(self), kwargs)

    def render(self) -> None:
        self._plot.render()

    def export_png(self, filename: str = "nbimplot.png") -> None:
        self._plot.export_png(filename)

    def copy_png_to_clipboard(self) -> None:
        self._plot.copy_png_to_clipboard()

    def set_theme(self, name: str = "nbimplot") -> None:
        self._plot.set_theme(name)

    def set_linked_crosshair(
        self,
        group_id: str = "default",
        *,
        enabled: bool = True,
        axis: str = "x",
    ) -> None:
        self._plot.set_linked_crosshair(group_id, enabled=enabled, axis=axis)

    def get_state(self, *, include_data: bool = False) -> dict[str, Any]:
        return self._plot.get_state(include_data=include_data)

    def set_state(self, state: dict[str, Any]) -> None:
        self._plot.set_state(state)

    def export_json_state(
        self,
        filename: str | pathlib.Path | None = None,
        *,
        include_data: bool = False,
    ) -> str:
        return self._plot.export_json_state(filename, include_data=include_data)

    def export_csv_selection(
        self,
        selection: dict[str, Any],
        series: str | LineHandle | None = None,
        *,
        filename: str | pathlib.Path | None = None,
    ) -> str:
        return self._plot.export_csv_selection(selection, series, filename=filename)

    def highlight_selection(
        self,
        selection: dict[str, Any],
        series: str | LineHandle | None = None,
        *,
        name: str = "selection",
        points: bool = True,
        rect: bool = True,
    ) -> None:
        self._plot.highlight_selection(selection, series, name=name, points=points, rect=rect)


class AlignedPlots(Subplots):
    """Compatibility wrapper for aligned-plot workflows via native subplots."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        group_id: str = "aligned",
        vertical: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(rows, cols, **kwargs)
        self._plot.set_aligned_group(group_id, enabled=True, vertical=vertical)


class Dashboard(Subplots):
    """Convenience wrapper for linked, realtime dashboard layouts."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        title: str = "Dashboard",
        width: int = 1100,
        height: int = 650,
        link_x: bool = True,
        link_y: bool = False,
        crosshairs: bool = True,
        linked_crosshair: bool = True,
        crosshair_group: str = "dashboard",
        theme: str | None = "nbimplot",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("link_all_x", bool(link_x))
        kwargs.setdefault("link_all_y", bool(link_y))
        super().__init__(rows, cols, title=title, width=width, height=height, **kwargs)
        if crosshairs:
            self._plot._plot_flags |= _PLOT_FLAG_CROSSHAIRS
            self._plot.send(
                {
                    "type": "plot_options",
                    "plot_flags": int(self._plot._plot_flags),
                    "axis_scale_x": int(self._plot._axis_scale_x),
                    "axis_scale_y": int(self._plot._axis_scale_y),
                }
            )
        if theme:
            self._plot.set_theme(theme)
        if linked_crosshair:
            self._plot.set_linked_crosshair(crosshair_group, enabled=True, axis="x")
