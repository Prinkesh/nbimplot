# nbimplot

Jupyter-native, ImPlot-powered plotting for very large arrays.

`nbimplot` is built around three constraints:

- notebook cell rendering only (no native windows, no side process)
- binary transfer (`numpy -> bytes -> wasm heap`)
- WASM-owned interaction, state, and LOD

The runtime is strict: ImPlot + WASM + WebGL2 are required.

Project note: `nbimplot` was vibe coded with Codex, then
hardened with tests, packaging checks, and runtime validation.

## Upstream Libraries

`nbimplot` is built on top of these upstream projects:

- Dear ImGui: https://github.com/ocornut/imgui
- ImPlot: https://github.com/epezent/implot

## Thanks

Special thanks to the ImPlot and Dear ImGui maintainers and contributors for
building and maintaining the core libraries that make `nbimplot` possible.

## Install

```bash
python -m pip install -U nbimplot
```

Minimum recommended widget/runtime stack:

```bash
python -m pip install -U "nbimplot>=0.1.8" "anywidget>=0.9.21" ipywidgets jupyterlab_widgets
```

## Compatibility

- Python `>=3.10`
- Jupyter widget stack: `anywidget`, `ipywidgets`, `traitlets`
- Frontend: JupyterLab/Notebook with widget manager enabled
- Browser/GPU: WebGL2 required
- Runtime mode: strict WASM + ImPlot only (no JS renderer fallback)

## Quick Start

```python
import numpy as np
import nbimplot as ip

y = np.sin(np.linspace(0, 100, 1_000_000, dtype=np.float32))

p = ip.Plot(width=900, height=450, title="Signal")
h = p.line("mid", y)
p.show()

# Update in place, then redraw
h.set_data((0.8 * y).astype(np.float32))
p.render()
```

## Interaction Defaults

- initial X/Y view auto-fits to available data
- double-click inside plot area resets view (`autoscale`)
- right-drag box zoom, wheel zoom, drag pan, legend toggle

## Core API

```python
import numpy as np
import nbimplot as ip

y = np.random.randn(200_000).astype(np.float32).cumsum()

p = ip.Plot(width=1000, height=420, title="Core API")
h = p.line("price", y, color="#22c55e", line_weight=2.0, marker="none")
p.set_plot_flags(no_legend=False, no_menus=False, no_box_select=False)
p.set_colormap("Viridis")
p.show()

# Later update
h.set_data((y * 1.01).astype(np.float32))
p.render()
```

## Common Examples

### 1) Line + Streaming

```python
import numpy as np
import nbimplot as ip

p = ip.Plot(width=1000, height=380, title="Streaming")
h = p.stream_line("ticks", capacity=200_000, initial=np.zeros(1000, dtype=np.float32))
p.show()

chunk = np.random.randn(20_000).astype(np.float32)
h.append(chunk)
p.render()
```

### 2) Scatter / Bars / Histogram

```python
import numpy as np
import nbimplot as ip

rng = np.random.default_rng(7)
x = rng.normal(0, 1, 4000).astype(np.float32)
y = (0.5 * x + 0.2 * rng.normal(size=x.size)).astype(np.float32)

p = ip.Plot(width=1100, height=420, title="Stat Plots")
p.scatter("cloud", y, x=x, size=2.0)
p.vlines("cuts", np.array([-1.0, 1.0], dtype=np.float32))
p.hlines("zero", np.array([0.0], dtype=np.float32))
p.show()

p2 = ip.Plot(width=1100, height=360, title="Histogram")
p2.histogram("x-dist", x, bins=60)
p2.show()
```

### 3) Heatmap / Histogram2D / Image

```python
import numpy as np
import nbimplot as ip

rng = np.random.default_rng(0)
z = rng.normal(size=(50, 80)).astype(np.float32)

p = ip.Plot(width=1100, height=420, title="Heatmap")
p.set_colormap("Plasma")
p.heatmap(
    "z",
    z,
    label_fmt="",  # empty format disables cell text
    show_colorbar=True,
    colorbar_label="Intensity",
    colorbar_format="%.3f",
)
p.show()

x = rng.normal(size=200_000).astype(np.float32)
y = (0.3 * x + rng.normal(size=x.size)).astype(np.float32)
p2 = ip.Plot(width=1100, height=420, title="Histogram2D")
p2.histogram2d(
    "h2d",
    x,
    y,
    x_bins=100,
    y_bins=80,
    label_fmt="",
    show_colorbar=True,
    colorbar_label="Count",
)
p2.show()
```

### 4) Subplots

```python
import numpy as np
import nbimplot as ip

sp = ip.Subplots(
    2,
    2,
    title="Dashboard",
    width=1100,
    height=760,
    link_rows=True,
    link_cols=True,
    share_items=True,
)

t = np.linspace(0, 30, 4000, dtype=np.float32)
sp.subplot(0, 0).line("sin", np.sin(t))
sp.subplot(0, 1).scatter("noise", np.random.randn(3000).astype(np.float32))
sp.subplot(1, 0).bars("bars", np.abs(np.random.randn(120)).astype(np.float32))
sp.subplot(1, 1).histogram("hist", np.random.randn(20_000).astype(np.float32), bins=50)
sp.show()
```

## Plot and Primitive Coverage

Implemented plot/primitive APIs include:

- `line`, `stream_line`
- `scatter`, `bubbles`, `stairs`, `stems`, `digital`
- `bars`, `bar_groups`, `bars_h`, `shaded`
- `error_bars`, `error_bars_h`
- `inf_lines`, `vlines`, `hlines`
- `histogram`, `histogram2d`, `heatmap`, `image`, `pie_chart`
- `text`, `annotation`, `dummy`
- `tag_x`, `tag_y`, `colormap_slider`, `colormap_button`, `colormap_selector`
- `drag_line_x`, `drag_line_y`, `drag_point`, `drag_rect`
- `drag_drop_plot`, `drag_drop_axis`, `drag_drop_legend`

For a broader cookbook, see `docs/EXAMPLES.md`.

## View, Axes, and Performance Controls

- `p.set_view(x_min, x_max, y_min, y_max)`
- `p.autoscale()`
- `p.set_axis_scale(x="linear|log", y="linear|log")`
- `p.set_axis_state("x2|x3|y2|y3", enabled=True|False, scale="linear|log|time")`
- `p.set_secondary_axes(x2=..., x3=..., y2=..., y3=...)`
- `p.set_time_axis("x1|x2|x3|y1|y2|y3")`
- `p.set_axis_label(...)`, `p.set_axis_format(...)`
- `p.set_axis_ticks(...)`, `p.clear_axis_ticks(...)`
- `p.set_axis_limits_constraints(...)`, `p.set_axis_zoom_constraints(...)`, `p.set_axis_link(...)`
- `p.hide_next_item()`
- `p.on_perf_stats(callback, interval_ms=500)`
- `p.on_tool_change(callback)`
- `p.on_selection_change(callback)`

## Example Notebooks

- `notebooks/nbimplot_examples.ipynb`
- `notebooks/nbimplot_api_gallery.ipynb`
- `notebooks/nbimplot_benchmarks.ipynb`

## Troubleshooting

### `Failed to load model class 'AnyModel' from module 'anywidget'`

This is usually a server-kernel env mismatch or stale lab assets.

```bash
python -m pip install -U "nbimplot>=0.1.8" "anywidget>=0.9.21" ipywidgets jupyterlab_widgets
jupyter lab clean
```

Then restart the full JupyterLab server.

Quick verification:

```python
import nbimplot as ip, anywidget, sys
print("python:", sys.executable)
print("anywidget:", anywidget.__version__)
print("has Plot:", hasattr(ip, "Plot"))
```

### `Unable to enable ImPlot in the WASM core` or WebGL context errors

Strict mode requires WebGL2.

```javascript
!!document.createElement("canvas").getContext("webgl2")
```

If this is `false`, run from a local desktop browser session with GPU
acceleration enabled.

## Known Limitations

- Rendering requires WebGL2-capable browser/runtime.
- Strict mode is enforced; non-ImPlot or non-WASM fallback is disabled.
- Headless/browser-restricted environments may fail to create GL context.

## Build WASM Core

Prerequisites:

- Emscripten SDK (`emcmake`, `emcc`)
- CMake >= 3.20
- Dear ImGui sources (`imgui`): https://github.com/ocornut/imgui
- ImPlot sources (`implot`): https://github.com/epezent/implot

Build:

```bash
scripts/build_wasm.sh
```

Expected outputs in `nbimplot/wasm/`:

- `nbimplot_wasm.js`
- `nbimplot_wasm.wasm`

Build explicitly with local ImGui/ImPlot sources:

```bash
NBIMPLOT_WITH_IMPLOT=ON \
NBIMPLOT_IMGUI_DIR=/path/to/imgui \
NBIMPLOT_IMPLOT_DIR=/path/to/implot \
scripts/build_wasm.sh
```

If you use vendored deps:

```bash
NBIMPLOT_WITH_IMPLOT=ON scripts/build_wasm.sh
```

## Performance Model

- raw path when visible points `<= 3 * pixel_width`
- min/max LOD when visible points `> 3 * pixel_width`
- LOD computed in WASM, complexity scales with screen pixels
