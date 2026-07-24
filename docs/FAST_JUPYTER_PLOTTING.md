# Fast Jupyter Plotting With nbimplot

`nbimplot` is a Jupyter-native plotting library for users who need interactive plots inside notebook output cells while working with large numeric arrays.

## Why nbimplot Exists

Common notebook plotting stacks often become slow when the frontend receives very large JSON arrays or when interaction redraws every raw point. `nbimplot` takes a different path:

- Python validates NumPy arrays and sends binary buffers.
- JavaScript owns notebook widget lifecycle and canvas/event wiring.
- WASM owns plot state, data buffers, LOD, ImGui input, and ImPlot rendering.
- WebGL2 renders into the notebook output cell canvas.

The result is a small explicit API for notebooks with ImPlot-style interaction.

## Install

```bash
python -m pip install -U nbimplot anywidget ipywidgets jupyterlab_widgets
```

## Minimal Notebook Example

```python
import numpy as np
import nbimplot as ip

x = np.linspace(0, 100, 1_000_000, dtype=np.float32)
y = np.sin(x)

p = ip.Plot(width=900, height=450, title="Signal")
h = p.line("signal", y, x=x)
p.show()
```

## Update Existing Data

```python
h.set_data((0.8 * y).astype(np.float32), x=x)
p.render()
```

Same-length custom-x line updates may omit x:

```python
h.set_data((0.6 * y).astype(np.float32))
p.render()
```

If the new y array changes length, pass a replacement x array too.

## Streaming

Use `stream_line` when samples arrive in chunks and implicit sample index is acceptable:

```python
p = ip.Plot(width=1000, height=360, title="Streaming")
h = p.stream_line("ticks", capacity=200_000, initial=np.zeros(1000, dtype=np.float32))
p.show()

chunk = np.random.randn(20_000).astype(np.float32)
h.append(chunk)
p.render()
```

For custom timestamps or irregular x values, maintain your own x/y arrays and call `set_data(y, x=x)`.

## Interaction Checklist

Inside the notebook cell:

- left-drag pans
- mouse wheel zooms
- wheel over axes zooms one axis
- right-click opens ImPlot context menus
- right-drag uses ImPlot box zoom / selection behavior
- double-click autoscale fits data
- legend entries toggle visibility

## Recommended Data Practices

- Use `float32` NumPy arrays for the fastest path.
- Keep x sorted for line plots with explicit x.
- Avoid Python per-point loops; build vectors with NumPy.
- Reuse plot handles with `set_data` or `append` instead of recreating plots.
- Dispose/clear notebooks normally; widget lifecycle cleanup releases WASM state.

## When To Use Another Tool

Use Matplotlib for static publication plots. Use Plotly for declarative charts where broad browser compatibility matters more than strict WASM rendering. Use Datashader when server-side raster aggregation is the main task. Use `nbimplot` when notebook-cell interactivity and large-array responsiveness are the priority.
