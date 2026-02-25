# nbimplot Examples Cookbook

All snippets below are valid against the current `nbimplot` API.

Project note: `nbimplot` was vibe coded with Codex, and
these examples are kept aligned with tested API signatures.

## Setup

```python
import numpy as np
import nbimplot as ip
```

## Line and In-Place Updates

```python
t = np.linspace(0, 100, 1_000_000, dtype=np.float32)
y = np.sin(t)

p = ip.Plot(width=1000, height=420, title="Line + Update")
h = p.line("signal", y, color="#3b82f6", line_weight=2.0)
p.show()

h.set_data((0.7 * y).astype(np.float32))
p.render()
```

## Streaming Ring Buffer

```python
p = ip.Plot(width=1000, height=360, title="Realtime")
h = p.stream_line("stream", capacity=200_000, initial=np.zeros(5000, dtype=np.float32))
p.show()

chunk = (0.1 * np.random.randn(20_000)).astype(np.float32)
h.append(chunk)
p.render()
```

## Scatter / Bubbles

```python
rng = np.random.default_rng(42)
x = rng.normal(0, 1, 8000).astype(np.float32)
y = (0.6 * x + rng.normal(0, 0.4, x.size)).astype(np.float32)
s = (8.0 + 12.0 * np.abs(rng.normal(size=x.size))).astype(np.float32)

p = ip.Plot(width=1100, height=420, title="Scatter + Bubbles")
p.scatter("scatter", y, x=x, size=2.5)
p.bubbles("bubbles", y, s, x=x)
p.show()
```

## Stairs / Stems / Digital

```python
x = np.arange(0, 120, dtype=np.float32)
y = np.sin(x * 0.1).astype(np.float32)
digital = (y > 0).astype(np.float32)

p = ip.Plot(width=1100, height=420, title="Discrete")
p.stairs("stairs", y, x=x)
p.stems("stems", y, x=x)
p.digital("digital", digital, x=x)
p.show()
```

## Bars / Horizontal Bars / Grouped Bars

```python
rng = np.random.default_rng(5)
y = np.abs(rng.normal(size=15)).astype(np.float32)

p = ip.Plot(width=1000, height=380, title="Bars")
p.bars("bars", y, bar_width=0.8)
p.bars_h("bars_h", y)
p.show()

p2 = ip.Plot(width=1000, height=380, title="Grouped Bars")
labels = ["A", "B", "C"]
vals = np.array(
    [
        [4.0, 5.5, 3.8],
        [2.4, 4.8, 5.2],
        [3.1, 2.7, 4.9],
        [5.4, 4.2, 3.7],
    ],
    dtype=np.float32,
)
p2.bar_groups(labels, vals, group_size=0.8)
p2.show()
```

## Shaded Region and Error Bars

```python
x = np.linspace(0, 20, 500, dtype=np.float32)
y = np.sin(x).astype(np.float32)
err = (0.1 + 0.05 * np.abs(np.cos(x))).astype(np.float32)

p = ip.Plot(width=1100, height=420, title="Shaded + Error")
p.shaded("band", y + err, y - err, x=x, alpha=0.2)
p.error_bars("err", y, err=err, x=x)
p.error_bars_h("err_h", x, err=err, y=y)
p.show()
```

## Infinite / Vertical / Horizontal Reference Lines

```python
y = np.cumsum(np.random.randn(1000).astype(np.float32))
p = ip.Plot(width=1000, height=360, title="Reference Lines")
p.line("series", y)
p.inf_lines("infx", np.array([200, 500, 800], dtype=np.float32), axis="x")
p.vlines("vlines", np.array([100, 300, 700], dtype=np.float32))
p.hlines("hlines", np.array([0], dtype=np.float32))
p.show()
```

## Histogram and Histogram2D

```python
rng = np.random.default_rng(9)
x = rng.normal(size=100_000).astype(np.float32)
y = (0.4 * x + rng.normal(size=x.size)).astype(np.float32)

p = ip.Plot(width=1000, height=360, title="Histogram")
p.histogram("hist", x, bins=80)
p.show()

p2 = ip.Plot(width=1000, height=420, title="Histogram2D")
p2.set_colormap("Viridis")
p2.histogram2d(
    "h2d",
    x,
    y,
    x_bins=100,
    y_bins=80,
    label_fmt="",
    show_colorbar=True,
    colorbar_label="Count",
    colorbar_format="%g",
)
p2.show()
```

## Heatmap and Colormap Controls

```python
rng = np.random.default_rng(123)
z = rng.normal(size=(40, 60)).astype(np.float32)

p = ip.Plot(width=1000, height=420, title="Heatmap")
p.set_colormap("Plasma")
p.heatmap(
    "heat",
    z,
    label_fmt="",  # hide per-cell text
    show_colorbar=True,
    colorbar_label="Intensity",
    colorbar_format="%.2f",
)
p.colormap_slider(label="cmap", t=0.5)
p.colormap_button(label="preview")
p.colormap_selector(label="select")
p.show()
```

## Image / Texture Plot

```python
h, w = 80, 120
img = np.zeros((h, w, 3), dtype=np.float32)
img[..., 0] = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
img[..., 1] = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
img[..., 2] = 0.25

p = ip.Plot(width=900, height=420, title="Image")
p.image(
    "rgb",
    img,
    bounds=((0.0, 0.0), (10.0, 8.0)),
    uv0=(0.0, 0.0),
    uv1=(1.0, 1.0),
)
p.show()
```

## Pie / Text / Annotation / Tags

```python
p = ip.Plot(width=1000, height=420, title="Text and Pie")
p.pie_chart("pie", np.array([30, 20, 15, 35], dtype=np.float32), labels=["A", "B", "C", "D"])
p.text("origin", 0.0, 0.0)
p.annotation("peak", 2.0, 3.0, offset_x=10, offset_y=-10)
p.tag_x(1.5, label_fmt="x=%.2f")
p.tag_y(0.0, label_fmt="y=%.2f")
p.show()
```

## Drag Tools

```python
y = np.sin(np.linspace(0, 10, 1000, dtype=np.float32))
p = ip.Plot(width=1000, height=420, title="Drag Tools")
p.line("signal", y)
p.drag_line_x("x-threshold", 200.0)
p.drag_line_y("y-threshold", 0.5)
p.drag_point("cursor", 250.0, 0.0, size=6.0)
p.drag_rect("window", 150.0, -0.5, 350.0, 0.5)
p.show()
```

## Axis, View, and Tick Controls

```python
y = np.cumsum(np.random.randn(50_000).astype(np.float32))
p = ip.Plot(width=1100, height=420, title="Axes")
p.line("series", y)

p.set_secondary_axes(x2=True, y2=True)
p.set_axis_label("x1", "Index")
p.set_axis_label("y1", "Value")
p.set_axis_format("y1", "%.3f")
p.set_axis_ticks("x1", np.array([0, 10_000, 20_000, 30_000, 40_000], dtype=np.float32))
p.set_axis_limits_constraints("x1", 0.0, float(len(y)))
p.set_axis_zoom_constraints("x1", 50.0, 200_000.0)
p.set_axis_link("y2", "y1")

p.show()
p.set_view(1000, 5000, float(y.min()), float(y.max()))
p.render()
```

## Subplots

```python
sp = ip.Subplots(
    2,
    2,
    title="Subplot Grid",
    width=1200,
    height=780,
    link_rows=True,
    link_cols=True,
    share_items=True,
)

sp.subplot(0, 0).line("line", np.sin(np.linspace(0, 40, 6000, dtype=np.float32)))
sp.subplot(0, 1).scatter("scatter", np.random.randn(3000).astype(np.float32))
sp.subplot(1, 0).histogram("hist", np.random.randn(25_000).astype(np.float32), bins=60)
sp.subplot(1, 1).heatmap("heat", np.random.randn(45, 75).astype(np.float32), label_fmt="")
sp.show()
```

## Event Callbacks

```python
p = ip.Plot(width=900, height=360, title="Callbacks")
p.line("y", np.sin(np.linspace(0, 10, 2000, dtype=np.float32)))

def on_perf(plot, stats):
    print("fps=", round(stats.get("fps", 0.0), 1))

def on_tool(plot, payload):
    print("tool:", payload)

def on_sel(plot, payload):
    print("selection:", payload)

p.on_perf_stats(on_perf, interval_ms=1000)
p.on_tool_change(on_tool)
p.on_selection_change(on_sel)
p.show()
```
