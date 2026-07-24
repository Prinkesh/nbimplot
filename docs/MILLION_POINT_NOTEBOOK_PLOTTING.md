# Million-Point Notebook Plotting

`nbimplot` is designed around a simple performance rule: render only what the screen can show.

For line plots with many visible points, the WASM core switches from raw points to min/max LOD buckets. The target rendered point count is proportional to canvas width, not dataset length.

## Core Pattern

```python
import numpy as np
import nbimplot as ip

n = 1_000_000
x = np.linspace(0, 1000, n, dtype=np.float32)
y = (
    np.sin(x * 0.04)
    + 0.15 * np.sin(x * 0.31)
    + 0.03 * np.cos(x * 2.7)
).astype(np.float32)

p = ip.Plot(width=1100, height=420, title="Million Point Signal")
p.line("signal", y, x=x, color="#2563eb", line_weight=1.8)
p.show()
```

## Why Explicit X Matters

Explicit x data lets notebook users plot real time, irregular timestamps, or domain-specific coordinates:

```python
x = np.cumsum(np.random.default_rng(7).uniform(0.001, 0.010, 1_000_000)).astype(np.float32)
y = np.sin(x).astype(np.float32)
p.line("irregular", y, x=x)
```

The x array must be sorted non-decreasing. This lets the WASM core binary-search the visible range during pan/zoom and avoid scanning unrelated points.

## Update Without Recreating The Plot

```python
h = p.line("signal", y, x=x)

# Same length: existing x is preserved.
h.set_data((y * 0.8).astype(np.float32))
p.render()

# New length: replace x and y together.
x2 = np.linspace(0, 2000, 2_000_000, dtype=np.float32)
y2 = np.sin(x2).astype(np.float32)
h.set_data(y2, x=x2)
p.render()
```

## Preserving Data Quality

The default line LOD uses min/max buckets. For each visible pixel bucket, the core emits the minimum and maximum y values in that bucket. This preserves spikes and extrema that average or sampling-based downsampling can hide.

## Practical Notebook Checklist

- Use `np.float32` arrays.
- Send y-only data when x is the sample index.
- Send `x=` when x is time, distance, frequency, or another real coordinate.
- Keep x sorted and finite.
- Use `p.autoscale()` or double-click the plot area to refit.
- Use `on_perf_stats` if you need runtime draw counts and frame times during tuning.
