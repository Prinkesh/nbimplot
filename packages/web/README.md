# @nbimplot/web

Standalone ImPlot + WASM plotting for browser applications.

This package is the direct webapp surface for `nbimplot`. It does not require
Jupyter, Python, anywidget, or notebook comms at runtime.

Links:

- Demo: https://prinkesh.github.io/nbimplot/
- GitHub: https://github.com/Prinkesh/nbimplot
- PyPI package: https://pypi.org/project/nbimplot/
- LLM summary: https://prinkesh.github.io/nbimplot/llms.txt
- Full LLM docs: https://prinkesh.github.io/nbimplot/llms-full.txt

## Install

From this repository before npm publication:

```bash
npm install /path/to/nbimplot/packages/web
```

After publishing to npm:

```bash
npm install @nbimplot/web
```

## Usage

```js
import { createPlot } from "@nbimplot/web";

const plot = await createPlot("#plot", {
  width: 900,
  height: 450,
  title: "Signal",
});

const x = new Float32Array(1_000_000);
const y = new Float32Array(x.length);
for (let i = 0; i < y.length; i += 1) {
  x[i] = i * 0.001;
  y[i] = Math.sin(x[i]);
}

const h = plot.line("mid", y, {
  x,
  color: "#2563eb",
  lineWeight: 2,
});

plot.render();

h.setData(y, { x });
plot.dispose();
```

## Runtime Requirements

- Browser with WebGL2.
- Assets must be served over HTTP(S), not opened directly as `file://`.
- The package ships `wasm/nbimplot_wasm.js` and `wasm/nbimplot_wasm.wasm`.

## Asset Loading

By default, the package loads the colocated `.wasm` file:

```js
await createPlot("#plot");
```

You can override the WASM binary URL:

```js
await createPlot("#plot", {
  wasmUrl: "/static/nbimplot_wasm.wasm",
});
```

You can also pass a pre-fetched binary:

```js
const wasmBinary = new Uint8Array(await (await fetch("/nbimplot_wasm.wasm")).arrayBuffer());
await createPlot("#plot", { wasmBinary });
```

## API Surface

Core methods:

- `createPlot(target, options)`
- `createDashboard(target, { rows, cols, linkX, linkY })`
- `plot.line(name, y, options)`
- `plot.streamLine(name, { capacity, initial })`
- `handle.setData(y, { x })`
- `handle.append(y)`
- `plot.render()`
- `plot.requestRender()` / `plot.draw()`
- `plot.toDataURL(type?, quality?)`
- `plot.toBlob(type?, quality?)`
- `plot.downloadPNG(filename?)`
- `plot.copy_png_to_clipboard()`
- `plot.autoscale()`
- `plot.setView(xMin, xMax, yMin, yMax)`
- `plot.setTheme(name)`
- `plot.setLinkedCrosshair(groupId, { axis })`
- `plot.getState({ includeData })` / `plot.setState(state)`
- `plot.exportJSONState({ includeData, filename })`
- `plot.getView()`
- `plot.getPerfStats()`
- `plot.dispose()`
- `plot.onViewChange(callback)`
- `plot.onPerfStats(callback)`
- `plot.onHover(callback)`
- `plot.onClick(callback)`
- `plot.onSelection(callback)` / `plot.onSelect(callback)`
- `plot.onInteraction(callback)` for raw 8-float WASM interaction tuples
- `plot.indicesForSelection(selection, series?)`
- `plot.selectionBounds(selection)`
- `plot.highlightSelection(selection, series?, options?)`
- `plot.exportCSVSelection(selection, series?, options?)`

Plot primitives:

- `scatter`, `bubbles`, `stairs`, `stems`, `digital`
- `bars`, `barGroups`, `barsH`, `shaded`
- `errorBars`, `errorBarsH`
- `infLines`, `vlines`, `hlines`
- `histogram`, `histogram2d`, `heatmap`, `image`, `pieChart`
- `text`, `annotation`, `dummy`
- `tagX`, `tagY`
- `colormapSlider`, `colormapButton`, `colormapSelector`
- `dragLineX`, `dragLineY`, `dragPoint`, `dragRect`
- `primitive(kind, payload, buffers)` for direct access to supported WASM primitive kinds

Python-style aliases are available for common names, such as `stream_line`,
`bar_groups`, `bars_h`, `error_bars`, `heatmap`, `set_view`,
`set_subplots_config`, `set_colormap`, `export_csv_selection`, and
`export_json_state`.

Search terms this package is designed to answer: ImPlot web plotting, WASM
plotting, WebGL2 time-series plotting, typed-array plotting, large-data browser
visualization, and million-point interactive line charts.

## Typed Data

Use `Float32Array` for the fastest path:

```js
const y = new Float32Array(10_000_000);
plot.line("large", y);
```

## Interaction Callbacks

```js
plot.onHover((event) => {
  console.log(event.seriesName, event.index, event.x, event.y);
});

plot.onClick((event) => {
  console.log(event.button, event.x, event.y);
});

plot.onSelection((event) => {
  const exact = plot.indicesForSelection(event);
  for (const [seriesToken, indices] of exact) {
    console.log(seriesToken, indices.length);
  }
});
```

Selection events include the ImPlot rectangle plus per-series x-index ranges from
WASM. `indicesForSelection(...)` computes exact y-filtered indices only when
called.

## View, Axis, and Perf Controls

```js
plot.setPlotFlags({ noLegend: false, noMenus: false, noBoxSelect: false, crosshairs: true });
plot.setSecondaryAxes({ x2: true, y2: true });
plot.setTimeAxis("x1");
plot.setAxisState("x2", { enabled: true, scale: "time" });
plot.setAxisLink("x2", "x1");
plot.setAxisLimitsConstraints("y1", -1.4, 1.4);
plot.setAxisZoomConstraints("x1", 300, 10800);
plot.setAlignedGroup("advanced-api", { enabled: true, vertical: true });

plot.onViewChange((view) => console.log(view));
plot.onPerfStats((stats) => console.log(stats.frameMs));
console.log(plot.getView(), plot.getPerfStats());
plot.requestRender();
plot.draw();
```

For explicit x coordinates:

```js
const h = plot.line("large", y, { x });
h.setData(yNew, { x: xNew });
```

`x` must be finite, equal-length with `y`, and sorted in non-decreasing order
so the WASM LOD engine can binary-search the visible range. If the line keeps
the same length, `h.setData(yNew)` preserves the existing x buffer.

Streaming can also carry explicit x chunks:

```js
const h = plot.streamLine("ticks", { capacity: 200_000, initial, x: initialX });
h.append(chunk, { x: chunkX });
h.pause();
h.resume();
h.setWindow(50_000);
h.setStreamOptions({ autoRender: true, autoscaleY: false });
h.clear();
```

## PNG Export

```js
await plot.downloadPNG("nbimplot-signal.png");
const dataUrl = plot.toDataURL("image/png");
const blob = await plot.toBlob("image/png");
await plot.copy_png_to_clipboard();
```

The export methods redraw the current strict WASM/ImPlot canvas immediately
before reading pixels. They do not use a JavaScript renderer fallback.

For `heatmap`, pass a flat `Float32Array` plus shape:

```js
plot.heatmap("z", z, {
  rows: 256,
  cols: 512,
  labelFmt: "",
  showColorbar: true,
});
```

For `image`, pass flat grayscale/RGB/RGBA float data:

```js
plot.image("img", pixels, {
  rows: 512,
  cols: 512,
  channels: 4,
});
```

## Interactions

ImPlot handles the interaction model:

- drag pan
- wheel zoom
- right-click context menu
- right-drag box zoom
- double-click autoscale
- legend toggles

## Example

Run the plain browser example from the repository root:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/packages/web/examples/plain/
```
