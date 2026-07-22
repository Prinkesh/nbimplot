# @nbimplot/web

Standalone ImPlot + WASM plotting for browser applications.

This package is the direct webapp surface for `nbimplot`. It does not require
Jupyter, Python, anywidget, or notebook comms at runtime.

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

const y = new Float32Array(1_000_000);
for (let i = 0; i < y.length; i += 1) {
  y[i] = Math.sin(i * 0.001);
}

const h = plot.line("mid", y, {
  color: "#2563eb",
  lineWeight: 2,
});

plot.render();

h.setData(y);
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
- `plot.line(name, y, options)`
- `plot.streamLine(name, { capacity, initial })`
- `handle.setData(y)`
- `handle.append(y)`
- `plot.render()`
- `plot.autoscale()`
- `plot.setView(xMin, xMax, yMin, yMax)`
- `plot.dispose()`

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
`bar_groups`, `bars_h`, `error_bars`, `heatmap`, `set_view`, and
`set_colormap`.

## Typed Data

Use `Float32Array` for the fastest path:

```js
const y = new Float32Array(10_000_000);
plot.line("large", y);
```

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
