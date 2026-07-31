# Direct Web App Usage

`nbimplot` can now be used outside notebooks through the standalone
`@nbimplot/web` package.

The notebook package and the web package share the same WASM core:

```text
Python/Jupyter API -> anywidget -> WASM core
Web app API       -> DOM canvas -> WASM core
```

The web package does not depend on Jupyter. It creates a canvas inside a DOM
element, loads the packaged WASM assets, forwards browser input into ImGui, and
lets ImPlot render directly into the canvas.

## Install

From this repository before npm publication:

```bash
npm install /path/to/nbimplot/packages/web
```

After publishing to npm:

```bash
npm install @nbimplot/web
```

## Minimal Example

```html
<div id="plot"></div>

<script type="module">
  import { createPlot } from "@nbimplot/web";

  const plot = await createPlot("#plot", {
    width: 900,
    height: 450,
    title: "Million Point Signal",
  });

  const x = new Float32Array(1_000_000);
  const y = new Float32Array(x.length);
  for (let i = 0; i < y.length; i += 1) {
    x[i] = i * 0.001;
    y[i] = Math.sin(x[i]) + 0.1 * Math.sin(i * 0.021);
  }

  plot.line("signal", y, {
    x,
    color: "#2563eb",
    lineWeight: 2,
  });

  plot.render();
</script>
```

## Vite / React / Vue / Svelte

Use `createPlot` from the mounted DOM element. Dispose the plot when the
component unmounts.

```js
import { createPlot } from "@nbimplot/web";

let plot;

async function mount(element) {
  plot = await createPlot(element, {
    width: 900,
    height: 420,
    responsive: true,
    title: "Signal",
  });

  const x = new Float32Array([0, 0.5, 1.4, 3.0, 4.2]);
  const y = new Float32Array([0, 1, 0, -1, 0]);
  plot.line("y", y, { x });
}

function unmount() {
  plot?.dispose();
}
```

## Data Path

The fastest path is always typed arrays:

```js
const y = new Float32Array(10_000_000);
plot.line("large", y);
```

`line` uses implicit X values (`0..N-1`) by default. Pass `{ x }` for explicit
line coordinates:

```js
const h = plot.line("large", y, { x });
h.setData(yNew, { x: xNew });
```

Line `x` buffers must be finite, same length as `y`, and sorted in
non-decreasing order. `xAxis` chooses the ImPlot axis slot (`x1`, `x2`, `x3`);
it is separate from the x-data buffer.

Batch related time-series with `lines(...)`:

```js
plot.lines({
  mid: { x, y: mid, color: "#1f6f66" },
  vwap: { x, y: vwap, color: "#b74b2b" },
});
```

`Date` arrays become ImPlot time axes automatically. String/category arrays are
mapped to numeric ticks and labels:

```js
plot.line("session latency", latency, { x: dateArray });
plot.scatter("model scores", scores, { x: ["baseline", "candidate", "prod"] });
```

## Heatmaps and Images

Flat arrays need explicit shape:

```js
plot.heatmap("z", z, {
  rows: 256,
  cols: 512,
  labelFmt: "",
  showColorbar: true,
  colorbarLabel: "Intensity",
});

plot.image("rgba", pixels, {
  rows: 512,
  cols: 512,
  channels: 4,
});
```

## PNG Export

```js
await plot.downloadPNG("nbimplot-signal.png");
const dataUrl = plot.toDataURL("image/png");
const blob = await plot.toBlob("image/png");
await plot.copy_png_to_clipboard();
```

Export redraws the strict WASM/ImPlot canvas immediately before reading pixels;
it does not use a JavaScript plotting fallback.

For a standalone shareable page:

```js
const html = plot.exportHTML({ title: "Signal Export" });
const htmlAlias = plot.export_html({ title: "Signal Export" });
console.log(html.length, htmlAlias.length);
```

The HTML export reloads `@nbimplot/web` and the packaged WASM assets.

## Themes And Specialty Plots

Theme presets are applied inside the C++/WASM layer:

```js
plot.setTheme("nbimplot");
plot.setTheme("publication");
plot.setTheme("finance");
plot.setTheme("lab");
plot.setTheme("dark-terminal");
```

Financial and scientific helpers route into the ImPlot/WASM primitive path:

```js
plot.candlestick("candles", open, high, low, close, { x, width: 0.7 });
plot.ohlc("ohlc", open, high, low, close, { x, width: 0.35 });
plot.quiver("field", x, y, u, v, { scale: 0.08, normalize: true });
plot.contour("contours", z, { rows, cols, levels, bounds: [[-3, -3], [3, 3]] });
plot.waterfall("waterfall", z, { rows, cols, scale: 0.18 });
plot.spectrogram("spectrogram", z, { rows, cols, labelFmt: "", showColorbar: true });
```

## Streaming, Selection, State, and Dashboards

```js
const h = plot.streamLine("ticks", {
  capacity: 200_000,
  initial,
  x: initialX,
  autoRender: true,
});
h.append(chunk, { x: chunkX });
h.pause();
h.resume();
h.setWindow(50_000);
h.clear();

plot.setTheme("nbimplot");
plot.setLinkedCrosshair("desk", { axis: "xy" });
plot.set_subplots_config({ rows: 2, cols: 2, linkAllX: true });

plot.onSelection((selection) => {
  const bounds = plot.selectionBounds(selection);
  const indices = plot.indicesForSelection(selection, h);
  plot.highlightSelection(selection, h, { name: "picked" });
  const csv = plot.exportCSVSelection(selection, h);
  const csvAlias = plot.export_csv_selection(selection, h);
  console.log(bounds, indices.length, csv, csvAlias);
});

const state = plot.getState({ includeData: true });
const json = plot.exportJSONState({ includeData: true });
const jsonAlias = plot.export_json_state({ includeData: true });
plot.setState(state);
console.log(json, jsonAlias);
```

```js
import { createDashboard } from "@nbimplot/web";

const dashboard = await createDashboard("#dashboard", {
  rows: 2,
  cols: 2,
  title: "Realtime Desk",
  theme: "nbimplot",
});
dashboard.line("cpu", cpu, { x: t, subplotIndex: 0 });
dashboard.line("latency", latency, { x: t, subplotIndex: 1 });
```

## Lifecycle

Always dispose plots when removing them from the DOM:

```js
plot.dispose();
```

This releases the WASM plot handle, cancels pending animation frames, removes
event listeners, and removes the canvas wrapper.

## Constraints

- WebGL2 is required.
- WASM assets must be served over HTTP(S).
- Very old browser runtimes and headless environments may fail to create the GL
  context.
- Histogram binning happens once in JS at upload time; line LOD and all drawing
  happen in WASM.
