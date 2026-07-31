# Using nbimplot Directly In Web Apps

`@nbimplot/web` is the standalone browser package for using the same strict WASM + ImGui + ImPlot core outside Jupyter.

## Install

```bash
npm install @nbimplot/web
```

## Plain JavaScript

```html
<div id="plot" style="height: 450px"></div>
<script type="module">
  import { createPlot } from "@nbimplot/web";

  const plot = await createPlot("#plot", {
    width: 900,
    height: 450,
    responsive: true,
    title: "Signal",
  });

  const x = new Float32Array(1_000_000);
  const y = new Float32Array(x.length);
  for (let i = 0; i < y.length; i += 1) {
    x[i] = i * 0.001;
    y[i] = Math.sin(x[i]);
  }

  const h = plot.line("signal", y, { x, color: "#2563eb", lineWeight: 2 });
  plot.render();

  plot.onHover((event) => {
    console.log("hover", event.seriesName, event.index, event.x, event.y);
  });

  plot.onSelection((event) => {
    const exact = plot.indicesForSelection(event);
    console.log([...exact.entries()].map(([token, indices]) => [token, indices.length]));
  });

  window.addEventListener("beforeunload", () => plot.dispose(), { once: true });
</script>
```

To export the active canvas:

```js
await plot.downloadPNG("signal.png");
const blob = await plot.toBlob("image/png");
const dataUrl = plot.toDataURL("image/png");
await plot.copy_png_to_clipboard();
```

## Advanced Web API

```js
plot.setTheme("nbimplot");
plot.setLinkedCrosshair("shared", { axis: "x" });
plot.set_subplots_config({ rows: 2, cols: 2, linkAllX: true });

const h = plot.streamLine("ticks", { capacity: 100_000, initial, x: initialX });
h.append(chunk, { x: chunkX });
h.pause();
h.resume();
h.setWindow(25_000);
h.setStreamOptions({ autoRender: true });
h.clear();

plot.onSelection((selection) => {
  const bounds = plot.selectionBounds(selection);
  plot.highlightSelection(selection, h);
  const csv = plot.exportCSVSelection(selection, h);
  const csvAlias = plot.export_csv_selection(selection, h);
  console.log(bounds, csv, csvAlias);
});

const state = plot.getState({ includeData: true });
plot.setState(state);
const json = plot.exportJSONState({ includeData: true });
const jsonAlias = plot.export_json_state({ includeData: true });
console.log(json, jsonAlias);
```

Batch lines, theme presets, standalone HTML state export, and specialty
financial/scientific primitives are available in the same direct web API:

```js
plot.lines({
  bid: { x, y: bid, color: "#1f6f66" },
  ask: { x, y: ask, color: "#b74b2b" },
});

plot.line("datetime", latency, { x: dateArray });
plot.scatter("category", scores, { x: ["A", "B", "C"] });
plot.setTheme("finance");
plot.candlestick("candles", open, high, low, close, { x, width: 0.7 });
plot.ohlc("ohlc", open, high, low, close, { x, width: 0.35 });
plot.quiver("field", xField, yField, uField, vField, { scale: 0.08, normalize: true });
plot.contour("contour", matrix, { rows, cols, levels });
plot.waterfall("waterfall", matrix, { rows, cols, scale: 0.18 });
plot.spectrogram("spectrogram", matrix, { rows, cols, labelFmt: "", showColorbar: true });

const html = plot.exportHTML({ title: "Desk Snapshot" });
const htmlAlias = plot.export_html({ title: "Desk Snapshot" });
console.log(html.length, htmlAlias.length);
```

## React Pattern

```jsx
import { useEffect, useRef } from "react";
import { createPlot } from "@nbimplot/web";

export function SignalPlot({ x, y }) {
  const hostRef = useRef(null);

  useEffect(() => {
    let disposed = false;
    let plot;
    let handle;

    async function mount() {
      plot = await createPlot(hostRef.current, { responsive: true, title: "Signal" });
      if (disposed) {
        plot.dispose();
        return;
      }
      handle = plot.line("signal", y, { x });
      plot.render();
    }

    mount();
    return () => {
      disposed = true;
      plot?.dispose();
    };
  }, []);

  return <div ref={hostRef} style={{ height: 450 }} />;
}
```

For frequent updates, keep the plot and handle in refs and call `handle.setData(yNew, { x: xNew })` instead of recreating the plot.

## Asset Loading

By default, the package loads its colocated WASM asset. You can override asset loading:

```js
await createPlot("#plot", {
  wasmUrl: "/assets/nbimplot_wasm.wasm",
});
```

or pass a prefetched binary:

```js
const wasmBinary = new Uint8Array(await (await fetch("/nbimplot_wasm.wasm")).arrayBuffer());
await createPlot("#plot", { wasmBinary });
```

## Lifecycle Rules

- Call `plot.dispose()` when removing the canvas from the DOM.
- Do not create unbounded WebGL contexts in long scrolling pages; lazy-load and release offscreen plots.
- Use `responsive: true` when the host container can resize.
- Serve WASM assets over HTTP(S), not `file://`.

## Browser Requirement

`@nbimplot/web` requires WebGL2. If the browser or environment cannot create a WebGL2 context, use another plotting path or run in a local browser session with GPU acceleration enabled.
