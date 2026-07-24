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
