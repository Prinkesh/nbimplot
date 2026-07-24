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

  window.addEventListener("beforeunload", () => plot.dispose(), { once: true });
</script>
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
