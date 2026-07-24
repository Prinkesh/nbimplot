import DemoLoader from "./DemoLoader";

const stats = [
  ["Runtime", "Strict WASM"],
  ["Renderer", "ImPlot + WebGL2"],
  ["Transport", "Typed arrays"],
  ["LOD", "Min/max buckets"],
];

const pipeline = [
  "Notebook API",
  "Binary buffers",
  "WASM heap",
  "ImGui + ImPlot",
  "WebGL2 canvas",
];

const examples = [
  {
    id: "line-lod-plot",
    section: "Performance",
    title: "Million Point Line + Custom X + LOD",
    text: "Large time-series path using explicit x data and the default WASM min/max LOD pipeline. Pan, wheel zoom, double-click autoscale, and use the legend/right-click menu.",
    code: 'const h = plot.line("signal", y, { x });\nh.setData(yNew, { x });\nplot.autoscale();',
  },
  {
    id: "streaming-plot",
    section: "Performance",
    title: "Realtime Streaming Ring Buffer",
    text: "Appends small typed-array chunks into a fixed-capacity line without recreating the plot.",
    code: 'const h = plot.streamLine("ticks", { capacity: 12000 });\nh.append(chunk);',
  },
  {
    id: "scatter-plot",
    section: "Points",
    title: "Scatter + Bubble Encodings",
    text: "Explicit x/y buffers, marker rendering, and bubble sizes for dense point-cloud style workflows.",
    code: 'plot.scatter("samples", y, { x });\nplot.bubbles("volume", y, sizes, { x });',
  },
  {
    id: "curve-plot",
    section: "Curves",
    title: "Stairs, Stems, Digital, Shaded, Error Bars",
    text: "Common signal-analysis overlays in one canvas: stepped series, impulses, digital states, confidence bands, and uncertainty intervals.",
    code: 'plot.stairs("step", y, { x });\nplot.shaded("band", lower, upper, { x });\nplot.errorBars("fit", y, { x, err });',
  },
  {
    id: "bars-plot",
    section: "Categorical",
    title: "Bars, Grouped Bars, Horizontal Bars",
    text: "Three subplot panels showing vertical bars, grouped categorical bars, and horizontal rankings.",
    code: 'plot.setSubplots(1, 3);\nplot.bars("sales", values);\nplot.barGroups(labels, matrix);\nplot.barsH("rank", values);',
  },
  {
    id: "distribution-plot",
    section: "Statistics",
    title: "Histogram + 2D Histogram",
    text: "1D and 2D distributions, including a colorbar for density inspection.",
    code: 'plot.histogram("returns", values, { bins: 80 });\nplot.histogram2d("density", x, y, { xBins: 80, yBins: 60 });',
  },
  {
    id: "heatmap-image-plot",
    section: "Matrices",
    title: "Heatmap + Image",
    text: "Matrix plotting with empty heatmap labels, colorbar formatting, and a float RGB image buffer.",
    code: 'plot.setColormap("Viridis");\nplot.heatmap("z", matrix, { rows, cols, labelFmt: "" });\nplot.image("rgb", image, { rows, cols, channels: 3 });',
  },
  {
    id: "overlays-plot",
    section: "Overlays",
    title: "Annotations, Tags, Text, Infinite Lines, Pie",
    text: "ImPlot overlays for thresholds, labels, callouts, tags, and pie chart composition.",
    code: 'plot.vlines("events", xs);\nplot.tagY(0, { labelFmt: "zero" });\nplot.annotation("peak", x, y);\nplot.pieChart("mix", values, { labels });',
  },
  {
    id: "axes-plot",
    section: "Axes",
    title: "Axis Labels, Formats, Ticks, Log Scale, Secondary Axis",
    text: "Secondary y-axis, custom ticks, numeric formatting, and log scaling from the same plot object.",
    code: 'plot.setSecondaryAxes({ y2: true });\nplot.setAxisScale({ x: "linear", y: "log" });\nplot.setAxisTicks("x1", ticks, { labels });',
  },
  {
    id: "subplots-plot",
    section: "Layout",
    title: "Linked Subplots",
    text: "A 2x2 ImPlot subplot grid with shared x-axis interaction and mixed plot primitives.",
    code: 'plot.setSubplots(2, 2, { linkAllX: true });\nplot.line("a", y, { x, subplotIndex: 0 });\nplot.scatter("b", y, { x, subplotIndex: 1 });',
  },
  {
    id: "drag-plot",
    section: "Interaction",
    title: "Drag Lines, Drag Point, Drag Rect, Drag/Drop Targets",
    text: "Interactive ImPlot primitives. Drag the vertical/horizontal guides, point, and rectangle; inspect interaction values below.",
    code: 'plot.dragLineX("cursor", 40);\nplot.dragPoint("anchor", 25, 0.5);\nplot.onInteraction(events => ...);',
  },
  {
    id: "colormap-plot",
    section: "Colormaps",
    title: "Colormap Widgets + Runtime Switching",
    text: "Use the selector, slider, and buttons to verify that heatmaps and colorbar widgets use the active ImPlot colormap.",
    code: 'plot.setColormap("Plasma");\nplot.colormapSelector({ label: "Choose map" });\nplot.colormapSlider({ label: "Sample" });',
  },
];

export default function Page() {
  return (
    <main className="demo-shell">
      <nav className="topbar" aria-label="Project links">
        <a className="brand-mark" href="https://github.com/Prinkesh/nbimplot">
          <span className="brand-glyph">nb</span>
          <span>nbimplot</span>
        </a>
        <div className="topbar-links">
          <a href="https://pypi.org/project/nbimplot/">PyPI</a>
          <a href="https://www.npmjs.com/package/@nbimplot/web">npm</a>
          <a href="https://github.com/Prinkesh/nbimplot">GitHub</a>
        </div>
      </nav>

      <section className="hero-panel">
        <div className="hero-copy-block">
          <p className="eyebrow">ImPlot quality, notebook reach</p>
          <h1>Fast plotting for serious data in notebooks and web apps.</h1>
          <p className="hero-copy">
            nbimplot renders inside notebook output cells with the same strict WASM/ImPlot
            core shown here: typed-array uploads, screen-resolution LOD, native pan/zoom,
            hover inspection, context menus, subplots, colormaps, and streaming updates.
          </p>
          <div className="hero-actions">
            <a className="primary-link" href="#examples">Explore Examples</a>
            <a className="secondary-link" href="https://github.com/Prinkesh/nbimplot">View Source</a>
          </div>
        </div>
        <div className="hero-card" aria-label="Rendering pipeline">
          <div className="hero-card-header">
            <span>Pipeline</span>
            <strong>No JS renderer</strong>
          </div>
          <ol className="pipeline-list">
            {pipeline.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <div className="hero-card-footer">
            Interaction cost scales with canvas pixels, not raw point count.
          </div>
        </div>
      </section>

      <section className="control-panel" aria-label="Global demo controls">
        <div className="control-copy">
          <span>Live Controls</span>
          <strong>Drive the loaded WASM plots</strong>
        </div>
        <div className="toolbar">
          <button id="update-data" type="button">Update Data</button>
          <button id="toggle-stream" type="button">Start Stream</button>
          <button id="autoscale" type="button">Autoscale All</button>
          <label className="select-wrap">
            <span>Colormap</span>
            <select id="colormap-select" defaultValue="Viridis">
              <option value="Viridis">Viridis</option>
              <option value="Plasma">Plasma</option>
              <option value="Hot">Hot</option>
              <option value="Cool">Cool</option>
              <option value="Jet">Jet</option>
              <option value="Deep">Deep</option>
              <option value="Dark">Dark</option>
              <option value="Pastel">Pastel</option>
              <option value="Paired">Paired</option>
            </select>
          </label>
        </div>
      </section>

      <section className="metrics" aria-live="polite">
        {stats.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
        <div>
          <span>Mode</span>
          <strong id="mode">initializing</strong>
        </div>
        <div>
          <span>Examples</span>
          <strong>{examples.length} lazy plots</strong>
        </div>
        <div>
          <span>Last frame</span>
          <strong id="frame-ms">-- ms</strong>
        </div>
      </section>

      <section className="notes-panel">
        <strong>Interaction checklist:</strong> examples lazy-load as they approach the
        viewport and offscreen canvases are released to keep WebGL contexts bounded.
        Once loaded, left-drag pans, wheel zooms, scroll over axes zooms that axis,
        right-click opens ImPlot menus, right-drag box-select/box-zoom follows ImPlot
        behavior, and double-click autofits.
      </section>

      <section id="examples" className="examples-heading">
        <p className="eyebrow">Examples Gallery</p>
        <h2>Every supported plot primitive, grouped like release documentation.</h2>
        <p>
          These canvases are intentionally lazy-loaded. Scroll through the gallery to
          verify multiple independent WASM sessions, lifecycle cleanup, interaction
          primitives, and colormap propagation.
        </p>
      </section>

      <section className="examples-grid">
        {examples.map((example, index) => (
          <article className="example-card" key={example.id}>
            <div className="example-copy">
              <div className="example-kicker">
                <span>{String(index + 1).padStart(2, "0")}</span>
                <p className="section-label">{example.section}</p>
              </div>
              <h2>{example.title}</h2>
              <p>{example.text}</p>
              <pre><code>{example.code}</code></pre>
            </div>
            <div className="plot-frame">
              <div className="plot-frame-top">
                <span>{example.title}</span>
                <i aria-hidden="true" />
              </div>
              <div id={example.id} className="plot-host" />
            </div>
            {example.id === "drag-plot" ? (
              <p id="interaction-readout" className="readout">Interaction events: move a drag primitive.</p>
            ) : null}
          </article>
        ))}
      </section>

      <DemoLoader />
    </main>
  );
}
