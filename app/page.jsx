import DemoLoader from "./DemoLoader";

const stats = [
  ["Runtime", "Strict WASM"],
  ["Renderer", "ImPlot + WebGL2"],
  ["Transport", "Typed arrays"],
  ["LOD", "Min/max buckets"],
];

const heroBadges = ["WASM only", "ImPlot native", "Jupyter + Web"];

const heroProofs = [
  ["10M", "line points target"],
  ["O(px)", "interaction cost"],
  ["0", "JS render fallback"],
];

const resourceLinks = [
  ["LLM Summary", `${process.env.NEXT_PUBLIC_BASE_PATH || ""}/llms.txt`],
  ["Full LLM Docs", `${process.env.NEXT_PUBLIC_BASE_PATH || ""}/llms-full.txt`],
  ["Fast Jupyter Guide", "https://github.com/Prinkesh/nbimplot/blob/main/docs/FAST_JUPYTER_PLOTTING.md"],
  ["Million-Point Guide", "https://github.com/Prinkesh/nbimplot/blob/main/docs/MILLION_POINT_NOTEBOOK_PLOTTING.md"],
  ["Web App Guide", "https://github.com/Prinkesh/nbimplot/blob/main/docs/WEBAPP_INTEGRATION.md"],
  ["Positioning", "https://github.com/Prinkesh/nbimplot/blob/main/docs/POSITIONING.md"],
];

const examples = [
  {
    id: "line-lod-plot",
    section: "Performance",
    title: "Million Point Line + Custom X + LOD",
    text: "Large time-series path using explicit x data, WASM min/max LOD, and callback-driven hover/click/selection inspection.",
    code: 'const h = plot.line("signal", y, { x });\nplot.onHover(console.log);\nplot.onSelection((e) => plot.indicesForSelection(e, h));',
  },
  {
    id: "streaming-plot",
    section: "Performance",
    title: "Realtime Streaming Ring Buffer",
    text: "Appends explicit x/y typed-array chunks into a fixed-capacity line with pause/resume controls.",
    code: 'const h = plot.streamLine("ticks", { capacity: 12000, x: initialX });\nh.append(chunk, { x: chunkX });\nh.pause(); h.resume();',
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
    title: "Linked Subplots + Crosshair",
    text: "A 2x2 ImPlot subplot grid with shared x-axis interaction, linked crosshair, and mixed plot primitives.",
    code: 'plot.setSubplots(2, 2, { linkAllX: true });\nplot.setLinkedCrosshair("desk", { axis: "x" });\nplot.line("a", y, { x, subplotIndex: 0 });',
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
  {
    id: "advanced-api-plot",
    section: "Advanced API",
    title: "State, Selection, Export",
    text: "A focused API coverage example for view/perf callbacks, state snapshots, PNG export, selection CSV, highlighting, constraints, links, and direct primitive access.",
    code: 'plot.setTheme("nbimplot");\nconst state = plot.getState({ includeData: true });\nplot.highlightSelection(selection, h);\nconst csv = plot.exportCSVSelection(selection, h);\nawait plot.downloadPNG("advanced.png");',
  },
];

const exampleSections = Array.from(new Set(examples.map((example) => example.section)));

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
          <a className="topbar-cta" href="#command-center">Try APIs</a>
        </div>
      </nav>

      <section className="hero-panel">
        <div className="hero-copy-block">
          <div className="hero-badge-row" aria-label="Project qualities">
            {heroBadges.map((badge) => (
              <span key={badge}>{badge}</span>
            ))}
          </div>
          <p className="eyebrow">ImPlot quality, notebook reach</p>
          <h1>Fast plotting for <span>million-point</span> workflows.</h1>
          <p className="hero-copy">
            nbimplot brings an ImGui + ImPlot WASM core to notebooks and browser apps:
            binary typed-array uploads, screen-resolution LOD, native pan/zoom, context
            menus, subplots, colormaps, streaming updates, and exact selection APIs.
          </p>
          <div className="hero-actions">
            <a className="primary-link" href="#examples">Explore Examples</a>
            <a className="secondary-link" href="https://github.com/Prinkesh/nbimplot">View Source</a>
          </div>
          <div className="hero-proof-grid" aria-label="Performance proof points">
            {heroProofs.map(([value, label]) => (
              <div key={label}>
                <strong>{value}</strong>
                <span>{label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="hero-live-card" aria-label="Live nbimplot canvas">
          <div className="hero-live-header">
            <div>
              <span>Live Surface</span>
              <strong>ImPlot WASM canvas</strong>
            </div>
            <code>drag pan / wheel zoom / double-click fit</code>
          </div>
          <div className="hero-live-frame">
            <div id="hero-showcase-plot" className="plot-host hero-plot-host" />
          </div>
          <div className="hero-live-footer">
            <span>Binary data path</span>
            <span>LOD by default</span>
            <span>No native window</span>
          </div>
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
          <span>Gallery</span>
          <strong>{examples.length} lazy plots</strong>
        </div>
        <div>
          <span>Last frame</span>
          <strong id="frame-ms">-- ms</strong>
        </div>
      </section>

      <section id="command-center" className="command-center" aria-label="Interactive API command center">
        <div className="command-copy">
          <p className="eyebrow">Command Center</p>
          <h2>Drive real WASM plots from the page.</h2>
          <p>
            These controls load the target canvas, scroll it into view, and call the same
            public APIs available from notebook widgets and web apps.
          </p>
          <p id="feature-status" className="feature-status">
            Pick an action. The app will load the matching example and report the result here.
          </p>
        </div>

        <div className="command-stack">
          <section className="control-panel" aria-label="Global demo controls">
            <div className="control-copy">
              <span>Live Controls</span>
              <strong>Operate loaded plots</strong>
            </div>
            <div className="toolbar">
              <button id="update-data" type="button" data-label="data">Update Data</button>
              <button id="toggle-stream" type="button" data-label="stream">Start Stream</button>
              <button id="autoscale" type="button" data-label="view">Autoscale All</button>
              <button id="export-png" type="button" data-label="export">Export PNG</button>
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

          <section className="feature-workbench" aria-label="Feature workbench controls">
            <div className="feature-actions">
              <button id="stream-pause" type="button" data-label="stream">Pause Stream</button>
              <button id="stream-clear" type="button" data-label="buffer">Clear Stream</button>
              <button id="stream-window" type="button" data-label="window">Set 3k Window</button>
              <button id="run-selection-demo" type="button" data-label="select">Highlight Selection</button>
              <button id="export-state" type="button" data-label="state">Download State JSON</button>
              <button id="restore-state" type="button" data-label="state">Restore State</button>
              <button id="copy-png" type="button" data-label="image">Copy PNG</button>
              <button id="toggle-crosshair" type="button" data-label="link">Disable Crosshair Link</button>
            </div>
          </section>
        </div>
      </section>

      <section className="info-grid" aria-label="Usage notes and documentation">
        <section className="notes-panel">
          <strong>Interaction checklist:</strong> examples lazy-load as they approach the
          viewport and offscreen canvases are released to keep WebGL contexts bounded.
          Once loaded, left-drag pans, wheel zooms, scroll over axes zooms that axis,
          right-click opens ImPlot menus, right-drag box-select/box-zoom follows ImPlot
          behavior, and double-click autofits.
        </section>

        <section className="resource-panel" aria-label="Documentation and AI-readable resources">
          <div>
            <p className="eyebrow">Documentation</p>
            <h2>Human-readable and agent-readable guides.</h2>
            <p>
              These links make the project easier to classify for users, search engines,
              coding assistants, and retrieval-augmented agents looking for fast notebook
              plotting, ImPlot in Jupyter, WASM plotting, and large-data visualization.
            </p>
          </div>
          <div className="resource-links">
            {resourceLinks.map(([label, href]) => (
              <a key={label} href={href}>{label}</a>
            ))}
          </div>
        </section>
      </section>

      <section id="examples" className="examples-heading">
        <p className="eyebrow">Examples Gallery</p>
        <h2>Every supported plot primitive, grouped like release documentation.</h2>
        <p>
          Scroll through lazy-loaded canvases to verify independent WASM sessions,
          lifecycle cleanup, interaction primitives, and colormap propagation.
        </p>
        <div className="example-tabs" aria-label="Example categories">
          {exampleSections.map((section) => (
            <span key={section}>{section}</span>
          ))}
        </div>
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
                <strong>WASM</strong>
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
