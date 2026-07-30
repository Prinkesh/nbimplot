import DemoLoader from "./DemoLoader";

const stats = [
  ["Runtime", "Strict WASM"],
  ["Renderer", "ImPlot + WebGL2"],
  ["Transport", "Typed arrays"],
  ["LOD", "Min/max buckets"],
];

const proofPoints = [
  ["10M", "line-point workflow target"],
  ["O(px)", "pan and zoom work per frame"],
  ["0", "JSON point arrays on the hot path"],
];

const quickstart = [
  "pip install nbimplot",
  "",
  "import nbimplot as ip",
  "p = ip.Plot(width=900, height=450, title=\"Signal\")",
  "h = p.line(\"mid\", y, x=t)",
  "h.set_data(y_new, x=t_new)",
  "p.show()",
];

const webstart = [
  "npm install @nbimplot/web",
  "",
  "import { createPlot } from \"@nbimplot/web\";",
  "const plot = await createPlot(host, { title: \"Signal\" });",
  "const h = plot.line(\"mid\", y, { x: t });",
  "h.setData(yNext, { x: tNext });",
  "plot.render();",
];

const productPromises = [
  "Notebook output cell only",
  "Strict WASM + ImPlot rendering",
  "Binary array transport",
  "LOD inside the core",
];

const principles = [
  {
    title: "Binary data path",
    text: "NumPy and typed arrays move as buffers. The Python and browser APIs avoid JSON point lists for series data.",
  },
  {
    title: "ImPlot interaction model",
    text: "Pan, zoom, axis menus, legends, selection, and hover behavior come from the ImGui + ImPlot WASM runtime.",
  },
  {
    title: "Pixel-bounded rendering",
    text: "Large time series switch to min/max LOD so interaction cost tracks screen resolution, not raw array size.",
  },
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
    text: "Explicit x/y buffers, WASM min/max LOD, and callback-driven hover/click/selection inspection.",
    code: 'const h = plot.line("signal", y, { x });\nplot.onHover(console.log);\nplot.onSelection((e) => plot.indicesForSelection(e, h));',
  },
  {
    id: "streaming-plot",
    section: "Performance",
    title: "Realtime Streaming Ring Buffer",
    text: "Append explicit x/y chunks into a fixed-capacity line without recreating the plot object.",
    code: 'const h = plot.streamLine("ticks", { capacity: 12000, x: initialX });\nh.append(chunk, { x: chunkX });\nh.pause(); h.resume();',
  },
  {
    id: "scatter-plot",
    section: "Points",
    title: "Scatter + Bubble Encodings",
    text: "Point-cloud rendering with explicit x/y data and bubble-size encodings for dense browser workflows.",
    code: 'plot.scatter("samples", y, { x });\nplot.bubbles("volume", y, sizes, { x });',
  },
  {
    id: "curve-plot",
    section: "Curves",
    title: "Stairs, Stems, Digital, Shaded, Error Bars",
    text: "Signal-analysis overlays in one canvas: stepped series, impulses, bands, digital states, and uncertainty.",
    code: 'plot.stairs("step", y, { x });\nplot.shaded("band", lower, upper, { x });\nplot.errorBars("fit", y, { x, err });',
  },
  {
    id: "bars-plot",
    section: "Categorical",
    title: "Bars, Grouped Bars, Horizontal Bars",
    text: "Vertical bars, grouped categories, and horizontal rankings across ImPlot subplots.",
    code: 'plot.setSubplots(1, 3);\nplot.bars("sales", values);\nplot.barGroups(labels, matrix);\nplot.barsH("rank", values);',
  },
  {
    id: "distribution-plot",
    section: "Statistics",
    title: "Histogram + 2D Histogram",
    text: "1D and 2D distributions with colorbar-backed density inspection.",
    code: 'plot.histogram("returns", values, { bins: 80 });\nplot.histogram2d("density", x, y, { xBins: 80, yBins: 60 });',
  },
  {
    id: "heatmap-image-plot",
    section: "Matrices",
    title: "Heatmap + Image",
    text: "Matrix and image plotting with empty heatmap labels, colorbar formatting, and float RGB buffers.",
    code: 'plot.setColormap("Viridis");\nplot.heatmap("z", matrix, { rows, cols, labelFmt: "" });\nplot.image("rgb", image, { rows, cols, channels: 3 });',
  },
  {
    id: "overlays-plot",
    section: "Overlays",
    title: "Annotations, Tags, Text, Infinite Lines, Pie",
    text: "Thresholds, callouts, labels, tags, and pie-chart composition using ImPlot primitives.",
    code: 'plot.vlines("events", xs);\nplot.tagY(0, { labelFmt: "zero" });\nplot.annotation("peak", x, y);\nplot.pieChart("mix", values, { labels });',
  },
  {
    id: "axes-plot",
    section: "Axes",
    title: "Axis Labels, Formats, Ticks, Log Scale, Secondary Axis",
    text: "Secondary axes, custom ticks, numeric formats, linked axes, and log/time scale controls.",
    code: 'plot.setSecondaryAxes({ y2: true });\nplot.setAxisScale({ x: "linear", y: "log" });\nplot.setAxisTicks("x1", ticks, { labels });',
  },
  {
    id: "subplots-plot",
    section: "Layout",
    title: "Linked Subplots + Crosshair",
    text: "A 2x2 ImPlot subplot grid with linked x-axis behavior and crosshair synchronization.",
    code: 'plot.setSubplots(2, 2, { linkAllX: true });\nplot.setLinkedCrosshair("desk", { axis: "x" });\nplot.line("a", y, { x, subplotIndex: 0 });',
  },
  {
    id: "drag-plot",
    section: "Interaction",
    title: "Drag Lines, Drag Point, Drag Rect, Drag/Drop Targets",
    text: "Interactive ImPlot primitives for draggable guides, anchors, rectangles, and drop targets.",
    code: 'plot.dragLineX("cursor", 40);\nplot.dragPoint("anchor", 25, 0.5);\nplot.onInteraction(events => ...);',
  },
  {
    id: "colormap-plot",
    section: "Colormaps",
    title: "Colormap Widgets + Runtime Switching",
    text: "Selector, slider, and color button widgets that update heatmaps and colorbar primitives at runtime.",
    code: 'plot.setColormap("Plasma");\nplot.colormapSelector({ label: "Choose map" });\nplot.colormapSlider({ label: "Sample" });',
  },
  {
    id: "advanced-api-plot",
    section: "Advanced API",
    title: "State, Selection, Export",
    text: "View callbacks, state snapshots, PNG export, selection CSV, highlighting, constraints, links, and direct primitive access.",
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
          <a href="#why">Architecture</a>
          <a href="#command-center">Demo Controls</a>
          <a href="#examples">Examples</a>
          <a href="https://pypi.org/project/nbimplot/">PyPI</a>
          <a href="https://www.npmjs.com/package/@nbimplot/web">npm</a>
          <a className="topbar-cta" href="https://github.com/Prinkesh/nbimplot">GitHub</a>
        </div>
      </nav>

      <section className="hero-panel">
        <div className="hero-copy-block">
          <p className="eyebrow">Jupyter-native plotting, powered by ImPlot</p>
          <h1>WASM plots for serious notebook data.</h1>
          <p className="hero-copy">
            nbimplot gives notebooks and browser apps the same interaction model: binary
            arrays into a strict ImGui + ImPlot WASM core, rendered on a canvas with
            pixel-bounded LOD for large time series.
          </p>
          <div className="hero-actions">
            <a className="primary-link" href="#developer-api">Start Building</a>
            <a className="secondary-link" href="#examples">Explore Examples</a>
          </div>
          <div className="hero-promise-grid" aria-label="Product constraints">
            {productPromises.map((promise) => (
              <span key={promise}>{promise}</span>
            ))}
          </div>
        </div>

        <div className="hero-live-card" aria-label="Live nbimplot canvas">
          <div className="hero-live-header">
            <div>
              <span>Live WASM Surface</span>
              <strong>ImPlot canvas inside the page</strong>
            </div>
            <code>drag pan / wheel zoom / double-click fit</code>
          </div>
          <div className="hero-live-frame">
            <div id="hero-showcase-plot" className="plot-host hero-plot-host" />
          </div>
        </div>
      </section>

      <section id="developer-api" className="developer-section" aria-label="Notebook and web API">
        <div className="section-copy developer-copy">
          <p className="eyebrow">Minimal API, no global state</p>
          <h2>One plotting model for notebooks and web apps.</h2>
          <p>
            The public surface is intentionally small: create a plot, attach typed
            arrays, update handles in place, and let the WASM core manage interaction,
            state, LOD, subplots, and export.
          </p>
        </div>
        <div className="quickstart-grid">
          <div className="quickstart-card" aria-label="Notebook quickstart">
            <div className="quickstart-top">
              <span>Python</span>
              <strong>notebooks</strong>
            </div>
            <pre><code>{quickstart.join("\n")}</code></pre>
          </div>
          <div className="quickstart-card quickstart-card-alt" aria-label="Web quickstart">
            <div className="quickstart-top">
              <span>TypeScript</span>
              <strong>web apps</strong>
            </div>
            <pre><code>{webstart.join("\n")}</code></pre>
          </div>
        </div>
      </section>

      <section className="proof-strip" aria-label="Performance proof points">
        {proofPoints.map(([value, label]) => (
          <div key={label}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
        {stats.map(([label, value]) => (
          <div key={label}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
        <div>
          <strong id="mode">initializing</strong>
          <span>runtime status</span>
        </div>
        <div>
          <strong id="frame-ms">-- ms</strong>
          <span>last frame</span>
        </div>
      </section>

      <section id="why" className="principles-section">
        <div className="section-copy">
          <p className="eyebrow">Why this architecture</p>
          <h2>Designed around the expensive parts of plotting.</h2>
          <p>
            The Python layer validates and transfers buffers. The browser view owns the
            canvas lifecycle. The WASM core owns plot state, LOD, and ImPlot rendering.
          </p>
        </div>
        <div className="principle-grid">
          {principles.map((item, index) => (
            <article key={item.title} className="principle-card">
              <span>{String(index + 1).padStart(2, "0")}</span>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="command-center" className="command-center" aria-label="Interactive API command center">
        <div className="section-copy command-copy">
          <p className="eyebrow">Live API surface</p>
          <h2>Operate the examples from one control panel.</h2>
          <p>
            These buttons load the target canvas, scroll it into view, and call the same
            public APIs available in notebooks and web apps.
          </p>
          <p id="feature-status" className="feature-status">
            Pick an action. The app will load the matching example and report the result here.
          </p>
        </div>

        <div className="command-stack">
          <section className="control-panel" aria-label="Global demo controls">
            <div className="control-copy">
              <span>Global controls</span>
              <strong>Loaded plots</strong>
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
          <p className="eyebrow">Interaction checklist</p>
          <p>
            Examples lazy-load near the viewport and offscreen canvases are released to
            keep WebGL contexts bounded. Once loaded, left-drag pans, wheel zooms,
            right-click opens ImPlot menus, right-drag box-select/box-zoom follows
            ImPlot behavior, and double-click autofits.
          </p>
        </section>

        <section className="resource-panel" aria-label="Documentation and AI-readable resources">
          <div>
            <p className="eyebrow">Documentation</p>
            <h2>Guides for users, agents, and web integrations.</h2>
            <p>
              Direct resources for fast notebook plotting, million-point visualization,
              web app integration, and LLM/search positioning.
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
        <div className="section-copy">
          <p className="eyebrow">Examples gallery</p>
          <h2>Coverage across ImPlot-style primitives.</h2>
          <p>
            Each canvas is an independent WASM session. Scroll to lazy-load examples and
            verify lifecycle cleanup, interactions, subplots, colormaps, and exports.
          </p>
        </div>
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
