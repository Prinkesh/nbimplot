export default function Page() {
  return (
    <main className="demo-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">ImPlot + WASM + WebGL2</p>
          <h1>nbimplot web demo</h1>
        </div>
        <div className="toolbar">
          <button id="update-data" type="button">Update</button>
          <button id="toggle-stream" type="button">Stream</button>
          <button id="autoscale" type="button">Autoscale</button>
        </div>
      </section>

      <section className="metrics" aria-live="polite">
        <div>
          <span>Package</span>
          <strong>@nbimplot/web</strong>
        </div>
        <div>
          <span>Mode</span>
          <strong id="mode">initializing</strong>
        </div>
        <div>
          <span>Line points</span>
          <strong>1,000,000</strong>
        </div>
        <div>
          <span>Last frame</span>
          <strong id="frame-ms">-- ms</strong>
        </div>
      </section>

      <section className="plot-grid">
        <div className="plot-panel">
          <div id="line-plot" className="plot-host" />
        </div>
        <div className="plot-panel">
          <div id="heatmap-plot" className="plot-host" />
        </div>
      </section>

      <script type="module" src="/demo.js" />
    </main>
  );
}
