import { createPlot, probeWebGL2 } from "./vendor/nbimplot/src/index.js";

const previousDemo = window.__nbimplotExamplesDemo;
if (previousDemo?.dispose) previousDemo.dispose();

const MAX_ACTIVE_PLOTS = 3;

const state = {
  plots: [],
  plotById: new Map(),
  timers: [],
  controllers: [],
  loadPromises: new Map(),
  lazyChain: Promise.resolve(),
  loadedIds: new Set(),
  visitedIds: new Set(),
  visibleIds: new Set(),
  loadingIds: new Set(),
  builders: new Map(),
  colormapPlots: [],
  lineSeries: null,
  lineData: null,
  linePhase: 0,
  streamHandle: null,
  streamPlot: null,
  streamSample: 0,
  streaming: false,
  observer: null,
  totalExamples: 0,
  activeColormap: "Viridis",
  disposed: false,
  dispose() {
    this.disposed = true;
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
    for (const timer of this.timers) window.clearInterval(timer);
    this.timers = [];
    for (const controller of this.controllers) controller.abort();
    this.controllers = [];
    for (const plot of this.plots) {
      try {
        plot.dispose();
      } catch (error) {
        console.warn("Failed to dispose nbimplot example", error);
      }
    }
    this.plots = [];
    this.plotById.clear();
    this.loadPromises.clear();
    this.lazyChain = Promise.resolve();
    this.loadedIds.clear();
    this.visitedIds.clear();
    this.visibleIds.clear();
    this.loadingIds.clear();
    this.builders.clear();
    this.colormapPlots = [];
  },
};
window.__nbimplotExamplesDemo = state;

const ids = [
  "line-lod-plot",
  "streaming-plot",
  "scatter-plot",
  "curve-plot",
  "bars-plot",
  "distribution-plot",
  "heatmap-image-plot",
  "overlays-plot",
  "axes-plot",
  "subplots-plot",
  "drag-plot",
  "colormap-plot",
];

const mode = document.querySelector("#mode");
const frameMs = document.querySelector("#frame-ms");
const updateButton = document.querySelector("#update-data");
const streamButton = document.querySelector("#toggle-stream");
const autoscaleButton = document.querySelector("#autoscale");
const colormapSelect = document.querySelector("#colormap-select");
const interactionReadout = document.querySelector("#interaction-readout");

function setMode(text) {
  if (mode) mode.textContent = text;
}

function setHostStatus(id, text) {
  const host = document.querySelector(`#${id}`);
  if (!host) return;
  if (host.children.length > 0 && !host.firstElementChild?.classList.contains("plot-placeholder")) return;
  host.replaceChildren();
  const panel = document.createElement("div");
  panel.className = "plot-placeholder";
  panel.textContent = text;
  host.appendChild(panel);
}

function setHostError(id, error) {
  const host = document.querySelector(`#${id}`);
  if (!host) return;
  host.replaceChildren();
  const panel = document.createElement("div");
  panel.className = "plot-error";
  panel.textContent = error instanceof Error ? error.message : String(error);
  host.appendChild(panel);
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let v = t;
    v = Math.imul(v ^ (v >>> 15), v | 1);
    v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

function normalFactory(seed) {
  const random = mulberry32(seed);
  let spare = 0;
  let hasSpare = false;
  return () => {
    if (hasSpare) {
      hasSpare = false;
      return spare;
    }
    const u = Math.max(1e-7, random());
    const v = random();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    hasSpare = true;
    return mag * Math.cos(2 * Math.PI * v);
  };
}

function range(n, scale = 1, offset = 0) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = offset + i * scale;
  return out;
}

function makeSignal(target, phase = 0) {
  for (let i = 0; i < target.length; i += 1) {
    const x = i * 0.001;
    const spike = i % 131071 === 0 ? 1.8 : 0;
    target[i] = Math.sin(x + phase) + 0.18 * Math.sin(i * 0.017) + 0.08 * Math.cos(i * 0.00031) + spike;
  }
  return target;
}

function makeMatrix(rows, cols, phase = 0) {
  const z = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      z[r * cols + c] =
        Math.sin(r * 0.14 + phase) * Math.cos(c * 0.075) +
        0.35 * Math.sin((r + c) * 0.045) +
        0.18 * Math.cos(Math.hypot(r - rows / 2, c - cols / 2) * 0.16);
    }
  }
  return z;
}

function makeImage(rows, cols) {
  const image = new Float32Array(rows * cols * 3);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i = (r * cols + c) * 3;
      const nx = c / Math.max(1, cols - 1);
      const ny = r / Math.max(1, rows - 1);
      image[i] = nx;
      image[i + 1] = 0.35 + 0.65 * ny;
      image[i + 2] = 0.55 + 0.45 * Math.sin((nx + ny) * Math.PI);
    }
  }
  return image;
}

async function mountPlot(id, options = {}) {
  if (state.disposed) throw new Error("nbimplot demo was disposed before plot creation.");
  const host = document.querySelector(`#${id}`);
  if (!host) throw new Error(`Missing plot host: ${id}`);
  host.replaceChildren();
  const plot = await createPlot(host, {
    responsive: true,
    crosshairs: true,
    ...options,
  });
  state.plots.push(plot);
  state.plotById.set(id, plot);
  return plot;
}

async function runExample(id, create) {
  try {
    const plot = await create();
    return plot;
  } catch (error) {
    console.error(`nbimplot example failed: ${id}`, error);
    setHostError(id, error);
    return null;
  }
}

function on(element, type, listener) {
  if (!element) return;
  const controller = new AbortController();
  element.addEventListener(type, listener, { signal: controller.signal });
  state.controllers.push(controller);
}

function updateLoadMode() {
  const active = state.plotById.size;
  const visited = state.visitedIds.size;
  const total = state.totalExamples || ids.length;
  if (visited === total) {
    setMode(`${active} active | ${visited}/${total} examples visited`);
  } else {
    setMode(`${active} active | ${visited}/${total} visited - scroll for more`);
  }
}

function resetHandlesForReleasedPlot(id) {
  if (id === "line-lod-plot") {
    state.lineSeries = null;
    state.lineData = null;
  }
  if (id === "streaming-plot") {
    if (state.streaming) {
      for (const timer of state.timers) window.clearInterval(timer);
      state.timers = [];
      state.streaming = false;
      if (streamButton) streamButton.textContent = "Start Stream";
    }
    state.streamHandle = null;
    state.streamPlot = null;
  }
  if (id === "drag-plot" && interactionReadout) {
    interactionReadout.textContent = "Interaction events: move a drag primitive.";
  }
}

function releaseExample(id, message = "Released offscreen to keep WebGL contexts low. Scroll near this card to reload.") {
  const plot = state.plotById.get(id);
  if (!plot) return;

  try {
    plot.dispose();
  } catch (error) {
    // Some browser/headless WebGL stacks report noisy teardown errors after
    // context loss. The gallery still removes the DOM wrapper and drops refs.
    void error;
  } finally {
    plot.wrapper?.remove?.();
  }

  state.plotById.delete(id);
  state.plots = state.plots.filter((candidate) => candidate !== plot);
  state.colormapPlots = state.colormapPlots.filter((candidate) => candidate !== plot);
  state.loadedIds.delete(id);
  state.loadPromises.delete(id);
  resetHandlesForReleasedPlot(id);
  setHostStatus(id, message);
  updateLoadMode();
}

function enforceActiveBudget() {
  if (state.plotById.size <= MAX_ACTIVE_PLOTS) return;
  for (const id of state.plotById.keys()) {
    if (state.visibleIds.has(id)) continue;
    releaseExample(id);
    if (state.plotById.size <= MAX_ACTIVE_PLOTS) break;
  }
}

function loadExample(id) {
  if (state.plotById.has(id)) return Promise.resolve(true);
  if (state.loadPromises.has(id)) return state.loadPromises.get(id);
  const builder = state.builders.get(id);
  if (!builder) return Promise.resolve(false);

  state.loadingIds.add(id);
  const host = document.querySelector(`#${id}`);
  if (host) {
    host.replaceChildren();
    const panel = document.createElement("div");
    panel.className = "plot-placeholder";
    panel.textContent = "Loading WASM plot...";
    host.appendChild(panel);
  }

  const promise = state.lazyChain
    .then(() => (state.disposed || !state.visibleIds.has(id) ? null : runExample(id, builder)))
    .then((plot) => {
      state.loadingIds.delete(id);
      state.loadPromises.delete(id);
      if (plot) {
        state.loadedIds.add(id);
        state.visitedIds.add(id);
        if (state.visibleIds.has(id)) {
          enforceActiveBudget();
        } else {
          releaseExample(id);
        }
      } else if (!state.visibleIds.has(id)) {
        setHostStatus(id, "Scroll near this card to load the WASM plot.");
      }
      updateLoadMode();
      return Boolean(plot);
    })
    .catch((error) => {
      state.loadingIds.delete(id);
      state.loadPromises.delete(id);
      console.error(`nbimplot lazy load failed: ${id}`, error);
      setHostError(id, error);
      updateLoadMode();
      return false;
    });

  state.loadPromises.set(id, promise);
  state.lazyChain = promise.catch(() => false);
  return promise;
}

function setupLazyLoading() {
  for (const id of ids) {
    setHostStatus(id, "Scroll near this card to load the WASM plot.");
  }

  if (!("IntersectionObserver" in window)) {
    state.visibleIds.add("line-lod-plot");
    state.visibleIds.add("streaming-plot");
    loadExample("line-lod-plot");
    loadExample("streaming-plot");
    setMode("lazy loading unavailable - top examples loaded");
    return;
  }

  state.observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const id = entry.target.id;
      if (entry.isIntersecting) {
        state.visibleIds.add(id);
        loadExample(id);
      } else {
        state.visibleIds.delete(id);
        releaseExample(id);
      }
    }
  }, {
    root: null,
    rootMargin: "180px 0px 260px 0px",
    threshold: 0.01,
  });

  for (const id of ids) {
    const host = document.querySelector(`#${id}`);
    if (host) state.observer.observe(host);
  }
}

async function buildLineLod() {
  const n = 1_000_000;
  const y = makeSignal(new Float32Array(n));
  const plot = await mountPlot("line-lod-plot", {
    title: "Million Point Line - WASM LOD",
  });
  plot.setAxisLabel("x1", "sample");
  plot.setAxisLabel("y1", "value");
  plot.setAxisFormat("y1", "%.2f");
  const handle = plot.line("signal", y, { color: "#1f6f66", lineWeight: 2 });
  plot.hlines("baseline", new Float32Array([0]));
  plot.vlines("spike markers", new Float32Array([131071, 262142, 524284, 786426]));
  plot.tagY(0, { labelFmt: "zero", roundValue: false });
  plot.onPerfStats((stats) => {
    if (!frameMs) return;
    frameMs.textContent = `${stats.frameMs.toFixed(2)} ms | ${Math.round(stats.drawPoints).toLocaleString()} drawn`;
  });
  state.lineSeries = handle;
  state.lineData = y;
  return plot;
}

async function buildStreaming() {
  const plot = await mountPlot("streaming-plot", {
    title: "Realtime Streaming Append",
    autoFitOnDataChange: true,
  });
  plot.setAxisLabel("x1", "sample");
  plot.setAxisLabel("y1", "tick");
  const initial = new Float32Array(256);
  for (let i = 0; i < initial.length; i += 1) {
    initial[i] = Math.sin(i * 0.06) + 0.15 * Math.sin(i * 0.31);
  }
  state.streamSample = initial.length;
  state.streamHandle = plot.streamLine("ticks", {
    capacity: 12_000,
    initial,
    color: "#b74b2b",
    lineWeight: 2,
  });
  state.streamPlot = plot;
  return plot;
}

function appendStreamChunk() {
  if (!state.streamHandle) return;
  const chunk = new Float32Array(96);
  for (let i = 0; i < chunk.length; i += 1) {
    const t = state.streamSample + i;
    chunk[i] = Math.sin(t * 0.035) + 0.24 * Math.sin(t * 0.19) + 0.08 * Math.cos(t * 0.006);
  }
  state.streamSample += chunk.length;
  state.streamHandle.append(chunk);
}

async function buildScatter() {
  const normal = normalFactory(7);
  const n = 18_000;
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const cluster = i % 3;
    x[i] = normal() * (0.45 + cluster * 0.12) + cluster * 2.6;
    y[i] = normal() * (0.35 + cluster * 0.08) + Math.sin(cluster * 1.7) * 1.6;
  }
  const bubbleCount = 700;
  const bx = new Float32Array(bubbleCount);
  const by = new Float32Array(bubbleCount);
  const sizes = new Float32Array(bubbleCount);
  for (let i = 0; i < bubbleCount; i += 1) {
    bx[i] = 1.2 + 4.2 * (i / bubbleCount) + 0.35 * normal();
    by[i] = 1.2 * Math.sin(i * 0.045) + 0.28 * normal();
    sizes[i] = 2 + 8 * Math.abs(Math.sin(i * 0.08));
  }
  const plot = await mountPlot("scatter-plot", { title: "Scatter + Bubbles" });
  plot.setAxisLabel("x1", "factor A");
  plot.setAxisLabel("y1", "factor B");
  plot.scatter("clusters", y, { x, marker: "circle" });
  plot.bubbles("weighted samples", by, sizes, { x: bx });
  return plot;
}

async function buildCurveVariants() {
  const n = 260;
  const x = range(n, 0.08);
  const smooth = new Float32Array(n);
  const lower = new Float32Array(n);
  const upper = new Float32Array(n);
  const steps = new Float32Array(n);
  const stems = new Float32Array(n);
  const digital = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    smooth[i] = 0.55 * Math.sin(i * 0.08) + 0.2 * Math.sin(i * 0.21);
    lower[i] = smooth[i] - 0.18 - 0.06 * Math.sin(i * 0.09);
    upper[i] = smooth[i] + 0.18 + 0.06 * Math.cos(i * 0.05);
    steps[i] = Math.floor((Math.sin(i * 0.05) + 1) * 2) / 2 - 0.7;
    stems[i] = i % 19 === 0 ? 1.15 : 0.05 * Math.sin(i * 0.4);
    digital[i] = (Math.sin(i * 0.11) > 0.25 ? 1 : 0) - 1.6;
  }
  const sampleCount = 20;
  const sx = new Float32Array(sampleCount);
  const sy = new Float32Array(sampleCount);
  const err = new Float32Array(sampleCount);
  const xerr = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i += 1) {
    const idx = Math.min(n - 1, i * 12 + 8);
    sx[i] = x[idx];
    sy[i] = smooth[idx];
    err[i] = 0.08 + 0.03 * (i % 4);
    xerr[i] = 0.04 + 0.015 * (i % 3);
  }
  const plot = await mountPlot("curve-plot", { title: "Curve Variants + Uncertainty" });
  plot.setAxisLabel("x1", "time");
  plot.setAxisLabel("y1", "state");
  plot.shaded("confidence band", lower, upper, { x, alpha: 0.22 });
  plot.stairs("stairs", steps, { x });
  plot.stems("impulses", stems, { x });
  plot.digital("digital state", digital, { x });
  plot.scatter("sample points", sy, { x: sx, marker: "circle" });
  plot.errorBars("vertical error", sy, { x: sx, err });
  plot.errorBarsH("horizontal error", sx, { y: sy, err: xerr });
  return plot;
}

async function buildBars() {
  const plot = await mountPlot("bars-plot", {
    title: "Bars + Groups + Horizontal Bars",
    crosshairs: false,
  });
  plot.setSubplots(1, 3, { noResize: false });
  plot.bars("quarterly revenue", new Float32Array([9, 12, 15, 13, 17, 21]), { subplotIndex: 0, barWidth: 0.72 });
  plot.barGroups(["CPU", "GPU", "Memory"], new Float32Array([
    12, 17, 19, 24,
    8, 13, 18, 22,
    10, 12, 14, 18,
  ]), {
    itemCount: 3,
    groupCount: 4,
    groupSize: 0.78,
    subplotIndex: 1,
  });
  plot.barsH("latency budget", new Float32Array([23, 31, 18, 12, 28]), {
    y: new Float32Array([0, 1, 2, 3, 4]),
    barHeight: 0.55,
    subplotIndex: 2,
  });
  return plot;
}

async function buildDistributions() {
  const normal = normalFactory(11);
  const n = 28_000;
  const values = new Float32Array(n);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const regime = i % 5 === 0 ? 1.8 : 0;
    values[i] = 0.45 * normal() + regime;
    x[i] = 0.9 * normal() + 0.6 * Math.sin(i * 0.003);
    y[i] = 0.55 * x[i] + 0.85 * normal();
  }
  const plot = await mountPlot("distribution-plot", { title: "Histogram + Density Heatmap" });
  plot.setSubplots(1, 2, { noResize: false });
  plot.setColormap(state.activeColormap);
  plot.histogram("returns", values, { bins: 80, subplotIndex: 0 });
  plot.histogram2d("joint density", x, y, {
    xBins: 80,
    yBins: 64,
    labelFmt: "",
    showColorbar: true,
    colorbarLabel: "count",
    colorbarFormat: "%.0f",
    subplotIndex: 1,
  });
  state.colormapPlots.push(plot);
  return plot;
}

async function buildHeatmapImage() {
  const plot = await mountPlot("heatmap-image-plot", { title: "Heatmap + RGB Image" });
  plot.setSubplots(1, 2, { noResize: false });
  plot.setColormap(state.activeColormap);
  plot.heatmap("sensor grid", makeMatrix(96, 144), {
    rows: 96,
    cols: 144,
    labelFmt: "",
    showColorbar: true,
    colorbarLabel: "intensity",
    colorbarFormat: "%.2f",
    subplotIndex: 0,
  });
  plot.image("rgb image", makeImage(96, 128), {
    rows: 96,
    cols: 128,
    channels: 3,
    bounds: [[0, 0], [128, 96]],
    subplotIndex: 1,
  });
  state.colormapPlots.push(plot);
  return plot;
}

async function buildOverlays() {
  const n = 220;
  const signal = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    signal[i] = Math.sin(i * 0.06) + 0.18 * Math.sin(i * 0.31);
  }
  const plot = await mountPlot("overlays-plot", { title: "Overlays + Pie Chart" });
  plot.setAxisLabel("x1", "sample");
  plot.setAxisLabel("y1", "value");
  plot.line("signal", signal, { color: "#1f6f66", lineWeight: 2 });
  plot.vlines("release windows", new Float32Array([35, 82, 155]));
  plot.hlines("thresholds", new Float32Array([-0.75, 0.75]));
  plot.tagX(82, { labelFmt: "deploy", roundValue: false });
  plot.tagY(0, { labelFmt: "baseline", roundValue: false });
  plot.text("inline text", 18, 1.2);
  plot.annotation("largest visible peak", 27, 1.18, { offsetX: 12, offsetY: -20 });
  plot.pieChart("allocation", new Float32Array([42, 28, 18, 12]), {
    labels: ["compute", "io", "cache", "idle"],
    x: 170,
    y: 0,
    radius: 28,
    labelFmt: "%.0f",
  });
  plot.dummy("legend placeholder");
  plot.setView(0, 220, -1.7, 1.7);
  return plot;
}

async function buildAxes() {
  const n = 96;
  const x = range(n, 60);
  const latency = new Float32Array(n);
  const requests = new Float32Array(n);
  const tickValues = new Float32Array([0, 900, 1800, 2700, 3600, 4500, 5400]);
  const tickLabels = ["09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00"];
  for (let i = 0; i < n; i += 1) {
    latency[i] = 4 + 0.08 * i + 2.4 * Math.abs(Math.sin(i * 0.19));
    requests[i] = 900 + 260 * Math.sin(i * 0.08) + 120 * Math.cos(i * 0.17);
  }
  const plot = await mountPlot("axes-plot", { title: "Axis Controls" });
  plot.setSecondaryAxes({ y2: true });
  plot.setAxisScale({ x: "linear", y: "log" });
  plot.setAxisLabel("x1", "clock");
  plot.setAxisLabel("y1", "latency ms - log");
  plot.setAxisLabel("y2", "requests/sec");
  plot.setAxisFormat("y1", "%.1f");
  plot.setAxisFormat("y2", "%.0f");
  plot.setAxisTicks("x1", tickValues, { labels: tickLabels, keepDefault: false });
  plot.setAxisZoomConstraints("x1", 120, 7200);
  plot.scatter("latency p95", latency, { x, marker: "circle" });
  plot.scatter("throughput", requests, { x, yAxis: "y2", marker: "diamond" });
  return plot;
}

async function buildSubplots() {
  const n = 420;
  const x = range(n, 0.025);
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  const c = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    a[i] = Math.sin(i * 0.035);
    b[i] = Math.cos(i * 0.05) * Math.sin(i * 0.011);
    c[i] = Math.abs(Math.sin(i * 0.09)) + 0.05;
  }
  const plot = await mountPlot("subplots-plot", { title: "Linked 2x2 Subplots" });
  plot.setSubplots(2, 2, { linkAllX: true, shareItems: false });
  plot.line("trend", a, { subplotIndex: 0, color: "#1f6f66" });
  plot.scatter("phase", b, { x, subplotIndex: 1 });
  plot.bars("magnitude", c.subarray(0, 64), { subplotIndex: 2 });
  plot.heatmap("tile", makeMatrix(48, 64), {
    rows: 48,
    cols: 64,
    labelFmt: "",
    showColorbar: false,
    subplotIndex: 3,
  });
  state.colormapPlots.push(plot);
  return plot;
}

async function buildDrag() {
  const n = 180;
  const y = new Float32Array(n);
  for (let i = 0; i < n; i += 1) y[i] = Math.sin(i * 0.08);
  const plot = await mountPlot("drag-plot", { title: "Drag Primitives + Drag/Drop" });
  plot.line("reference", y, { color: "#1f6f66" });
  plot.dragLineX("cursor x", 42, { thickness: 2 });
  plot.dragLineY("cursor y", 0.35, { thickness: 2 });
  plot.dragPoint("anchor", 82, 0.75, { size: 8 });
  plot.dragRect("window", 105, -0.55, 145, 0.55);
  plot.dragDropPlot({ source: true, target: true });
  plot.dragDropAxis("x1", { source: true, target: true });
  plot.dragDropLegend({ target: true });
  plot.onInteraction((events) => {
    if (!interactionReadout) return;
    const active = events.find((event) => event.active) || events[events.length - 1];
    if (!active) return;
    interactionReadout.textContent = `Interaction events: kind=${active.kind}, id=${active.id}, active=${active.active}, values=(${active.v0.toFixed(2)}, ${active.v1.toFixed(2)}, ${active.v2.toFixed(2)}, ${active.v3.toFixed(2)})`;
  });
  return plot;
}

async function buildColormapWidgets() {
  const plot = await mountPlot("colormap-plot", { title: "Colormap Widgets" });
  plot.setColormap(state.activeColormap);
  plot.heatmap("surface", makeMatrix(72, 120, 0.6), {
    rows: 72,
    cols: 120,
    labelFmt: "",
    showColorbar: true,
    colorbarLabel: "z",
    colorbarFormat: "%.2f",
  });
  plot.colormapSelector({ label: "Choose map" });
  plot.colormapSlider({ label: "Sample", labelFmt: "%.2f", value: 0.62 });
  plot.colormapButton({ label: "Color button", width: 110, height: 24 });
  state.colormapPlots.push(plot);
  return plot;
}

async function main() {
  const probe = probeWebGL2();
  if (!probe.available) {
    setMode("WebGL2 unavailable");
    for (const id of ids) setHostError(id, probe.reason);
    return;
  }

  state.activeColormap = colormapSelect?.value || "Viridis";

  const builders = [
    ["line-lod-plot", buildLineLod],
    ["streaming-plot", buildStreaming],
    ["scatter-plot", buildScatter],
    ["curve-plot", buildCurveVariants],
    ["bars-plot", buildBars],
    ["distribution-plot", buildDistributions],
    ["heatmap-image-plot", buildHeatmapImage],
    ["overlays-plot", buildOverlays],
    ["axes-plot", buildAxes],
    ["subplots-plot", buildSubplots],
    ["drag-plot", buildDrag],
    ["colormap-plot", buildColormapWidgets],
  ];

  state.totalExamples = builders.length;
  state.builders = new Map(builders);
  updateLoadMode();
  setupLazyLoading();

  on(updateButton, "click", async () => {
    await loadExample("line-lod-plot");
    if (!state.lineSeries || !state.lineData) return;
    state.linePhase += 0.55;
    makeSignal(state.lineData, state.linePhase);
    state.lineSeries.setData(state.lineData);
  });

  on(streamButton, "click", async () => {
    await loadExample("streaming-plot");
    if (!state.streamHandle) return;
    state.streaming = !state.streaming;
    streamButton.textContent = state.streaming ? "Stop Stream" : "Start Stream";
    if (!state.streaming) {
      for (const timer of state.timers) window.clearInterval(timer);
      state.timers = [];
      return;
    }
    const timer = window.setInterval(appendStreamChunk, 220);
    state.timers.push(timer);
  });

  on(autoscaleButton, "click", () => {
    for (const plot of state.plots) plot.autoscale();
  });

  on(colormapSelect, "change", () => {
    state.activeColormap = colormapSelect.value;
    for (const plot of state.colormapPlots) plot.setColormap(state.activeColormap);
  });

  window.addEventListener("beforeunload", () => state.dispose(), { once: true });
}

main().catch((error) => {
  console.error("Failed to initialize nbimplot examples", error);
  setMode("failed");
  for (const id of ids) setHostError(id, error);
});
