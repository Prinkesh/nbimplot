import { createPlot, probeWebGL2 } from "./vendor/nbimplot/src/index.js?v=hero-showcase";

const previousDemo = window.__nbimplotExamplesDemo;
if (previousDemo?.dispose) previousDemo.dispose();

const HERO_ID = "hero-showcase-plot";
const MAX_ACTIVE_PLOTS = 5;

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
  forceLoadIds: new Set(),
  builders: new Map(),
  colormapPlots: [],
  lineSeries: null,
  lineData: null,
  lineX: null,
  linePhase: 0,
  streamHandle: null,
  streamPlot: null,
  streamSample: 0,
  streaming: false,
  streamPaused: false,
  savedPlotState: null,
  crosshairEnabled: true,
  lastSelectionHash: "",
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
    this.forceLoadIds.clear();
    this.builders.clear();
    this.colormapPlots = [];
  },
};
window.__nbimplotExamplesDemo = state;

const ids = [
  "line-lod-plot",
  "streaming-plot",
  "batch-datetime-plot",
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
  "finance-plot",
  "science-plot",
  "advanced-api-plot",
];

const mode = document.querySelector("#mode");
const frameMs = document.querySelector("#frame-ms");
const updateButton = document.querySelector("#update-data");
const streamButton = document.querySelector("#toggle-stream");
const autoscaleButton = document.querySelector("#autoscale");
const exportButton = document.querySelector("#export-png");
const colormapSelect = document.querySelector("#colormap-select");
const pauseStreamButton = document.querySelector("#stream-pause");
const clearStreamButton = document.querySelector("#stream-clear");
const streamWindowButton = document.querySelector("#stream-window");
const selectionDemoButton = document.querySelector("#run-selection-demo");
const exportStateButton = document.querySelector("#export-state");
const restoreStateButton = document.querySelector("#restore-state");
const copyPngButton = document.querySelector("#copy-png");
const crosshairButton = document.querySelector("#toggle-crosshair");
const featureStatus = document.querySelector("#feature-status");
const interactionReadout = document.querySelector("#interaction-readout");

function setMode(text) {
  if (mode) mode.textContent = text;
}

function setFeatureStatus(text) {
  if (featureStatus) featureStatus.textContent = text;
  if (interactionReadout) interactionReadout.textContent = text;
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

function makeFinancialBars(n, seed = 29) {
  const normal = normalFactory(seed);
  const x = range(n, 1);
  const open = new Float32Array(n);
  const high = new Float32Array(n);
  const low = new Float32Array(n);
  const close = new Float32Array(n);
  let price = 100;
  for (let i = 0; i < n; i += 1) {
    open[i] = price;
    const drift = 0.05 * Math.sin(i * 0.07) + normal() * 0.9;
    close[i] = Math.max(1, open[i] + drift);
    high[i] = Math.max(open[i], close[i]) + 0.4 + Math.abs(normal()) * 0.8;
    low[i] = Math.min(open[i], close[i]) - 0.4 - Math.abs(normal()) * 0.8;
    price = close[i];
  }
  return { x, open, high, low, close };
}

function makeField(rows, cols) {
  const z = new Float32Array(rows * cols);
  const qn = 12;
  const qx = new Float32Array(qn * qn);
  const qy = new Float32Array(qn * qn);
  const qu = new Float32Array(qn * qn);
  const qv = new Float32Array(qn * qn);
  for (let r = 0; r < rows; r += 1) {
    const y = -3 + (6 * r) / Math.max(1, rows - 1);
    for (let c = 0; c < cols; c += 1) {
      const x = -3 + (6 * c) / Math.max(1, cols - 1);
      z[r * cols + c] = Math.sin(x * y) + 0.25 * Math.cos(2 * x) - 0.15 * Math.sin(1.7 * y);
    }
  }
  let k = 0;
  for (let r = 0; r < qn; r += 1) {
    const y = -3 + (6 * r) / Math.max(1, qn - 1);
    for (let c = 0; c < qn; c += 1) {
      const x = -3 + (6 * c) / Math.max(1, qn - 1);
      qx[k] = x;
      qy[k] = y;
      qu[k] = -y;
      qv[k] = x;
      k += 1;
    }
  }
  return { z, qx, qy, qu, qv };
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
  plot.setTheme("nbimplot");
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

function clearStreamTimers() {
  for (const timer of state.timers) window.clearInterval(timer);
  state.timers = [];
  state.streaming = false;
  if (streamButton) streamButton.textContent = "Start Stream";
}

function startStreamTimers() {
  if (state.streaming) return;
  state.streaming = true;
  if (streamButton) streamButton.textContent = "Stop Stream";
  const timer = window.setInterval(appendStreamChunk, 220);
  state.timers.push(timer);
}

function currentLoadedPlot() {
  const selected = Array.from(state.visibleIds)
    .map((id) => [id, state.plotById.get(id)])
    .find(([, plot]) => Boolean(plot));
  if (selected) return selected;
  if (state.plots.length > 0) return ["nbimplot-demo", state.plots[0]];
  return null;
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
    clearStreamTimers();
    state.streamPaused = false;
    if (pauseStreamButton) pauseStreamButton.textContent = "Pause Stream";
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
    if (id === HERO_ID) continue;
    if (state.visibleIds.has(id)) continue;
    releaseExample(id);
    if (state.plotById.size <= MAX_ACTIVE_PLOTS) break;
  }
}

function loadExample(id, options = {}) {
  const force = Boolean(options.force);
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
    .then(() => {
      const forced = force || state.forceLoadIds.has(id);
      return state.disposed || (!forced && !state.visibleIds.has(id)) ? null : runExample(id, builder);
    })
    .then((plot) => {
      const forced = force || state.forceLoadIds.has(id);
      state.forceLoadIds.delete(id);
      state.loadingIds.delete(id);
      state.loadPromises.delete(id);
      if (plot) {
        state.loadedIds.add(id);
        state.visitedIds.add(id);
        if (state.visibleIds.has(id) || forced) {
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
      state.forceLoadIds.delete(id);
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

async function focusExample(id) {
  const host = document.querySelector(`#${id}`);
  if (host) {
    host.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  if (state.plotById.has(id)) {
    state.forceLoadIds.delete(id);
    return state.plotById.get(id);
  }
  state.forceLoadIds.add(id);
  const loaded = await loadExample(id, { force: true });
  if (!loaded) {
    throw new Error(`Unable to load ${id}.`);
  }
  return state.plotById.get(id) || null;
}

function action(handler) {
  return async (event) => {
    try {
      await handler(event);
    } catch (error) {
      console.error("nbimplot demo action failed", error);
      setFeatureStatus(error instanceof Error ? error.message : String(error));
    }
  };
}

function restorablePlotState(plot) {
  const snapshot = plot.getState({ includeData: false });
  return {
    theme: snapshot.theme,
    colormap: snapshot.colormap,
    view: snapshot.view,
    linkedCrosshair: snapshot.linkedCrosshair,
    axisLabels: snapshot.axisLabels,
    axisFormats: snapshot.axisFormats,
  };
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

async function buildHeroShowcase() {
  const n = 4_800;
  const x = range(n, 0.02);
  const signal = new Float32Array(n);
  const lower = new Float32Array(n);
  const upper = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const base = Math.sin(i * 0.018) * 0.55 + Math.cos(i * 0.006) * 0.28;
    const pulse = i % 829 === 0 ? 0.85 : 0;
    signal[i] = base + 0.11 * Math.sin(i * 0.13) + pulse;
    lower[i] = base - 0.28 - 0.05 * Math.sin(i * 0.025);
    upper[i] = base + 0.28 + 0.05 * Math.cos(i * 0.021) + pulse * 0.35;
  }

  const eventCount = 9;
  const eventX = new Float32Array(eventCount);
  const eventY = new Float32Array(eventCount);
  for (let i = 0; i < eventCount; i += 1) {
    const idx = Math.min(n - 1, 360 + i * 470);
    eventX[i] = x[idx];
    eventY[i] = signal[idx];
  }

  const plot = await mountPlot(HERO_ID, {
    title: "Live WASM ImPlot Surface",
    crosshairs: true,
  });
  plot.setAxisLabel("x1", "time");
  plot.setAxisLabel("y1", "signal");
  plot.setAxisFormat("y1", "%.2f");
  plot.shaded("envelope", lower, upper, { x, alpha: 0.22 });
  plot.line("live signal", signal, { x, color: "#41e2cd", lineWeight: 2 });
  plot.scatter("events", eventY, { x: eventX, marker: "diamond", size: 5, color: "#ffc46f" });
  plot.vlines("deploys", new Float32Array([18, 43, 72]));
  plot.tagY(0, { labelFmt: "zero", roundValue: false });
  plot.setView(0, 96, -1.15, 1.45);
  plot.onHover((event) => {
    setFeatureStatus(`Hero hover: ${event.seriesName} x=${event.x.toFixed(2)} y=${event.y.toFixed(3)}`);
  });
  plot.onPerfStats((stats) => {
    if (!frameMs) return;
    frameMs.textContent = `${stats.frameMs.toFixed(2)} ms | hero ${Math.round(stats.drawPoints).toLocaleString()} drawn`;
  });
  return plot;
}

async function buildLineLod() {
  const n = 1_000_000;
  const x = range(n, 0.001);
  const y = makeSignal(new Float32Array(n));
  const plot = await mountPlot("line-lod-plot", {
    title: "Million Point Line - Custom X + WASM LOD",
  });
  plot.setAxisLabel("x1", "time (s)");
  plot.setAxisLabel("y1", "value");
  plot.setAxisFormat("y1", "%.2f");
  const handle = plot.line("signal", y, { x, color: "#1f6f66", lineWeight: 2 });
  plot.hlines("baseline", new Float32Array([0]));
  plot.vlines("spike markers", new Float32Array([131.071, 262.142, 524.284, 786.426]));
  plot.tagY(0, { labelFmt: "zero", roundValue: false });
  plot.onHover((event) => {
    setFeatureStatus(`Hover: ${event.seriesName} index=${event.index.toLocaleString()} x=${event.x.toFixed(3)} y=${event.y.toFixed(3)}`);
  });
  plot.onClick((event) => {
    setFeatureStatus(`Click: button=${event.button} x=${event.x.toFixed(3)} y=${event.y.toFixed(3)}`);
  });
  plot.onSelection((event) => {
    const exact = plot.indicesForSelection(event, handle);
    const selectionHash = `${event.xMin}:${event.xMax}:${event.yMin}:${event.yMax}`;
    if (selectionHash !== state.lastSelectionHash) {
      state.lastSelectionHash = selectionHash;
      plot.highlightSelection(event, handle, { name: "selected", size: 5 });
    }
    const csv = plot.exportCSVSelection(event, handle);
    setFeatureStatus(`Selection: x=[${event.xMin.toFixed(3)}, ${event.xMax.toFixed(3)}], exact signal points=${exact.length.toLocaleString()}, CSV bytes=${csv.length.toLocaleString()}`);
  });
  plot.onPerfStats((stats) => {
    if (!frameMs) return;
    frameMs.textContent = `${stats.frameMs.toFixed(2)} ms | ${Math.round(stats.drawPoints).toLocaleString()} drawn`;
  });
  state.lineSeries = handle;
  state.lineData = y;
  state.lineX = x;
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
  const initialX = new Float32Array(256);
  for (let i = 0; i < initial.length; i += 1) {
    initialX[i] = i;
    initial[i] = Math.sin(i * 0.06) + 0.15 * Math.sin(i * 0.31);
  }
  state.streamSample = initial.length;
  state.streamHandle = plot.streamLine("ticks", {
    capacity: 12_000,
    initial,
    x: initialX,
    autoRender: true,
    color: "#b74b2b",
    lineWeight: 2,
  });
  state.streamHandle.setStreamOptions({ autoRender: true });
  state.streamPlot = plot;
  state.streamPaused = false;
  if (pauseStreamButton) pauseStreamButton.textContent = "Pause Stream";
  return plot;
}

async function buildBatchDatetime() {
  const n = 360;
  const dates = new Array(n);
  const mid = new Float32Array(n);
  const vwap = new Float32Array(n);
  const scoreX = ["baseline", "candidate-a", "candidate-b", "production"];
  const scores = new Float32Array([0.72, 0.91, 0.64, 0.83]);
  const start = Date.UTC(2026, 0, 1, 9, 30, 0);
  for (let i = 0; i < n; i += 1) {
    dates[i] = new Date(start + i * 60_000);
    mid[i] = Math.sin(i * 0.035) + 0.15 * Math.sin(i * 0.19);
    vwap[i] = mid[i] + 0.08 * Math.cos(i * 0.09);
  }

  const plot = await mountPlot("batch-datetime-plot", {
    title: "Batch Lines + Time + Categories",
  });
  plot.setTheme("publication");
  plot.setSubplots(1, 2, { noResize: false });
  plot.setAxisLabel("x1", "time/category");
  plot.setAxisLabel("y1", "value");
  const handles = plot.lines({
    mid: { x: dates, y: mid, color: "#1f6f66" },
    vwap: { x: dates, y: vwap, color: "#b74b2b" },
  }, {
    subplotIndex: 0,
    lineWeight: 1.7,
  });
  plot.scatter("model scores", scores, {
    x: scoreX,
    subplotIndex: 1,
    size: 6,
    marker: "diamond",
  });
  const html = plot.exportHTML({ title: "nbimplot batch datetime snapshot" });
  plot.onHover((event) => {
    setFeatureStatus(`Batch/datetime hover: ${event.seriesName} x=${event.x.toFixed(2)} y=${event.y.toFixed(3)}; HTML export=${html.length.toLocaleString()} bytes`);
  });
  setFeatureStatus(`Batch/datetime example loaded: ${handles.length} lines, categorical scatter, HTML export=${html.length.toLocaleString()} bytes.`);
  return plot;
}

function appendStreamChunk() {
  if (!state.streamHandle) return;
  const chunk = new Float32Array(96);
  const chunkX = new Float32Array(96);
  for (let i = 0; i < chunk.length; i += 1) {
    const t = state.streamSample + i;
    chunkX[i] = t;
    chunk[i] = Math.sin(t * 0.035) + 0.24 * Math.sin(t * 0.19) + 0.08 * Math.cos(t * 0.006);
  }
  state.streamSample += chunk.length;
  state.streamHandle.append(chunk, { x: chunkX });
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
  plot.setLinkedCrosshair("subplot-demo", { axis: "x" });
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
    const active = events.find((event) => event.active) || events[events.length - 1];
    if (!active) return;
    setFeatureStatus(`Interaction events: kind=${active.kind}, id=${active.id}, active=${active.active}, values=(${active.v0.toFixed(2)}, ${active.v1.toFixed(2)}, ${active.v2.toFixed(2)}, ${active.v3.toFixed(2)})`);
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

async function buildFinanceSpecialty() {
  const bars = makeFinancialBars(180);
  const plot = await mountPlot("finance-plot", {
    title: "Candlestick + OHLC",
    crosshairs: true,
  });
  plot.setTheme("finance");
  plot.setAxisLabel("x1", "bar");
  plot.setAxisLabel("y1", "price");
  plot.candlestick("candles", bars.open, bars.high, bars.low, bars.close, {
    x: bars.x,
    width: 0.72,
  });
  plot.ohlc("ohlc", bars.open, bars.high, bars.low, bars.close, {
    x: bars.x,
    width: 0.35,
  });
  plot.onHover((event) => {
    setFeatureStatus(`Finance hover: ${event.seriesName} x=${event.x.toFixed(0)} y=${event.y.toFixed(2)}`);
  });
  return plot;
}

async function buildScienceSpecialty() {
  const rows = 80;
  const cols = 96;
  const field = makeField(rows, cols);
  const levels = new Float32Array([-1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2]);
  const x = range(cols, 6 / Math.max(1, cols - 1), -3);
  const offsets = new Float32Array(16);
  for (let i = 0; i < offsets.length; i += 1) offsets[i] = -2.6 + i * 0.34;

  const plot = await mountPlot("science-plot", {
    title: "Contour + Quiver + Waterfall + Spectrogram",
  });
  plot.setTheme("lab");
  plot.setColormap(state.activeColormap);
  plot.setSubplots(2, 2, { noResize: false, shareItems: false });
  plot.contour("contour", field.z, {
    rows,
    cols,
    levels,
    bounds: [[-3, -3], [3, 3]],
    lineWeight: 1.4,
    subplotIndex: 0,
  });
  plot.quiver("vector field", field.qx, field.qy, field.qu, field.qv, {
    scale: 0.08,
    normalize: true,
    subplotIndex: 1,
  });
  plot.waterfall("waterfall", field.z.subarray(0, offsets.length * cols), {
    rows: offsets.length,
    cols,
    x,
    yOffsets: offsets,
    scale: 0.2,
    subplotIndex: 2,
  });
  plot.spectrogram("spectrogram", field.z, {
    rows,
    cols,
    bounds: [[-3, -3], [3, 3]],
    labelFmt: "",
    showColorbar: true,
    colorbarLabel: "z",
    colorbarFormat: "%.2f",
    subplotIndex: 3,
  });
  state.colormapPlots.push(plot);
  plot.onHover((event) => {
    setFeatureStatus(`Science hover: ${event.seriesName} x=${event.x.toFixed(2)} y=${event.y.toFixed(2)}`);
  });
  return plot;
}

async function buildAdvancedApi() {
  const n = 240;
  const x = range(n, 60);
  const primary = new Float32Array(n);
  const secondary = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    primary[i] = 0.7 * Math.sin(i * 0.08) + 0.25 * Math.cos(i * 0.017);
    secondary[i] = 120 + 35 * Math.sin(i * 0.05) + 12 * Math.cos(i * 0.19);
  }

  const plot = await mountPlot("advanced-api-plot", {
    title: "Advanced API Controls",
    crosshairs: true,
  });
  plot.setPlotFlags({ noLegend: false, noMenus: false, noBoxSelect: false, crosshairs: true });
  plot.setSubplots(1, 2, { linkAllX: true, shareItems: false });
  plot.setTheme("notebook");
  plot.setLinkedCrosshair("advanced-api-demo", { axis: "xy" });
  plot.setAlignedGroup("advanced-api-demo", { enabled: true, vertical: true });
  plot.setSecondaryAxes({ x2: true, y2: true });
  plot.setTimeAxis("x1");
  plot.setAxisState("x2", { enabled: true, scale: "time" });
  plot.setAxisState("y2", { enabled: true, scale: "linear" });
  plot.setAxisLink("x2", "x1");
  plot.setAxisLimitsConstraints("y1", -1.4, 1.4);
  plot.setAxisZoomConstraints("x1", 5 * 60, 180 * 60);
  plot.setAxisLabel("x1", "time axis");
  plot.setAxisLabel("x2", "linked time axis");
  plot.setAxisLabel("y1", "signal");
  plot.setAxisLabel("y2", "load");
  plot.setAxisFormat("y1", "%.2f");
  plot.setAxisFormat("y2", "%.0f");

  const tickValues = new Float32Array([0, 1800, 3600, 5400, 7200, 9000, 10800, 12600]);
  const tickLabels = ["00:00", "00:30", "01:00", "01:30", "02:00", "02:30", "03:00", "03:30"];
  plot.setAxisTicks("x1", tickValues, { labels: tickLabels, keepDefault: false });
  plot.setAxisTicks("x2", tickValues, { labels: tickLabels, keepDefault: false });
  plot.clearAxisTicks("x2");

  const handle = plot.line("primary", primary, {
    x,
    subplotIndex: 0,
    color: "#1f6f66",
    lineWeight: 2,
  });
  plot.line("secondary y2", secondary, {
    x,
    yAxis: "y2",
    subplotIndex: 0,
    color: "#b74b2b",
    lineWeight: 1.6,
  });
  plot.infLines("maintenance windows", new Float32Array([3600, 7200, 10800]), {
    axis: "x",
    subplotIndex: 0,
  });
  plot.primitive("tag_y", {
    value: 0.85,
    labelFmt: "direct primitive",
    roundValue: false,
    subplotIndex: 0,
  });

  const coarse = new Float32Array(48);
  const coarseX = new Float32Array(48);
  for (let i = 0; i < coarse.length; i += 1) {
    coarseX[i] = i * 5 * 60;
    coarse[i] = primary[Math.min(primary.length - 1, i * 5)];
  }
  plot.scatter("linked samples", coarse, {
    x: coarseX,
    subplotIndex: 1,
    marker: "diamond",
    size: 4,
  });
  plot.line("linked trend", primary, {
    x,
    subplotIndex: 1,
    color: "#3f5f8f",
  });
  plot.setView(0, 4 * 3600, -1.25, 1.25);

  plot.onViewChange((view) => {
    setFeatureStatus(`View: x=[${view.xMin.toFixed(0)}, ${view.xMax.toFixed(0)}], y=[${view.yMin.toFixed(2)}, ${view.yMax.toFixed(2)}]`);
  });
  plot.onSelection((event) => {
    const exact = plot.selectionIndices(event, handle);
    plot.highlightSelection(event, handle, { name: "advanced selection", size: 5 });
    const csv = plot.exportCSVSelection(event, handle);
    setFeatureStatus(`Advanced selection: ${exact.length.toLocaleString()} primary samples, CSV bytes=${csv.length.toLocaleString()}`);
  });
  plot.onPerfStats((stats) => {
    if (!frameMs) return;
    const current = plot.getPerfStats();
    frameMs.textContent = `${stats.frameMs.toFixed(2)} ms | latest ${current.frameMs.toFixed(2)} ms`;
  });

  const view = plot.getView();
  const perf = plot.getPerfStats();
  const stateSnapshot = plot.getState({ includeData: true });
  const jsonSnapshot = plot.exportJSONState({ includeData: false });
  plot.setState({
    theme: stateSnapshot.theme,
    colormap: stateSnapshot.colormap,
    linkedCrosshair: stateSnapshot.linkedCrosshair,
  });
  if (view && perf) {
    setFeatureStatus(`Initial advanced view ready; draw=${Math.round(perf.drawPoints).toLocaleString()} points, state JSON=${jsonSnapshot.length.toLocaleString()} bytes`);
  }
  plot.requestRender();
  plot.draw();
  return plot;
}

async function main() {
  setHostStatus(HERO_ID, "Loading live WASM plot...");
  const probe = probeWebGL2();
  if (!probe.available) {
    setMode("WebGL2 unavailable");
    setHostError(HERO_ID, probe.reason);
    for (const id of ids) setHostError(id, probe.reason);
    return;
  }

  state.activeColormap = colormapSelect?.value || "Viridis";

  const builders = [
    ["line-lod-plot", buildLineLod],
    ["streaming-plot", buildStreaming],
    ["batch-datetime-plot", buildBatchDatetime],
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
    ["finance-plot", buildFinanceSpecialty],
    ["science-plot", buildScienceSpecialty],
    ["advanced-api-plot", buildAdvancedApi],
  ];

  state.totalExamples = builders.length;
  state.builders = new Map(builders);
  updateLoadMode();
  await runExample(HERO_ID, buildHeroShowcase);
  updateLoadMode();
  setupLazyLoading();

  on(updateButton, "click", action(async () => {
    await focusExample("line-lod-plot");
    if (!state.lineSeries || !state.lineData) return;
    state.linePhase += 0.55;
    makeSignal(state.lineData, state.linePhase);
    state.lineSeries.setData(state.lineData, { x: state.lineX });
    setFeatureStatus(`Updated 1,000,000 y values in place at phase=${state.linePhase.toFixed(2)}.`);
  }));

  on(streamButton, "click", action(async () => {
    await focusExample("streaming-plot");
    if (!state.streamHandle) return;
    if (state.streamPaused) {
      state.streamHandle.resume();
      state.streamPaused = false;
      if (pauseStreamButton) pauseStreamButton.textContent = "Pause Stream";
    }
    state.streaming = !state.streaming;
    streamButton.textContent = state.streaming ? "Stop Stream" : "Start Stream";
    if (!state.streaming) {
      clearStreamTimers();
      setFeatureStatus("Streaming timer stopped. Existing ring-buffer data stays in the plot.");
      return;
    }
    const timer = window.setInterval(appendStreamChunk, 220);
    state.timers.push(timer);
    setFeatureStatus("Streaming started: appending explicit x/y chunks into the WASM-backed ring buffer.");
  }));

  on(autoscaleButton, "click", () => {
    for (const plot of state.plots) plot.autoscale();
    setFeatureStatus(`Autoscaled ${state.plots.length} active WASM plot(s).`);
  });

  on(exportButton, "click", action(async () => {
    let selected = currentLoadedPlot();
    if (!selected) {
      await focusExample("line-lod-plot");
      selected = ["line-lod-plot", state.plotById.get("line-lod-plot")];
    }
    const [id, plot] = selected;
    if (plot) {
      const blob = await plot.downloadPNG(`${id}.png`);
      setFeatureStatus(`Downloaded ${id}.png (${blob.size.toLocaleString()} bytes).`);
    }
  }));

  on(colormapSelect, "change", () => {
    state.activeColormap = colormapSelect.value;
    for (const plot of state.colormapPlots) plot.setColormap(state.activeColormap);
    setFeatureStatus(`Applied ${state.activeColormap} to ${state.colormapPlots.length} colormap-aware plot(s).`);
  });

  on(pauseStreamButton, "click", action(async () => {
    await focusExample("streaming-plot");
    if (!state.streamHandle) return;
    state.streamPaused = !state.streamPaused;
    if (state.streamPaused) {
      state.streamHandle.pause();
      clearStreamTimers();
      pauseStreamButton.textContent = "Resume Stream";
      setFeatureStatus("Stream paused. Appends are ignored until resume, and the timer has been stopped.");
      return;
    }
    state.streamHandle.resume();
    pauseStreamButton.textContent = "Pause Stream";
    startStreamTimers();
    setFeatureStatus("Stream resumed and timer restarted.");
  }));

  on(clearStreamButton, "click", action(async () => {
    await focusExample("streaming-plot");
    if (!state.streamHandle) return;
    clearStreamTimers();
    state.streamHandle.resume();
    state.streamPaused = false;
    state.streamSample = 1;
    state.streamHandle.clear({ x0: 0, y0: 0 });
    if (pauseStreamButton) pauseStreamButton.textContent = "Pause Stream";
    setFeatureStatus("Stream ring buffer cleared to one seed point without recreating the plot.");
  }));

  on(streamWindowButton, "click", action(async () => {
    await focusExample("streaming-plot");
    if (!state.streamHandle) return;
    state.streamHandle.setWindow(3000);
    setFeatureStatus("Stream window set to 3,000 points. Future appends keep only the latest samples.");
  }));

  on(selectionDemoButton, "click", action(async () => {
    const plot = await focusExample("line-lod-plot");
    if (!plot || !state.lineSeries) return;
    const selection = {
      subplotIndex: 0,
      xMin: 130.2,
      xMax: 132.0,
      yMin: -0.55,
      yMax: 2.15,
    };
    const exact = plot.indicesForSelection(selection, state.lineSeries);
    const csv = plot.exportCSVSelection(selection, state.lineSeries);
    plot.highlightSelection(selection, state.lineSeries, { name: "workbench selection", size: 6 });
    plot.setView(124, 138, -0.9, 2.35);
    setFeatureStatus(`Selection highlighted: ${exact.length.toLocaleString()} exact signal points, CSV bytes=${csv.length.toLocaleString()}.`);
  }));

  on(exportStateButton, "click", action(async () => {
    const plot = await focusExample("advanced-api-plot");
    if (!plot) return;
    state.savedPlotState = restorablePlotState(plot);
    const text = plot.exportJSONState({
      includeData: false,
      filename: "nbimplot-advanced-state.json",
    });
    setFeatureStatus(`Downloaded state JSON (${text.length.toLocaleString()} bytes) and cached it for Restore State.`);
  }));

  on(restoreStateButton, "click", action(async () => {
    const plot = await focusExample("advanced-api-plot");
    if (!plot) return;
    if (!state.savedPlotState) {
      state.savedPlotState = restorablePlotState(plot);
    }
    plot.setColormap("Hot");
    plot.setView(0, 45 * 60, -0.35, 0.35);
    plot.setState(state.savedPlotState);
    setFeatureStatus("Restored cached plot state: theme, colormap, axis labels, view, and linked-crosshair settings.");
  }));

  on(copyPngButton, "click", action(async () => {
    let selected = currentLoadedPlot();
    if (!selected) {
      await focusExample("line-lod-plot");
      selected = ["line-lod-plot", state.plotById.get("line-lod-plot")];
    }
    const [id, plot] = selected;
    if (!plot) return;
    const blob = await plot.copyPNGToClipboard();
    setFeatureStatus(`Copied ${id} PNG to clipboard (${blob.size.toLocaleString()} bytes).`);
  }));

  on(crosshairButton, "click", action(async () => {
    await focusExample("subplots-plot");
    state.crosshairEnabled = !state.crosshairEnabled;
    let updated = 0;
    for (const plot of state.plots) {
      plot.setLinkedCrosshair("demo-link", {
        enabled: state.crosshairEnabled,
        axis: "xy",
      });
      updated += 1;
    }
    crosshairButton.textContent = state.crosshairEnabled ? "Disable Crosshair Link" : "Enable Crosshair Link";
    setFeatureStatus(`${state.crosshairEnabled ? "Enabled" : "Disabled"} linked crosshair on ${updated} active plot(s).`);
  }));

  window.addEventListener("beforeunload", () => state.dispose(), { once: true });
}

main().catch((error) => {
  console.error("Failed to initialize nbimplot examples", error);
  setMode("failed");
  for (const id of ids) setHostError(id, error);
});
