import { createPlot, probeWebGL2 } from "/vendor/nbimplot/src/index.js";

const mode = document.querySelector("#mode");
const frameMs = document.querySelector("#frame-ms");
const updateButton = document.querySelector("#update-data");
const streamButton = document.querySelector("#toggle-stream");
const autoscaleButton = document.querySelector("#autoscale");

const probe = probeWebGL2();
if (!probe.available) {
  mode.textContent = "WebGL2 unavailable";
  throw new Error(probe.reason);
}

mode.textContent = "loading wasm";

const n = 1_000_000;
const y = new Float32Array(n);

function fillSignal(phase = 0) {
  for (let i = 0; i < n; i += 1) {
    const x = i * 0.001;
    y[i] = Math.sin(x + phase) + 0.18 * Math.sin(i * 0.017) + 0.08 * Math.cos(i * 0.00031);
  }
}

fillSignal();

const linePlot = await createPlot("#line-plot", {
  responsive: true,
  title: "Million Point Line",
  crosshairs: true,
});

const line = linePlot.line("signal", y, {
  color: "#2563eb",
  lineWeight: 2,
});

const rows = 96;
const cols = 144;
const z = new Float32Array(rows * cols);
for (let r = 0; r < rows; r += 1) {
  for (let c = 0; c < cols; c += 1) {
    z[r * cols + c] =
      Math.sin(r * 0.13) * Math.cos(c * 0.08) +
      0.35 * Math.sin((r + c) * 0.045);
  }
}

const heatmapPlot = await createPlot("#heatmap-plot", {
  responsive: true,
  title: "Heatmap + Colormap",
  colormap: "Viridis",
});

heatmapPlot.heatmap("z", z, {
  rows,
  cols,
  labelFmt: "",
  showColorbar: true,
  colorbarLabel: "Intensity",
  colorbarFormat: "%.2f",
});
heatmapPlot.tagX(40, { labelFmt: "x=%.0f" });
heatmapPlot.tagY(30, { labelFmt: "y=%.0f" });

linePlot.onPerfStats((stats) => {
  frameMs.textContent = `${stats.frameMs.toFixed(2)} ms`;
});

mode.textContent = "ready";

let phase = 0;
let streamTimer = 0;

updateButton.addEventListener("click", () => {
  phase += 0.6;
  fillSignal(phase);
  line.setData(y);
});

streamButton.addEventListener("click", () => {
  if (streamTimer) {
    clearInterval(streamTimer);
    streamTimer = 0;
    streamButton.textContent = "Stream";
    return;
  }
  streamButton.textContent = "Stop";
  streamTimer = window.setInterval(() => {
    phase += 0.15;
    fillSignal(phase);
    line.setData(y);
  }, 700);
});

autoscaleButton.addEventListener("click", () => {
  linePlot.autoscale();
  heatmapPlot.autoscale();
});
