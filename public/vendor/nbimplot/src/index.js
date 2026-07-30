import createNbImPlotModule from "../wasm/nbimplot_wasm.js?v=feature-dashboard";

const DEFAULT_WASM_URL = new URL("../wasm/nbimplot_wasm.wasm?v=feature-dashboard", import.meta.url);
const LABEL_SEP = "\x1f";
const PIE_FMT_SEP = "\x1e";
const HEATMAP_META_SEP = "\x1d";

export const PLOT_FLAGS = Object.freeze({
  NO_LEGEND: 1 << 0,
  NO_MENUS: 1 << 1,
  NO_BOX_SELECT: 1 << 2,
  NO_MOUSE_POS: 1 << 3,
  CROSSHAIRS: 1 << 4,
  EQUAL: 1 << 5,
});

export const SUBPLOT_FLAGS = Object.freeze({
  NO_LEGEND: 1 << 0,
  NO_MENUS: 1 << 1,
  NO_RESIZE: 1 << 2,
  NO_ALIGN: 1 << 3,
  SHARE_ITEMS: 1 << 4,
  LINK_ROWS: 1 << 5,
  LINK_COLS: 1 << 6,
  LINK_ALL_X: 1 << 7,
  LINK_ALL_Y: 1 << 8,
  COL_MAJOR: 1 << 9,
});

export const PRIMITIVE_KIND_CODES = Object.freeze({
  scatter: 1,
  bubbles: 2,
  stairs: 3,
  stems: 4,
  digital: 5,
  bars: 6,
  bar_groups: 7,
  bars_h: 8,
  shaded: 9,
  error_bars: 10,
  error_bars_h: 11,
  inf_lines: 12,
  histogram: 13,
  histogram2d: 14,
  heatmap: 15,
  image: 16,
  pie_chart: 17,
  text: 18,
  annotation: 19,
  dummy: 20,
  drag_line_x: 21,
  drag_line_y: 22,
  drag_point: 23,
  drag_rect: 24,
  tag_x: 25,
  tag_y: 26,
  colormap_slider: 27,
  colormap_button: 28,
  colormap_selector: 29,
  drag_drop_plot: 30,
  drag_drop_axis: 31,
  drag_drop_legend: 32,
});

export const AXES = Object.freeze({
  x1: 0,
  x2: 1,
  x3: 2,
  y1: 3,
  y2: 4,
  y3: 5,
});

export const AXIS_SCALES = Object.freeze({
  linear: 0,
  log: 1,
  time: 2,
});

export const MARKERS = Object.freeze({
  none: -2,
  auto: -1,
  circle: 0,
  square: 1,
  diamond: 2,
  up: 3,
  down: 4,
  left: 5,
  right: 6,
  cross: 7,
  plus: 8,
  asterisk: 9,
});

let modulePromise = null;
let moduleAssetKey = "";
const linkedCrosshairGroups = new Map();

function linkedCrosshairGroup(groupId) {
  const key = String(groupId || "default");
  let group = linkedCrosshairGroups.get(key);
  if (!group) {
    group = new Set();
    linkedCrosshairGroups.set(key, group);
  }
  return group;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function resolveElement(target) {
  if (typeof target === "string") {
    const found = document.querySelector(target);
    if (!found) {
      throw new Error(`nbimplot target selector did not match: ${target}`);
    }
    return found;
  }
  if (target instanceof Element) {
    return target;
  }
  throw new TypeError("createPlot target must be a DOM element or selector.");
}

function ensureFloat32(value, name = "data") {
  if (value instanceof Float32Array) {
    return value;
  }
  if (ArrayBuffer.isView(value)) {
    return Float32Array.from(value);
  }
  if (value instanceof ArrayBuffer) {
    if (value.byteLength % 4 !== 0) {
      throw new Error(`${name} ArrayBuffer byteLength must be divisible by 4.`);
    }
    return new Float32Array(value);
  }
  if (Array.isArray(value)) {
    return new Float32Array(value);
  }
  throw new TypeError(`${name} must be a Float32Array, typed array, ArrayBuffer, or numeric array.`);
}

function ensureVector(value, name = "data") {
  const out = ensureFloat32(value, name);
  if (out.length === 0) {
    throw new Error(`${name} must not be empty.`);
  }
  return out;
}

function dataUrlToBlob(dataUrl) {
  const comma = dataUrl.indexOf(",");
  if (comma < 0) {
    return new Blob([], { type: "application/octet-stream" });
  }
  const header = dataUrl.slice(0, comma);
  const payload = dataUrl.slice(comma + 1);
  const mime = /^data:([^;,]+)/.exec(header)?.[1] || "application/octet-stream";
  const binary = header.includes(";base64") ? atob(payload) : decodeURIComponent(payload);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}

function sanitizePngFilename(filename) {
  const raw = String(filename || "nbimplot.png").trim() || "nbimplot.png";
  const safe = raw.replace(/[\\/:*?"<>|]+/g, "_");
  return /\.png$/i.test(safe) ? safe : `${safe}.png`;
}

function sanitizeJsonFilename(filename) {
  const raw = String(filename || "nbimplot-state.json").trim() || "nbimplot-state.json";
  const safe = raw.replace(/[\\/:*?"<>|]+/g, "_");
  return /\.json$/i.test(safe) ? safe : `${safe}.json`;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 30000);
}

function ensureLineX(value, yLength) {
  const out = ensureVector(value, "x");
  if (out.length !== yLength) {
    throw new Error("x and y must have the same length.");
  }
  let previous = out[0];
  if (!Number.isFinite(previous)) {
    throw new Error("x must contain only finite values.");
  }
  for (let i = 1; i < out.length; i += 1) {
    const current = out[i];
    if (!Number.isFinite(current) || current < previous) {
      throw new Error("x must be sorted in non-decreasing order for line LOD.");
    }
    previous = current;
  }
  return out;
}

function concatFloat32(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function rangeFloat32(length) {
  return Float32Array.from({ length: Math.max(0, Number(length) | 0) }, (_, i) => i);
}

function resolveAxisName(code) {
  for (const [name, value] of Object.entries(AXES)) {
    if (value === (Number(code) | 0)) return name;
  }
  return "x1";
}

function resolveScaleName(code) {
  for (const [name, value] of Object.entries(AXIS_SCALES)) {
    if (value === (Number(code) | 0)) return name;
  }
  return "linear";
}

function resolveMarkerName(code) {
  for (const [name, value] of Object.entries(MARKERS)) {
    if (value === (Number(code) | 0)) return name;
  }
  return "none";
}

function csvValue(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function normalizeMatrix(value, options = {}, name = "data") {
  if (Array.isArray(value) && Array.isArray(value[0])) {
    const rows = value.length;
    const cols = rows > 0 ? value[0].length : 0;
    if (rows <= 0 || cols <= 0) {
      throw new Error(`${name} must not be empty.`);
    }
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r += 1) {
      if (!Array.isArray(value[r]) || value[r].length !== cols) {
        throw new Error(`${name} rows must have a consistent length.`);
      }
      out.set(value[r], r * cols);
    }
    return { data: out, rows, cols };
  }

  const data = ensureVector(value, name);
  const rows = Math.max(0, Number(options.rows ?? 0) | 0);
  const cols = Math.max(0, Number(options.cols ?? 0) | 0);
  if (rows <= 0 || cols <= 0 || rows * cols !== data.length) {
    throw new Error(`${name} flat arrays require rows and cols with rows * cols === data.length.`);
  }
  return { data, rows, cols };
}

function normalizeImage(value, options = {}) {
  const rows = Math.max(0, Number(options.rows ?? 0) | 0);
  const cols = Math.max(0, Number(options.cols ?? 0) | 0);
  const channels = Math.max(1, Number(options.channels ?? 1) | 0);
  const data = ensureVector(value, "image data");
  if (![1, 3, 4].includes(channels)) {
    throw new Error("image channels must be 1, 3, or 4.");
  }
  if (rows <= 0 || cols <= 0 || rows * cols * channels !== data.length) {
    throw new Error("image data requires rows, cols, and channels with rows * cols * channels === data.length.");
  }
  return { data, rows, cols, channels };
}

function axisCode(axis) {
  const key = String(axis || "").toLowerCase();
  if (!(key in AXES)) {
    throw new Error("axis must be one of x1, x2, x3, y1, y2, y3.");
  }
  return AXES[key];
}

function axesCodes(xAxis = "x1", yAxis = "y1") {
  const x = axisCode(xAxis);
  const y = axisCode(yAxis);
  if (x > 2 || y < 3) {
    throw new Error("xAxis must be x1/x2/x3 and yAxis must be y1/y2/y3.");
  }
  return [x, y];
}

function scaleCode(scale) {
  const key = String(scale || "linear").toLowerCase();
  if (!(key in AXIS_SCALES)) {
    throw new Error("axis scale must be linear, log, or time.");
  }
  return AXIS_SCALES[key];
}

function markerCode(marker) {
  const key = String(marker || "none").toLowerCase();
  if (!(key in MARKERS)) {
    throw new Error(`marker must be one of: ${Object.keys(MARKERS).join(", ")}.`);
  }
  return MARKERS[key];
}

function normalizeColor(color) {
  if (color == null || color === "") {
    return null;
  }
  if (Array.isArray(color) || ArrayBuffer.isView(color)) {
    const vals = Array.from(color, Number);
    if (vals.length === 3) vals.push(1);
    if (vals.length !== 4 || vals.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
      throw new Error("color arrays must contain 3 or 4 finite values in [0, 1].");
    }
    return vals;
  }
  const text = String(color).trim();
  if (!text.startsWith("#")) {
    throw new Error("color strings must be hex values like #3b82f6 or #3b82f680.");
  }
  let hex = text.slice(1);
  if (hex.length === 3) {
    hex = hex.split("").map((c) => c + c).join("") + "ff";
  } else if (hex.length === 4) {
    hex = hex.split("").map((c) => c + c).join("");
  } else if (hex.length === 6) {
    hex += "ff";
  } else if (hex.length !== 8) {
    throw new Error("color hex must be #RGB, #RGBA, #RRGGBB, or #RRGGBBAA.");
  }
  const vals = [0, 2, 4, 6].map((i) => Number.parseInt(hex.slice(i, i + 2), 16) / 255);
  if (vals.some((v) => !Number.isFinite(v))) {
    throw new Error("color hex contains invalid characters.");
  }
  return vals;
}

function domButtonToImGuiButton(button) {
  const b = button | 0;
  if (b === 0) return 0;
  if (b === 2) return 1;
  if (b === 1) return 2;
  if (b === 3) return 3;
  if (b === 4) return 4;
  return -1;
}

function plotFlagsFromOptions(options = {}) {
  if (Number.isFinite(Number(options.flags))) return Number(options.flags) | 0;
  let flags = 0;
  if (options.noLegend) flags |= PLOT_FLAGS.NO_LEGEND;
  if (options.noMenus) flags |= PLOT_FLAGS.NO_MENUS;
  if (options.noBoxSelect) flags |= PLOT_FLAGS.NO_BOX_SELECT;
  if (options.noMousePos) flags |= PLOT_FLAGS.NO_MOUSE_POS;
  if (options.crosshairs) flags |= PLOT_FLAGS.CROSSHAIRS;
  if (options.equal) flags |= PLOT_FLAGS.EQUAL;
  return flags;
}

function subplotFlagsFromOptions(options = {}) {
  if (Number.isFinite(Number(options.flags))) return Number(options.flags) | 0;
  let flags = 0;
  if (options.noLegend) flags |= SUBPLOT_FLAGS.NO_LEGEND;
  if (options.noMenus) flags |= SUBPLOT_FLAGS.NO_MENUS;
  if (options.noResize) flags |= SUBPLOT_FLAGS.NO_RESIZE;
  if (options.noAlign) flags |= SUBPLOT_FLAGS.NO_ALIGN;
  if (options.shareItems) flags |= SUBPLOT_FLAGS.SHARE_ITEMS;
  if (options.linkRows) flags |= SUBPLOT_FLAGS.LINK_ROWS;
  if (options.linkCols) flags |= SUBPLOT_FLAGS.LINK_COLS;
  if (options.linkAllX) flags |= SUBPLOT_FLAGS.LINK_ALL_X;
  if (options.linkAllY) flags |= SUBPLOT_FLAGS.LINK_ALL_Y;
  if (options.colMajor) flags |= SUBPLOT_FLAGS.COL_MAJOR;
  return flags;
}

function minMax(values) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = Number(values[i]);
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return { min: 0, max: 1 };
  }
  return { min, max };
}

function histogram1d(values, bins) {
  const nBins = Math.max(1, Number(bins) | 0);
  const { min, max } = minMax(values);
  const width = (max - min) / nBins;
  const edges = new Float32Array(nBins + 1);
  const counts = new Float32Array(nBins);
  for (let i = 0; i <= nBins; i += 1) {
    edges[i] = min + width * i;
  }
  for (let i = 0; i < values.length; i += 1) {
    const v = Number(values[i]);
    if (!Number.isFinite(v)) continue;
    let idx = Math.floor((v - min) / width);
    if (idx === nBins) idx = nBins - 1;
    if (idx >= 0 && idx < nBins) counts[idx] += 1;
  }
  return { edges, counts };
}

function histogram2d(x, y, xBins, yBins) {
  const xb = Math.max(1, Number(xBins) | 0);
  const yb = Math.max(1, Number(yBins) | 0);
  const xbnd = minMax(x);
  const ybnd = minMax(y);
  const xw = (xbnd.max - xbnd.min) / xb;
  const yw = (ybnd.max - ybnd.min) / yb;
  const xEdges = new Float32Array(xb + 1);
  const yEdges = new Float32Array(yb + 1);
  const counts = new Float32Array(xb * yb);
  for (let i = 0; i <= xb; i += 1) xEdges[i] = xbnd.min + xw * i;
  for (let i = 0; i <= yb; i += 1) yEdges[i] = ybnd.min + yw * i;
  for (let i = 0; i < x.length; i += 1) {
    const xv = Number(x[i]);
    const yv = Number(y[i]);
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
    let xi = Math.floor((xv - xbnd.min) / xw);
    let yi = Math.floor((yv - ybnd.min) / yw);
    if (xi === xb) xi = xb - 1;
    if (yi === yb) yi = yb - 1;
    if (xi >= 0 && xi < xb && yi >= 0 && yi < yb) {
      counts[xi * yb + yi] += 1;
    }
  }
  return { xEdges, yEdges, counts, rows: xb, cols: yb };
}

export function probeWebGL2() {
  try {
    const canvas = document.createElement("canvas");
    const webgl2 = canvas.getContext("webgl2");
    if (webgl2) {
      return { available: true, reason: "" };
    }
    const webgl1 = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (webgl1) {
      return { available: false, reason: "WebGL1 is available, but WebGL2 is required." };
    }
    return {
      available: false,
      reason: "Browser could not create a WebGL context.",
    };
  } catch (error) {
    return {
      available: false,
      reason: `WebGL2 probe failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

export async function loadNbImPlotModule(options = {}) {
  const wasmUrl = options.wasmUrl || DEFAULT_WASM_URL;
  let wasmBinary = options.wasmBinary;
  if (!(wasmBinary instanceof Uint8Array)) {
    const response = await fetch(wasmUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch nbimplot WASM binary: ${response.status} ${response.statusText}`);
    }
    wasmBinary = new Uint8Array(await response.arrayBuffer());
  }
  const assetKey = `${String(wasmUrl)}:${wasmBinary.byteLength}`;
  if (modulePromise && moduleAssetKey === assetKey) {
    return modulePromise;
  }
  moduleAssetKey = assetKey;
  modulePromise = createNbImPlotModule({
    wasmBinary,
    locateFile: () => String(wasmUrl),
  });
  return modulePromise;
}

class WasmCoreSession {
  constructor(module) {
    this.module = module;
    this.handle = 0;
    this.ready = false;
    this.lastError = "";
    this.encoder = new TextEncoder();
    this.perfPtr = 0;
  }

  init() {
    this.handle = this.module._nbp_create();
    this.ready = this.handle !== 0;
    if (!this.ready) {
      this.lastError = "WASM module loaded but _nbp_create returned 0.";
      return false;
    }
    this.perfPtr = this.module._malloc(32);
    return true;
  }

  isReady() {
    return this.ready && this.handle !== 0;
  }

  destroy() {
    if (!this.isReady()) return;
    if (this.perfPtr) {
      this.module._free(this.perfPtr);
      this.perfPtr = 0;
    }
    this.module._nbp_destroy(this.handle);
    this.handle = 0;
    this.ready = false;
  }

  withCString(text, fn) {
    const encoded = this.encoder.encode(`${String(text || "")}\0`);
    const ptr = this.module._malloc(encoded.byteLength);
    if (ptr === 0) return false;
    this.module.HEAPU8.set(encoded, ptr);
    try {
      return fn(ptr);
    } finally {
      this.module._free(ptr);
    }
  }

  setCanvas(width, height, dpr) {
    return (
      this.module._nbp_set_canvas(
        this.handle,
        Math.max(1, width | 0),
        Math.max(1, height | 0),
        Math.max(1, Number(dpr)),
      ) === 0
    );
  }

  setCanvasSelector(selector) {
    return this.withCString(selector, (ptr) => this.module._nbp_set_canvas_selector(this.handle, ptr) === 0);
  }

  upsertLine(token, data, isNewSeries) {
    const view = ensureVector(data, "line data");
    const ptr = this.module._malloc(view.byteLength);
    if (ptr === 0) return false;
    this.module.HEAPF32.set(view, ptr >>> 2);
    const rc = this.module._nbp_line_set_data(
      this.handle,
      token >>> 0,
      ptr,
      view.length >>> 0,
      isNewSeries ? 1 : 0,
    );
    this.module._free(ptr);
    return rc === 0;
  }

  upsertLineXY(token, xData, yData, isNewSeries) {
    if (typeof this.module._nbp_line_set_data_xy !== "function") return false;
    const xView = ensureVector(xData, "x");
    const yView = ensureVector(yData, "y");
    if (xView.length !== yView.length) return false;
    const bytes = yView.byteLength;
    const ptr = this.module._malloc(bytes * 2);
    if (ptr === 0) return false;
    const xPtr = ptr;
    const yPtr = ptr + bytes;
    this.module.HEAPF32.set(xView, xPtr >>> 2);
    this.module.HEAPF32.set(yView, yPtr >>> 2);
    const rc = this.module._nbp_line_set_data_xy(
      this.handle,
      token >>> 0,
      xPtr,
      yPtr,
      yView.length >>> 0,
      isNewSeries ? 1 : 0,
    );
    this.module._free(ptr);
    return rc === 0;
  }

  appendLineData(token, data, maxPoints) {
    const view = ensureVector(data, "append data");
    const ptr = this.module._malloc(view.byteLength);
    if (ptr === 0) return false;
    this.module.HEAPF32.set(view, ptr >>> 2);
    const rc = this.module._nbp_line_append_data(
      this.handle,
      token >>> 0,
      ptr,
      view.length >>> 0,
      Math.max(0, Number(maxPoints || 0) | 0),
    );
    this.module._free(ptr);
    return rc === 0;
  }

  setSeriesName(token, name) {
    return this.withCString(name, (ptr) => this.module._nbp_line_set_name(this.handle, token >>> 0, ptr) === 0);
  }

  setSeriesVisible(token, visible) {
    return this.module._nbp_set_series_visible(this.handle, token >>> 0, visible ? 1 : 0) === 0;
  }

  setSeriesSubplot(token, subplotIndex) {
    return this.module._nbp_set_series_subplot(this.handle, token >>> 0, Math.max(0, Number(subplotIndex) | 0)) === 0;
  }

  setSeriesAxes(token, xAxis, yAxis) {
    return this.module._nbp_set_series_axes(this.handle, token >>> 0, xAxis | 0, yAxis | 0) === 0;
  }

  setSeriesStyle(token, style = {}) {
    const color = normalizeColor(style.color);
    return (
      this.module._nbp_set_series_style(
        this.handle,
        token >>> 0,
        color ? 1 : 0,
        color ? color[0] : 0,
        color ? color[1] : 0,
        color ? color[2] : 0,
        color ? color[3] : 0,
        Number(style.lineWeight ?? 1),
        markerCode(style.marker ?? "none"),
        Number(style.markerSize ?? 4),
      ) === 0
    );
  }

  setPrimitiveVisible(token, visible) {
    return this.module._nbp_primitive_set_visible(this.handle, token >>> 0, visible ? 1 : 0) === 0;
  }

  removePrimitive(token) {
    return this.module._nbp_primitive_remove(this.handle, token >>> 0) === 0;
  }

  upsertPrimitive(token, kind, payload) {
    const alloc = (view) => {
      if (!(view instanceof Float32Array) || view.length === 0) return { ptr: 0, len: 0 };
      const ptr = this.module._malloc(view.byteLength);
      if (ptr === 0) return null;
      this.module.HEAPF32.set(view, ptr >>> 2);
      return { ptr, len: view.length >>> 0 };
    };

    const b0 = alloc(payload.data0);
    if (b0 == null) return false;
    const b1 = alloc(payload.data1);
    if (b1 == null) {
      if (b0.ptr) this.module._free(b0.ptr);
      return false;
    }
    const b2 = alloc(payload.data2);
    if (b2 == null) {
      if (b1.ptr) this.module._free(b1.ptr);
      if (b0.ptr) this.module._free(b0.ptr);
      return false;
    }

    const ints = Array.from({ length: 8 }, (_, i) => Number(payload.ints?.[i] ?? 0) | 0);
    const floats = Array.from({ length: 8 }, (_, i) => {
      const value = Number(payload.floats?.[i] ?? 0);
      return Number.isFinite(value) ? value : Number.NaN;
    });

    try {
      return this.withCString(payload.text || "", (textPtr) => {
        const rc = this.module._nbp_primitive_set_data(
          this.handle,
          token >>> 0,
          kind | 0,
          b0.ptr,
          b0.len,
          b1.ptr,
          b1.len,
          b2.ptr,
          b2.len,
          ints[0],
          ints[1],
          ints[2],
          ints[3],
          ints[4],
          ints[5],
          ints[6],
          ints[7],
          floats[0],
          floats[1],
          floats[2],
          floats[3],
          floats[4],
          floats[5],
          floats[6],
          floats[7],
          textPtr,
        );
        return rc === 0;
      });
    } finally {
      if (b2.ptr) this.module._free(b2.ptr);
      if (b1.ptr) this.module._free(b1.ptr);
      if (b0.ptr) this.module._free(b0.ptr);
    }
  }

  setPlotOptions(flags, axisScaleX, axisScaleY) {
    return this.module._nbp_set_plot_options(this.handle, flags | 0, axisScaleX | 0, axisScaleY | 0) === 0;
  }

  setAxisState(axis, enabled, scale) {
    return this.module._nbp_set_axis_state(this.handle, axis | 0, enabled ? 1 : 0, scale | 0) === 0;
  }

  setAxisLabel(axis, label) {
    return this.withCString(label || "", (ptr) => this.module._nbp_set_axis_label(this.handle, axis | 0, ptr) === 0);
  }

  setAxisFormat(axis, format) {
    return this.withCString(format || "", (ptr) => this.module._nbp_set_axis_format(this.handle, axis | 0, ptr) === 0);
  }

  setAxisTicks(axis, values, labels, keepDefault) {
    const ticks = values ? ensureVector(values, "tick values") : new Float32Array(0);
    let ptr = 0;
    if (ticks.length > 0) {
      ptr = this.module._malloc(ticks.byteLength);
      if (ptr === 0) return false;
      this.module.HEAPF32.set(ticks, ptr >>> 2);
    }
    const labelBlob = Array.isArray(labels) ? labels.map(String).join(LABEL_SEP) : "";
    try {
      return this.withCString(labelBlob, (labelPtr) => (
        this.module._nbp_set_axis_ticks(this.handle, axis | 0, ptr, ticks.length >>> 0, labelPtr, keepDefault ? 1 : 0) === 0
      ));
    } finally {
      if (ptr) this.module._free(ptr);
    }
  }

  clearAxisTicks(axis) {
    return this.module._nbp_clear_axis_ticks(this.handle, axis | 0) === 0;
  }

  setAxisLimitsConstraints(axis, enabled, minValue, maxValue) {
    return this.module._nbp_set_axis_limits_constraints(
      this.handle,
      axis | 0,
      enabled ? 1 : 0,
      Number(minValue ?? 0),
      Number(maxValue ?? 0),
    ) === 0;
  }

  setAxisZoomConstraints(axis, enabled, minValue, maxValue) {
    return this.module._nbp_set_axis_zoom_constraints(
      this.handle,
      axis | 0,
      enabled ? 1 : 0,
      Number(minValue ?? 0),
      Number(maxValue ?? 0),
    ) === 0;
  }

  setAxisLink(axis, targetAxis) {
    return this.module._nbp_set_axis_link(this.handle, axis | 0, targetAxis == null ? -1 : targetAxis | 0) === 0;
  }

  setSubplots(rows, cols, flags) {
    return this.module._nbp_set_subplots(this.handle, Math.max(1, rows | 0), Math.max(1, cols | 0), flags | 0) === 0;
  }

  setAlignedGroup(groupId, enabled, vertical) {
    return this.withCString(groupId || "", (ptr) => (
      this.module._nbp_set_aligned_group(this.handle, ptr, enabled ? 1 : 0, vertical ? 1 : 0) === 0
    ));
  }

  setColormap(name) {
    return this.withCString(name || "", (ptr) => this.module._nbp_set_colormap(this.handle, ptr) === 0);
  }

  setView(view) {
    this.module._nbp_set_view(this.handle, view.xMin, view.xMax, view.yMin, view.yMax);
    return true;
  }

  getView() {
    const ptr = this.module._malloc(16);
    if (ptr === 0) return null;
    this.module._nbp_get_view(this.handle, ptr);
    const base = ptr >>> 2;
    const out = {
      xMin: this.module.HEAPF32[base],
      xMax: this.module.HEAPF32[base + 1],
      yMin: this.module.HEAPF32[base + 2],
      yMax: this.module.HEAPF32[base + 3],
    };
    this.module._free(ptr);
    return out;
  }

  autoscale() {
    this.module._nbp_autoscale(this.handle);
    return this.getView();
  }

  setMousePos(x, y, inside) {
    return this.module._nbp_set_mouse_pos(this.handle, Number(x), Number(y), inside ? 1 : 0) === 0;
  }

  setMouseButton(button, down) {
    return this.module._nbp_set_mouse_button(this.handle, button | 0, down ? 1 : 0) === 0;
  }

  addMouseWheel(wheelX, wheelY) {
    return this.module._nbp_add_mouse_wheel(this.handle, Number(wheelX), Number(wheelY)) === 0;
  }

  isImPlotCompiled() {
    return this.module._nbp_is_implot_compiled() === 1;
  }

  setImPlotEnabled(enabled) {
    this.module._nbp_set_implot_enabled(this.handle, enabled ? 1 : 0);
    return this.module._nbp_is_implot_enabled(this.handle) === 1;
  }

  render(title) {
    return this.withCString(title || "", (ptr) => this.module._nbp_render(this.handle, ptr) === 0);
  }

  getPerfStats() {
    if (!this.perfPtr) return null;
    if (this.module._nbp_get_perf_stats(this.handle, this.perfPtr) !== 0) return null;
    const base = this.perfPtr >>> 2;
    return {
      lodMs: this.module.HEAPF32[base],
      segmentBuildMs: this.module.HEAPF32[base + 1],
      renderMs: this.module.HEAPF32[base + 2],
      frameMs: this.module.HEAPF32[base + 3],
      drawPoints: this.module.HEAPF32[base + 4],
      drawSegments: this.module.HEAPF32[base + 5],
      primitiveCount: this.module.HEAPF32[base + 6],
      pixelWidth: this.module.HEAPF32[base + 7],
    };
  }

  getInteractions() {
    const len = this.module._nbp_get_interaction_len(this.handle) >>> 0;
    if (len === 0) return new Float32Array(0);
    const ptr = this.module._nbp_get_interaction_ptr(this.handle);
    if (!ptr) return new Float32Array(0);
    return this.module.HEAPF32.subarray(ptr >>> 2, (ptr >>> 2) + len * 8);
  }
}

class LineHandle {
  constructor(plot, token, options = {}) {
    this.plot = plot;
    this.token = token >>> 0;
    this.capacity = Math.max(0, Number(options.capacity || 0) | 0);
    this.xData = options.xData || null;
    this.record = options.record || null;
    this.paused = Boolean(options.paused);
    this.autoRender = Boolean(options.autoRender ?? options.auto_render);
    this.autoscaleY = Boolean(options.autoscaleY ?? options.autoscale_y);
  }

  setData(y, options = {}) {
    this.plot._assertReady();
    const data = ensureVector(y, "y");
    let xData = null;
    if (options.x != null) {
      xData = ensureLineX(options.x, data.length);
    } else if (this.xData) {
      if (data.length !== this.xData.length) {
        throw new Error("x must be provided when resizing a custom-x line.");
      }
      xData = this.xData;
    }

    const upload = this.capacity > 0 && data.length > this.capacity ? data.subarray(data.length - this.capacity) : data;
    const uploadX =
      xData && this.capacity > 0 && xData.length > this.capacity ? xData.subarray(xData.length - this.capacity) : xData;
    const ok = uploadX
      ? this.plot.wasm.upsertLineXY(this.token, uploadX, upload, false)
      : this.plot.wasm.upsertLine(this.token, upload, false);
    if (!ok) {
      throw new Error("Failed to update line data.");
    }
    this.xData = uploadX || null;
    if (this.record) {
      this.record.data = upload;
      this.record.xData = uploadX || null;
    }
    this.plot._afterDataChange();
    return this;
  }

  append(y, options = {}) {
    this.plot._assertReady();
    if (this.paused) return this;
    const appended = ensureVector(y, "y");
    const appendX = options.x != null ? ensureLineX(options.x, appended.length) : null;
    if (this.xData && !appendX) {
      throw new Error("x must be provided when appending to a custom-x line.");
    }
    if (appendX) {
      const existingX = this.xData || rangeFloat32(this.record?.data?.length || 0);
      if (existingX.length > 0 && appendX.length > 0 && appendX[0] < existingX[existingX.length - 1]) {
        throw new Error("appended x values must continue the non-decreasing x order.");
      }
      let mergedY = concatFloat32(this.record?.data || new Float32Array(0), appended);
      let mergedX = concatFloat32(existingX, appendX);
      const maxPoints = Math.max(0, Number(options.maxPoints ?? options.max_points ?? this.capacity) | 0);
      if (maxPoints > 0) this.capacity = maxPoints;
      if (maxPoints > 0 && mergedY.length > maxPoints) {
        mergedY = mergedY.slice(mergedY.length - maxPoints);
        mergedX = mergedX.slice(mergedX.length - maxPoints);
      }
      this.setData(mergedY, { x: mergedX });
      this._afterAppend();
      return this;
    }
    const maxPoints = Math.max(0, Number(options.maxPoints ?? options.max_points ?? this.capacity) | 0);
    if (maxPoints > 0) this.capacity = maxPoints;
    if (!this.plot.wasm.appendLineData(this.token, appended, this.capacity)) {
      throw new Error("Failed to append line data.");
    }
    if (this.record) {
      let merged = concatFloat32(this.record.data, appended);
      if (this.capacity > 0 && merged.length > this.capacity) {
        merged = merged.slice(merged.length - this.capacity);
      }
      this.record.data = merged;
      this.record.xData = null;
    }
    this.plot._afterDataChange();
    this._afterAppend();
    return this;
  }

  _afterAppend() {
    if (this.record) {
      this.record.streamPaused = this.paused;
      this.record.streamAutoRender = this.autoRender;
      this.record.streamAutoscaleY = this.autoscaleY;
      this.record.capacity = this.capacity;
    }
    if (this.autoscaleY) {
      this.plot.autoscale();
    } else if (this.autoRender) {
      this.plot.requestRender();
    }
  }

  pause() {
    this.paused = true;
    if (this.record) this.record.streamPaused = true;
    return this;
  }

  resume() {
    this.paused = false;
    if (this.record) this.record.streamPaused = false;
    return this;
  }

  clear(options = {}) {
    const y0 = Number(options.y0 ?? 0);
    const x0 = Number(options.x0 ?? 0);
    return this.setData(new Float32Array([y0]), this.xData ? { x: new Float32Array([x0]) } : {});
  }

  setWindow(capacity) {
    const cap = Math.max(1, Number(capacity) | 0);
    this.capacity = cap;
    if (this.record) {
      this.record.capacity = cap;
      if (this.record.data.length > cap) {
        const y = this.record.data.slice(this.record.data.length - cap);
        const x = this.record.xData ? this.record.xData.slice(this.record.xData.length - cap) : null;
        this.setData(y, x ? { x } : {});
      }
    }
    return this;
  }

  setStreamOptions(options = {}) {
    if (options.autoRender != null || options.auto_render != null) {
      this.autoRender = Boolean(options.autoRender ?? options.auto_render);
    }
    if (options.autoscaleY != null || options.autoscale_y != null) {
      this.autoscaleY = Boolean(options.autoscaleY ?? options.autoscale_y);
    }
    if (this.record) {
      this.record.streamAutoRender = this.autoRender;
      this.record.streamAutoscaleY = this.autoscaleY;
    }
    return this;
  }

  setVisible(visible) {
    this.plot._assertReady();
    this.plot.wasm.setSeriesVisible(this.token, Boolean(visible));
    if (this.record) {
      this.record.visible = Boolean(visible);
    }
    this.plot.requestRender();
    return this;
  }

  setStyle(style = {}) {
    this.plot._assertReady();
    this.plot.wasm.setSeriesStyle(this.token, style);
    if (this.record) {
      this.record.style = { ...this.record.style, ...style };
    }
    this.plot.requestRender();
    return this;
  }
}

class PrimitiveHandle {
  constructor(plot, token, record = null) {
    this.plot = plot;
    this.token = token >>> 0;
    this.record = record;
  }

  setVisible(visible) {
    this.plot._assertReady();
    this.plot.wasm.setPrimitiveVisible(this.token, Boolean(visible));
    if (this.record) this.record.visible = Boolean(visible);
    this.plot.requestRender();
    return this;
  }

  remove() {
    this.plot._assertReady();
    this.plot.wasm.removePrimitive(this.token);
    this.plot.primitiveRecords.delete(this.token);
    this.plot.requestRender();
  }
}

export class WebPlot {
  constructor(target, options = {}) {
    this.target = resolveElement(target);
    this.options = { ...options };
    this.title = String(options.title || "");
    this.width = Math.max(120, Number(options.width || 900));
    this.height = Math.max(100, Number(options.height || 450));
    this.responsive = Boolean(options.responsive);
    this.initialAutoFitActive = options.autoFit !== false;
    this.autoFitOnDataChange = Boolean(options.autoFitOnDataChange);
    this.autoRender = options.autoRender !== false;
    this.disposed = false;
    this.ready = false;
    this.dirty = false;
    this.rafId = 0;
    this.nextSeriesToken = 1;
    this.nextPrimitiveToken = 1;
    this.hasRenderableData = false;
    this.view = null;
    this.viewCallbacks = new Set();
    this.interactionCallbacks = new Set();
    this.hoverCallbacks = new Set();
    this.clickCallbacks = new Set();
    this.selectionCallbacks = new Set();
    this.perfCallbacks = new Set();
    this.seriesByToken = new Map();
    this.primitiveRecords = new Map();
    this.lastInteractionHash = "";
    this.plotFlags = plotFlagsFromOptions(options);
    this.axisScaleX = scaleCode(options.axisScaleX || "linear");
    this.axisScaleY = scaleCode(options.axisScaleY || "linear");
    this.axisState = Array.from({ length: 6 }, (_, axis) => ({
      enabled: axis === AXES.x1 || axis === AXES.y1,
      scale: axis === AXES.x1 ? this.axisScaleX : axis === AXES.y1 ? this.axisScaleY : AXIS_SCALES.linear,
    }));
    this.axisLabels = new Array(6).fill("");
    this.axisFormats = new Array(6).fill("");
    this.axisTicks = new Map();
    this.axisLimitsConstraints = new Map();
    this.axisZoomConstraints = new Map();
    this.axisLinks = new Map();
    this.subplotRows = Math.max(1, Number(options.subplotRows || 1) | 0);
    this.subplotCols = Math.max(1, Number(options.subplotCols || 1) | 0);
    this.subplotFlags = subplotFlagsFromOptions(options);
    this.colormapName = options.colormap ? String(options.colormap) : "";
    this.themeName = options.theme ? String(options.theme) : "";
    this.alignedGroup = null;
    this.linkedCrosshair = { enabled: false, groupId: "", axis: "x" };
    this.linkedCrosshairTokenBase = 0x70000000 + ((Math.random() * 0x0fffffff) >>> 0);
    this.linkedCrosshairVisible = false;
    this._buildDom(options);
  }

  async init() {
    const probe = probeWebGL2();
    if (!probe.available) {
      throw new Error(`nbimplot requires WebGL2. ${probe.reason}`);
    }
    const module = await loadNbImPlotModule(this.options);
    this.wasm = new WasmCoreSession(module);
    if (!this.wasm.init()) {
      throw new Error(this.wasm.lastError || "Failed to initialize nbimplot WASM core.");
    }
    this._resize();
    if (!this.wasm.setCanvasSelector(`#${this.canvas.id}`)) {
      throw new Error("Failed to bind nbimplot WASM core to canvas.");
    }
    if (!this.wasm.isImPlotCompiled()) {
      throw new Error("WASM module was built without ImPlot.");
    }
    if (!this.wasm.setImPlotEnabled(true)) {
      throw new Error("Unable to enable ImPlot in the WASM core.");
    }
    this._syncOptions();
    this._bindEvents();
    this.ready = true;
    this.requestRender();
    return this;
  }

  _buildDom(options) {
    this.wrapper = document.createElement("div");
    this.wrapper.className = "nbimplot-web";
    this.wrapper.dataset.nbimplotTheme = this.themeName || "nbimplot";
    this.wrapper.style.position = "relative";
    this.wrapper.style.width = `${this.width}px`;
    this.wrapper.style.height = `${this.height}px`;
    this.wrapper.style.minWidth = "120px";
    this.wrapper.style.minHeight = "100px";

    this.canvas = document.createElement("canvas");
    this.canvas.id = options.canvasId || `nbimplot-web-${Math.random().toString(36).slice(2)}`;
    this.canvas.tabIndex = 0;
    this.canvas.style.display = "block";
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    this.canvas.style.touchAction = "none";
    this.canvas.style.outline = "none";

    this.wrapper.appendChild(this.canvas);
    if (options.replace === false) {
      this.target.appendChild(this.wrapper);
    } else {
      this.target.replaceChildren(this.wrapper);
    }

    this.resizeObserver = typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(() => this._resize())
      : null;
    if (this.resizeObserver) {
      this.resizeObserver.observe(this.wrapper);
    }
  }

  _bindEvents() {
    this.onMouseMove = (event) => {
      const pos = this._pointerPosition(event);
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      this.requestRender();
    };
    this.onMouseDown = (event) => {
      this.canvas.focus();
      const pos = this._pointerPosition(event);
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      const button = domButtonToImGuiButton(event.button);
      if (button >= 0) this.wasm.setMouseButton(button, true);
      if (event.button === 2) event.preventDefault();
      this.requestRender();
    };
    this.onMouseUp = (event) => {
      const pos = this._pointerPosition(event);
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      const button = domButtonToImGuiButton(event.button);
      if (button >= 0) this.wasm.setMouseButton(button, false);
      if (event.button === 2) event.preventDefault();
      this.requestRender();
    };
    this.onMouseLeave = () => {
      this.wasm.setMousePos(0, 0, false);
      this.requestRender();
    };
    this.onWheel = (event) => {
      event.preventDefault();
      const pos = this._pointerPosition(event);
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      const scale = event.deltaMode === 1 ? 1.0 : event.deltaMode === 2 ? 12.0 : 0.01;
      this.wasm.addMouseWheel(-event.deltaX * scale, -event.deltaY * scale);
      this.requestRender();
    };
    this.onDoubleClick = (event) => {
      event.preventDefault();
      this.autoscale();
    };
    this.onContextMenu = (event) => {
      event.preventDefault();
      const pos = this._pointerPosition(event);
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      this.requestRender();
    };
    this.onWindowResize = () => this._resize();

    this.canvas.addEventListener("mousemove", this.onMouseMove);
    this.canvas.addEventListener("mousedown", this.onMouseDown);
    window.addEventListener("mouseup", this.onMouseUp);
    this.canvas.addEventListener("mouseleave", this.onMouseLeave);
    this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
    this.canvas.addEventListener("dblclick", this.onDoubleClick);
    this.canvas.addEventListener("contextmenu", this.onContextMenu);
    window.addEventListener("resize", this.onWindowResize);
  }

  _syncOptions() {
    this.wasm.setPlotOptions(this.plotFlags, this.axisScaleX, this.axisScaleY);
    this.wasm.setSubplots(this.subplotRows, this.subplotCols, this.subplotFlags);
    this.wasm.setColormap(this.colormapName);
    for (let axis = 0; axis < this.axisState.length; axis += 1) {
      const state = this.axisState[axis];
      this.wasm.setAxisState(axis, state.enabled, state.scale);
    }
    if (this.alignedGroup) {
      this.wasm.setAlignedGroup(
        this.alignedGroup.groupId,
        this.alignedGroup.enabled,
        this.alignedGroup.vertical,
      );
    }
  }

  _linkedCrosshairToken(subplotIndex, axis) {
    const offset = Math.max(0, Number(subplotIndex) | 0) * 2 + (axis === "y" ? 1 : 0);
    return (this.linkedCrosshairTokenBase + offset) >>> 0;
  }

  _syncLinkedCrosshairPrimitive(axis, value, subplotIndex) {
    if (!this.ready || !Number.isFinite(Number(value))) return;
    const isY = axis === "y";
    const ints = [0, 0, 0, 1, 0, 3, 0, Math.max(0, Number(subplotIndex) | 0)];
    const floats = [0, 0, 0, 0, 0, 0, 1.4, 0];
    if (isY) {
      floats[5] = Number(value);
    } else {
      floats[4] = Number(value);
    }
    this.wasm.upsertPrimitive(this._linkedCrosshairToken(subplotIndex, axis), PRIMITIVE_KIND_CODES[isY ? "drag_line_y" : "drag_line_x"], {
      data0: new Float32Array(0),
      data1: new Float32Array(0),
      data2: new Float32Array(0),
      ints,
      floats,
      text: isY ? "linked-y" : "linked-x",
    });
    this.wasm.setPrimitiveVisible(this._linkedCrosshairToken(subplotIndex, axis), true);
  }

  _receiveLinkedCrosshair(event) {
    if (!this.linkedCrosshair.enabled || !event || !this.ready) {
      this._hideLinkedCrosshair();
      return;
    }
    const count = Math.max(1, this.subplotRows * this.subplotCols);
    const wantsX = this.linkedCrosshair.axis === "x" || this.linkedCrosshair.axis === "xy";
    const wantsY = this.linkedCrosshair.axis === "y" || this.linkedCrosshair.axis === "xy";
    for (let subplotIndex = 0; subplotIndex < count; subplotIndex += 1) {
      if (wantsX && Number.isFinite(event.x)) {
        this._syncLinkedCrosshairPrimitive("x", event.x, subplotIndex);
      } else {
        this.wasm.setPrimitiveVisible(this._linkedCrosshairToken(subplotIndex, "x"), false);
      }
      if (wantsY && Number.isFinite(event.y)) {
        this._syncLinkedCrosshairPrimitive("y", event.y, subplotIndex);
      } else {
        this.wasm.setPrimitiveVisible(this._linkedCrosshairToken(subplotIndex, "y"), false);
      }
    }
    this.linkedCrosshairVisible = true;
    this.requestRender();
  }

  _hideLinkedCrosshair() {
    if (!this.ready || !this.linkedCrosshairVisible) return;
    const count = Math.max(1, this.subplotRows * this.subplotCols);
    for (let subplotIndex = 0; subplotIndex < count; subplotIndex += 1) {
      this.wasm.setPrimitiveVisible(this._linkedCrosshairToken(subplotIndex, "x"), false);
      this.wasm.setPrimitiveVisible(this._linkedCrosshairToken(subplotIndex, "y"), false);
    }
    this.linkedCrosshairVisible = false;
    this.requestRender();
  }

  _broadcastLinkedCrosshair(hover) {
    if (!this.linkedCrosshair.enabled || !this.linkedCrosshair.groupId) return;
    const group = linkedCrosshairGroups.get(this.linkedCrosshair.groupId);
    if (!group) return;
    const event = hover && hover.active ? { x: Number(hover.x), y: Number(hover.y) } : null;
    for (const plot of group) {
      plot._receiveLinkedCrosshair(event);
    }
  }

  _assertReady() {
    if (!this.ready || this.disposed || !this.wasm?.isReady()) {
      throw new Error("nbimplot web plot is not ready.");
    }
  }

  _pointerPosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  _insideCanvas(pos) {
    return pos.x >= 0 && pos.x <= this.cssWidth && pos.y >= 0 && pos.y <= this.cssHeight;
  }

  _resize() {
    if (this.disposed) return;
    if (this.responsive) {
      this.cssWidth = Math.max(120, this.target.clientWidth || this.width);
      this.cssHeight = Math.max(100, this.target.clientHeight || this.height);
    } else {
      this.cssWidth = this.width;
      this.cssHeight = this.height;
    }
    this.dpr = Math.max(1, Number(this.options.devicePixelRatio || window.devicePixelRatio || 1));
    this.wrapper.style.width = `${this.cssWidth}px`;
    this.wrapper.style.height = `${this.cssHeight}px`;
    this.canvas.width = Math.max(1, Math.round(this.cssWidth * this.dpr));
    this.canvas.height = Math.max(1, Math.round(this.cssHeight * this.dpr));
    if (this.wasm?.isReady()) {
      this.wasm.setCanvas(this.cssWidth, this.cssHeight, this.dpr);
      this.requestRender();
    }
  }

  requestRender() {
    if (this.disposed || !this.autoRender) return;
    this.dirty = true;
    if (this.rafId === 0) {
      this.rafId = window.requestAnimationFrame(() => this._frame());
    }
  }

  _frame() {
    this.rafId = 0;
    if (this.disposed || !this.dirty) return;
    this.dirty = false;
    this.draw();
  }

  draw() {
    this._assertReady();
    const ok = this.wasm.render(this.title);
    if (!ok) {
      throw new Error("WASM draw pipeline failed. WebGL context may be unavailable or lost.");
    }
    const nextView = this.wasm.getView();
    if (nextView) {
      const changed = !this.view ||
        Math.abs(this.view.xMin - nextView.xMin) > 1e-9 ||
        Math.abs(this.view.xMax - nextView.xMax) > 1e-9 ||
        Math.abs(this.view.yMin - nextView.yMin) > 1e-9 ||
        Math.abs(this.view.yMax - nextView.yMax) > 1e-9;
      this.view = nextView;
      if (changed) this._emitViewChange();
    }
    this._emitInteractions();
    this._emitPerfStats();
    return this;
  }

  render() {
    return this.draw();
  }

  toDataURL(type = "image/png", quality) {
    this._assertReady();
    this.draw();
    return quality === undefined
      ? this.canvas.toDataURL(type)
      : this.canvas.toDataURL(type, quality);
  }

  toBlob(type = "image/png", quality) {
    this._assertReady();
    this.draw();
    if (typeof this.canvas.toBlob !== "function") {
      return Promise.resolve(dataUrlToBlob(this.canvas.toDataURL(type, quality)));
    }
    return new Promise((resolve, reject) => {
      const onBlob = (blob) => {
        if (blob) {
          resolve(blob);
          return;
        }
        try {
          resolve(dataUrlToBlob(this.canvas.toDataURL(type, quality)));
        } catch (error) {
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      };
      if (quality === undefined) {
        this.canvas.toBlob(onBlob, type);
      } else {
        this.canvas.toBlob(onBlob, type, quality);
      }
    });
  }

  async downloadPNG(filename = "nbimplot.png") {
    const blob = await this.toBlob("image/png");
    downloadBlob(blob, sanitizePngFilename(filename));
    return blob;
  }

  async copyPNGToClipboard() {
    if (!navigator.clipboard || typeof ClipboardItem === "undefined") {
      throw new Error("Clipboard PNG copy is not available in this browser.");
    }
    const blob = await this.toBlob("image/png");
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    return blob;
  }

  copy_png_to_clipboard() {
    return this.copyPNGToClipboard();
  }

  _emitViewChange() {
    for (const callback of this.viewCallbacks) {
      callback({ ...this.view }, this);
    }
  }

  _emitPerfStats() {
    if (this.perfCallbacks.size === 0) return;
    const stats = this.wasm.getPerfStats();
    if (!stats) return;
    for (const callback of this.perfCallbacks) {
      callback(stats, this);
    }
  }

  _emitInteractions() {
    if (
      this.interactionCallbacks.size === 0 &&
      this.hoverCallbacks.size === 0 &&
      this.clickCallbacks.size === 0 &&
      this.selectionCallbacks.size === 0 &&
      !this.linkedCrosshair.enabled
    ) {
      return;
    }
    const tuples = this.wasm.getInteractions();
    if (!(tuples instanceof Float32Array) || tuples.length === 0) {
      this.lastInteractionHash = "";
      this._broadcastLinkedCrosshair(null);
      return;
    }
    const payload = [];
    const selectionSeries = [];
    let selection = null;
    let hover = null;
    let click = null;
    for (let i = 0; i + 7 < tuples.length; i += 8) {
      const kind = tuples[i] | 0;
      const id = tuples[i + 1] | 0;
      const subplotIndex = tuples[i + 2] | 0;
      const active = (tuples[i + 3] | 0) !== 0;
      const v0 = Number(tuples[i + 4]);
      const v1 = Number(tuples[i + 5]);
      const v2 = Number(tuples[i + 6]);
      const v3 = Number(tuples[i + 7]);
      const record = this.seriesByToken.get(id) || null;
      const event = {
        kind,
        id,
        subplotIndex,
        active,
        v0,
        v1,
        v2,
        v3,
      };
      if (record) {
        event.seriesName = record.name;
        event.seriesToken = record.token;
      }
      payload.push(event);

      if (kind === 100) {
        selection = {
          subplotIndex,
          xMin: Math.min(v0, v1),
          xMax: Math.max(v0, v1),
          yMin: Math.min(v2, v3),
          yMax: Math.max(v2, v3),
          series: [],
        };
      } else if (kind === 101) {
        selectionSeries.push({
          seriesToken: id,
          seriesName: record ? record.name : "",
          subplotIndex,
          indexMin: Math.max(0, Math.round(Math.min(v0, v1))),
          indexMax: Math.max(0, Math.round(Math.max(v0, v1))),
          count: Math.max(0, Math.round(v2)),
          hasX: v3 !== 0,
        });
      } else if (kind === 102) {
        hover = {
          seriesToken: id,
          seriesName: record ? record.name : "",
          subplotIndex,
          active,
          x: v0,
          y: v1,
          index: Math.round(v2),
          distancePx: v3,
        };
      } else if (kind === 103) {
        click = {
          seriesToken: id,
          seriesName: record ? record.name : "",
          subplotIndex,
          active,
          x: v0,
          y: v1,
          button: Math.max(0, Math.round(v2)),
          index: Math.round(v3),
        };
      }
    }
    if (selection) {
      selection.series = selectionSeries.filter((entry) => entry.subplotIndex === selection.subplotIndex);
    }
    this._broadcastLinkedCrosshair(hover);
    const enriched = { payload, selection, hover, click };
    const hash = JSON.stringify(enriched);
    if (hash === this.lastInteractionHash) return;
    this.lastInteractionHash = hash;
    for (const callback of this.interactionCallbacks) {
      callback(payload, this);
    }
    if (selection) {
      for (const callback of this.selectionCallbacks) {
        callback(selection, this);
      }
    }
    if (hover) {
      for (const callback of this.hoverCallbacks) {
        callback(hover, this);
      }
    }
    if (click) {
      for (const callback of this.clickCallbacks) {
        callback(click, this);
      }
    }
  }

  _afterDataChange() {
    this.hasRenderableData = true;
    if (this.initialAutoFitActive || this.autoFitOnDataChange) {
      this.autoscale();
      this.initialAutoFitActive = false;
      return;
    }
    this.requestRender();
  }

  onViewChange(callback) {
    this.viewCallbacks.add(callback);
    return () => this.viewCallbacks.delete(callback);
  }

  onPerfStats(callback) {
    this.perfCallbacks.add(callback);
    return () => this.perfCallbacks.delete(callback);
  }

  onInteraction(callback) {
    this.interactionCallbacks.add(callback);
    return () => this.interactionCallbacks.delete(callback);
  }

  onHover(callback) {
    this.hoverCallbacks.add(callback);
    return () => this.hoverCallbacks.delete(callback);
  }

  onClick(callback) {
    this.clickCallbacks.add(callback);
    return () => this.clickCallbacks.delete(callback);
  }

  onSelection(callback) {
    this.selectionCallbacks.add(callback);
    return () => this.selectionCallbacks.delete(callback);
  }

  onSelect(callback) {
    return this.onSelection(callback);
  }

  _resolveSeriesRecord(series) {
    if (series == null) return null;
    const token = typeof series === "number" ? series : Number(series?.token ?? 0);
    if (token && this.seriesByToken.has(token)) return this.seriesByToken.get(token);
    const name = String(series?.name ?? series);
    for (const record of this.seriesByToken.values()) {
      if (record.name === name) return record;
    }
    throw new Error(`Unknown series: ${name}`);
  }

  selectionBounds(selection) {
    if (!selection || typeof selection !== "object") {
      throw new TypeError("selection must be the object passed to onSelection/onSelect.");
    }
    const xMin = Number(selection.xMin ?? selection.x_min);
    const xMax = Number(selection.xMax ?? selection.x_max);
    const yMin = Number(selection.yMin ?? selection.y_min);
    const yMax = Number(selection.yMax ?? selection.y_max);
    if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) {
      throw new Error("selection must contain finite x/y bounds.");
    }
    return {
      xMin: Math.min(xMin, xMax),
      xMax: Math.max(xMin, xMax),
      yMin: Math.min(yMin, yMax),
      yMax: Math.max(yMin, yMax),
    };
  }

  indicesForSelection(selection, series = null) {
    const bounds = this.selectionBounds(selection);
    const x0 = bounds.xMin;
    const x1 = bounds.xMax;
    const y0 = bounds.yMin;
    const y1 = bounds.yMax;

    const resolveRecords = () => {
      if (series == null) return Array.from(this.seriesByToken.values());
      return [this._resolveSeriesRecord(series)];
    };

    const exactForRecord = (record) => {
      const y = record.data || new Float32Array(0);
      const out = [];
      if (record.xData) {
        const x = record.xData;
        for (let i = 0; i < Math.min(x.length, y.length); i += 1) {
          const xv = x[i];
          const yv = y[i];
          if (Number.isFinite(xv) && Number.isFinite(yv) && xv >= x0 && xv <= x1 && yv >= y0 && yv <= y1) {
            out.push(i);
          }
        }
      } else {
        const start = Math.max(0, Math.ceil(x0));
        const stop = Math.min(y.length, Math.floor(x1) + 1);
        for (let i = start; i < stop; i += 1) {
          const yv = y[i];
          if (Number.isFinite(yv) && yv >= y0 && yv <= y1) {
            out.push(i);
          }
        }
      }
      return Uint32Array.from(out);
    };

    const records = resolveRecords();
    if (series != null) {
      return exactForRecord(records[0]);
    }
    const result = new Map();
    for (const record of records) {
      result.set(record.token, exactForRecord(record));
    }
    return result;
  }

  selectionIndices(selection, series = null) {
    return this.indicesForSelection(selection, series);
  }

  highlightSelection(selection, series = null, options = {}) {
    const bounds = this.selectionBounds(selection);
    const subplotIndex = Number(selection.subplotIndex ?? selection.subplot_index ?? 0) | 0;
    const name = String(options.name || "selection");
    if (options.rect !== false) {
      this.dragRect(name, bounds.xMin, bounds.yMin, bounds.xMax, bounds.yMax, { subplotIndex });
    }
    if (options.points === false) return this;
    const selected = this.indicesForSelection(selection, series);
    const entries = selected instanceof Map ? selected.entries() : [[this._resolveSeriesRecord(series).token, selected]];
    for (const [token, indices] of entries) {
      const record = this.seriesByToken.get(token);
      if (!record || indices.length === 0) continue;
      const x = new Float32Array(indices.length);
      const y = new Float32Array(indices.length);
      for (let i = 0; i < indices.length; i += 1) {
        const idx = indices[i];
        x[i] = record.xData ? record.xData[idx] : idx;
        y[i] = record.data[idx];
      }
      this.scatter(`${name}:${record.name}`, y, {
        x,
        size: options.size ?? 5,
        subplotIndex: record.subplotIndex,
        xAxis: resolveAxisName(record.xAxis),
        yAxis: resolveAxisName(record.yAxis),
      });
    }
    return this;
  }

  highlight_selection(selection, series = null, options = {}) {
    return this.highlightSelection(selection, series, options);
  }

  exportCSVSelection(selection, series = null, options = {}) {
    const selected = this.indicesForSelection(selection, series);
    const rows = [["series_token", "series_name", "index", "x", "y"]];
    const entries = selected instanceof Map ? selected.entries() : [[this._resolveSeriesRecord(series).token, selected]];
    for (const [token, indices] of entries) {
      const record = this.seriesByToken.get(token);
      if (!record) continue;
      for (const idxRaw of indices) {
        const idx = Number(idxRaw) | 0;
        rows.push([
          token,
          record.name,
          idx,
          record.xData ? record.xData[idx] : idx,
          record.data[idx],
        ]);
      }
    }
    const text = `${rows.map((row) => row.map(csvValue).join(",")).join("\n")}\n`;
    if (options.download || options.filename) {
      const filename = String(options.filename || "nbimplot-selection.csv").replace(/[\\/:*?"<>|]+/g, "_");
      downloadBlob(new Blob([text], { type: "text/csv;charset=utf-8" }), /\.csv$/i.test(filename) ? filename : `${filename}.csv`);
    }
    return text;
  }

  export_csv_selection(selection, series = null, options = {}) {
    return this.exportCSVSelection(selection, series, options);
  }

  line(name, y, options = {}) {
    this._assertReady();
    const token = this.nextSeriesToken++;
    const data = ensureVector(y, "y");
    const xData = options.x != null ? ensureLineX(options.x, data.length) : null;
    const [xAxis, yAxis] = axesCodes(options.xAxis || options.x_axis || "x1", options.yAxis || options.y_axis || "y1");
    const capacity = Math.max(0, Number(options.maxPoints || options.max_points || 0) | 0);
    const upload = capacity > 0 && data.length > capacity ? data.subarray(data.length - capacity) : data;
    const uploadX = xData && capacity > 0 && xData.length > capacity ? xData.subarray(xData.length - capacity) : xData;
    const style = {
      color: options.color,
      lineWeight: options.lineWeight ?? options.line_weight ?? 1,
      marker: options.marker ?? "none",
      markerSize: options.markerSize ?? options.marker_size ?? 4,
    };
    const ok = uploadX ? this.wasm.upsertLineXY(token, uploadX, upload, true) : this.wasm.upsertLine(token, upload, true);
    if (!ok) {
      throw new Error("Failed to upload line data.");
    }
    this.wasm.setSeriesName(token, name);
    this.wasm.setSeriesSubplot(token, options.subplotIndex ?? options.subplot_index ?? 0);
    this.wasm.setSeriesAxes(token, xAxis, yAxis);
    this.wasm.setSeriesStyle(token, style);
    if (options.visible === false || options.hidden === true) {
      this.wasm.setSeriesVisible(token, false);
    }
    const record = {
      token,
      name: String(name || ""),
      data: upload,
      xData: uploadX || null,
      subplotIndex: Number(options.subplotIndex ?? options.subplot_index ?? 0) | 0,
      xAxis,
      yAxis,
      visible: !(options.visible === false || options.hidden === true),
      capacity,
      style,
      streamPaused: false,
      streamAutoRender: Boolean(options.autoRender ?? options.auto_render),
      streamAutoscaleY: Boolean(options.autoscaleY ?? options.autoscale_y),
    };
    this.seriesByToken.set(token, record);
    this._afterDataChange();
    return new LineHandle(this, token, {
      capacity,
      xData: uploadX,
      record,
      autoRender: record.streamAutoRender,
      autoscaleY: record.streamAutoscaleY,
    });
  }

  streamLine(name, options = {}) {
    const capacity = Math.max(1, Number(options.capacity) | 0);
    const initial = options.initial ? ensureVector(options.initial, "initial") : new Float32Array([0]);
    const initialX = options.x ?? options.initialX ?? options.initial_x;
    return this.line(name, initial, { ...options, x: initialX, maxPoints: capacity });
  }

  stream_line(name, options = {}) {
    return this.streamLine(name, options);
  }

  primitive(kind, payload = {}, buffers = []) {
    this._assertReady();
    const token = this.nextPrimitiveToken++;
    const normalizedKind = String(kind);
    const kindCode = PRIMITIVE_KIND_CODES[normalizedKind];
    if (!kindCode) {
      throw new Error(`Unknown nbimplot primitive kind: ${kind}`);
    }
    const fullPayload = { ...payload, kind: normalizedKind, buffers };
    this._syncPrimitive(token, fullPayload);
    const record = {
      token,
      kind: normalizedKind,
      payload: { ...payload, kind: normalizedKind },
      buffers: buffers.map((buffer) => ensureFloat32(buffer).slice()),
      visible: !(payload.hidden === true || payload.visible === false),
    };
    this.primitiveRecords.set(token, record);
    this._afterDataChange();
    return new PrimitiveHandle(this, token, record);
  }

  _xyPrimitive(kind, name, y, options = {}) {
    const yv = ensureVector(y, "y");
    const buffers = [];
    let hasX = false;
    if (options.x != null) {
      const xv = ensureVector(options.x, "x");
      if (xv.length !== yv.length) throw new Error("x and y must have the same length.");
      buffers.push(xv);
      hasX = true;
    }
    buffers.push(yv);
    return this.primitive(kind, {
      name,
      hasX,
      length: yv.length,
      ...options,
    }, buffers);
  }

  scatter(name, y, options = {}) {
    return this._xyPrimitive("scatter", name, y, options);
  }

  bubbles(name, y, sizes, options = {}) {
    const yv = ensureVector(y, "y");
    const sv = ensureVector(sizes, "sizes");
    if (yv.length !== sv.length) throw new Error("y and sizes must have the same length.");
    const buffers = [];
    let hasX = false;
    if (options.x != null) {
      const xv = ensureVector(options.x, "x");
      if (xv.length !== yv.length) throw new Error("x, y, and sizes must have the same length.");
      buffers.push(xv);
      hasX = true;
    }
    buffers.push(yv, sv);
    return this.primitive("bubbles", { name, hasX, length: yv.length, ...options }, buffers);
  }

  stairs(name, y, options = {}) {
    return this._xyPrimitive("stairs", name, y, options);
  }

  stems(name, y, options = {}) {
    return this._xyPrimitive("stems", name, y, options);
  }

  digital(name, y, options = {}) {
    return this._xyPrimitive("digital", name, y, options);
  }

  bars(name, y, options = {}) {
    return this._xyPrimitive("bars", name, y, { barWidth: 0.67, ...options });
  }

  barGroups(labels, values, options = {}) {
    const matrix = normalizeMatrix(values, {
      rows: options.itemCount,
      cols: options.groupCount,
    }, "values");
    if (!Array.isArray(labels) || labels.length !== matrix.rows) {
      throw new Error("labels length must equal item count.");
    }
    return this.primitive("bar_groups", {
      labels,
      itemCount: matrix.rows,
      groupCount: matrix.cols,
      groupSize: options.groupSize ?? 0.67,
      shift: options.shift ?? 0,
      ...options,
    }, [matrix.data]);
  }

  bar_groups(labels, values, options = {}) {
    return this.barGroups(labels, values, options);
  }

  barsH(name, x, options = {}) {
    const xv = ensureVector(x, "x");
    const yv = options.y == null
      ? Float32Array.from({ length: xv.length }, (_, i) => i)
      : ensureVector(options.y, "y");
    if (xv.length !== yv.length) throw new Error("x and y must have the same length.");
    return this.primitive("bars_h", { name, length: xv.length, barHeight: options.barHeight ?? 0.67, ...options }, [xv, yv]);
  }

  bars_h(name, x, options = {}) {
    return this.barsH(name, x, options);
  }

  shaded(name, y1, y2, options = {}) {
    const a = ensureVector(y1, "y1");
    const b = ensureVector(y2, "y2");
    if (a.length !== b.length) throw new Error("y1 and y2 must have the same length.");
    const buffers = [];
    let hasX = false;
    if (options.x != null) {
      const xv = ensureVector(options.x, "x");
      if (xv.length !== a.length) throw new Error("x and y arrays must have the same length.");
      buffers.push(xv);
      hasX = true;
    }
    buffers.push(a, b);
    return this.primitive("shaded", { name, hasX, length: a.length, alpha: options.alpha ?? 0.2, ...options }, buffers);
  }

  errorBars(name, y, options = {}) {
    const yv = ensureVector(y, "y");
    const buffers = [];
    let asymmetric = false;
    if (options.errNeg != null || options.errPos != null || options.err_neg != null || options.err_pos != null) {
      const neg = ensureVector(options.errNeg ?? options.err_neg, "errNeg");
      const pos = ensureVector(options.errPos ?? options.err_pos, "errPos");
      if (neg.length !== yv.length || pos.length !== yv.length) throw new Error("asymmetric error arrays must match y length.");
      const interleaved = new Float32Array(yv.length * 2);
      for (let i = 0; i < yv.length; i += 1) {
        interleaved[i * 2] = neg[i];
        interleaved[i * 2 + 1] = pos[i];
      }
      buffers.push(yv, interleaved);
      asymmetric = true;
    } else {
      const err = ensureVector(options.err, "err");
      if (err.length !== yv.length) throw new Error("err must match y length.");
      buffers.push(yv, err);
    }
    let hasX = false;
    if (options.x != null) {
      const xv = ensureVector(options.x, "x");
      if (xv.length !== yv.length) throw new Error("x and y must have the same length.");
      buffers.unshift(xv);
      hasX = true;
    }
    return this.primitive("error_bars", { name, hasX, asymmetric, length: yv.length, ...options }, buffers);
  }

  error_bars(name, y, options = {}) {
    return this.errorBars(name, y, options);
  }

  errorBarsH(name, x, options = {}) {
    const xv = ensureVector(x, "x");
    const yv = options.y == null
      ? Float32Array.from({ length: xv.length }, (_, i) => i)
      : ensureVector(options.y, "y");
    if (xv.length !== yv.length) throw new Error("x and y must have the same length.");
    let err;
    let asymmetric = false;
    if (options.errNeg != null || options.errPos != null || options.err_neg != null || options.err_pos != null) {
      const neg = ensureVector(options.errNeg ?? options.err_neg, "errNeg");
      const pos = ensureVector(options.errPos ?? options.err_pos, "errPos");
      if (neg.length !== xv.length || pos.length !== xv.length) throw new Error("asymmetric error arrays must match x length.");
      err = new Float32Array(xv.length * 2);
      for (let i = 0; i < xv.length; i += 1) {
        err[i * 2] = neg[i];
        err[i * 2 + 1] = pos[i];
      }
      asymmetric = true;
    } else {
      err = ensureVector(options.err, "err");
      if (err.length !== xv.length) throw new Error("err must match x length.");
    }
    return this.primitive("error_bars_h", { name, asymmetric, length: xv.length, ...options }, [xv, err, yv]);
  }

  error_bars_h(name, x, options = {}) {
    return this.errorBarsH(name, x, options);
  }

  infLines(name, values, options = {}) {
    return this.primitive("inf_lines", { name, axis: options.axis || "x", length: ensureVector(values, "values").length, ...options }, [
      ensureVector(values, "values"),
    ]);
  }

  inf_lines(name, values, options = {}) {
    return this.infLines(name, values, options);
  }

  vlines(name, values, options = {}) {
    return this.infLines(name, values, { ...options, axis: "x" });
  }

  hlines(name, values, options = {}) {
    return this.infLines(name, values, { ...options, axis: "y" });
  }

  histogram(name, y, options = {}) {
    const values = ensureVector(y, "y");
    const { edges, counts } = histogram1d(values, options.bins ?? 50);
    return this.primitive("histogram", { name, bins: counts.length, ...options }, [edges, counts]);
  }

  histogram2d(name, x, y, options = {}) {
    const xv = ensureVector(x, "x");
    const yv = ensureVector(y, "y");
    if (xv.length !== yv.length) throw new Error("x and y must have the same length.");
    const hist = histogram2d(xv, yv, options.xBins ?? options.x_bins ?? 64, options.yBins ?? options.y_bins ?? 64);
    return this.primitive("histogram2d", {
      name,
      rows: hist.rows,
      cols: hist.cols,
      labelFmt: options.labelFmt ?? options.label_fmt ?? "%.0f",
      scaleMin: options.scaleMin ?? options.scale_min,
      scaleMax: options.scaleMax ?? options.scale_max,
      heatmapFlags: options.heatmapFlags ?? options.heatmap_flags ?? 0,
      showColorbar: Boolean(options.showColorbar ?? options.show_colorbar),
      colorbarLabel: options.colorbarLabel ?? options.colorbar_label ?? "",
      colorbarFormat: options.colorbarFormat ?? options.colorbar_format ?? "%g",
      colorbarFlags: options.colorbarFlags ?? options.colorbar_flags ?? 0,
      ...options,
    }, [hist.xEdges, hist.yEdges, hist.counts]);
  }

  heatmap(name, z, options = {}) {
    const matrix = normalizeMatrix(z, options, "z");
    return this.primitive("heatmap", {
      name,
      rows: matrix.rows,
      cols: matrix.cols,
      labelFmt: options.labelFmt ?? options.label_fmt ?? "%.2f",
      scaleMin: options.scaleMin ?? options.scale_min,
      scaleMax: options.scaleMax ?? options.scale_max,
      heatmapFlags: options.heatmapFlags ?? options.heatmap_flags ?? 0,
      showColorbar: Boolean(options.showColorbar ?? options.show_colorbar),
      colorbarLabel: options.colorbarLabel ?? options.colorbar_label ?? "",
      colorbarFormat: options.colorbarFormat ?? options.colorbar_format ?? "%g",
      colorbarFlags: options.colorbarFlags ?? options.colorbar_flags ?? 0,
      ...options,
    }, [matrix.data]);
  }

  image(name, z, options = {}) {
    const image = normalizeImage(z, options);
    const bounds = options.bounds || [[0, 0], [image.cols, image.rows]];
    const uv0 = options.uv0 || [0, 0];
    const uv1 = options.uv1 || [1, 1];
    const tint = new Float32Array(normalizeColor(options.tint || [1, 1, 1, 1]) || [1, 1, 1, 1]);
    return this.primitive("image", {
      name,
      rows: image.rows,
      cols: image.cols,
      channels: image.channels,
      boundsXMin: bounds[0][0],
      boundsYMin: bounds[0][1],
      boundsXMax: bounds[1][0],
      boundsYMax: bounds[1][1],
      uv0X: uv0[0],
      uv0Y: uv0[1],
      uv1X: uv1[0],
      uv1Y: uv1[1],
      imageFlags: options.imageFlags ?? options.image_flags ?? 0,
      ...options,
    }, [image.data, tint]);
  }

  pieChart(name, values, options = {}) {
    const vals = ensureVector(values, "values");
    const labels = options.labels || Array.from({ length: vals.length }, (_, i) => String(i));
    if (!Array.isArray(labels) || labels.length !== vals.length) throw new Error("labels length must match values length.");
    return this.primitive("pie_chart", {
      name,
      labels,
      x: options.x ?? 0,
      y: options.y ?? 0,
      radius: options.radius ?? 1,
      angle0: options.angle0 ?? 90,
      labelFmt: options.labelFmt ?? options.label_fmt ?? "%.1f",
      ...options,
    }, [vals]);
  }

  pie_chart(name, values, options = {}) {
    return this.pieChart(name, values, options);
  }

  text(label, x, y, options = {}) {
    return this.primitive("text", { label, x, y, ...options }, []);
  }

  annotation(label, x, y, options = {}) {
    return this.primitive("annotation", {
      label,
      x,
      y,
      offsetX: options.offsetX ?? options.offset_x ?? 8,
      offsetY: options.offsetY ?? options.offset_y ?? -8,
      ...options,
    }, []);
  }

  dummy(name, options = {}) {
    return this.primitive("dummy", { name, ...options }, []);
  }

  tagX(value, options = {}) {
    return this.primitive("tag_x", {
      value,
      labelFmt: options.labelFmt ?? options.label_fmt ?? "%g",
      roundValue: Boolean(options.roundValue ?? options.round_value),
      ...options,
    }, []);
  }

  tag_x(value, options = {}) {
    return this.tagX(value, options);
  }

  tagY(value, options = {}) {
    return this.primitive("tag_y", {
      value,
      labelFmt: options.labelFmt ?? options.label_fmt ?? "%g",
      roundValue: Boolean(options.roundValue ?? options.round_value),
      ...options,
    }, []);
  }

  tag_y(value, options = {}) {
    return this.tagY(value, options);
  }

  colormapSlider(options = {}) {
    return this.primitive("colormap_slider", {
      label: options.label ?? "Colormap",
      value: options.t ?? options.value ?? 0.5,
      labelFmt: options.fmt ?? options.labelFmt ?? "",
      ...options,
    }, []);
  }

  colormap_slider(options = {}) {
    return this.colormapSlider(options);
  }

  colormapButton(options = {}) {
    return this.primitive("colormap_button", {
      label: options.label ?? "Colormap",
      x: options.width ?? 0,
      y: options.height ?? 0,
      ...options,
    }, []);
  }

  colormap_button(options = {}) {
    return this.colormapButton(options);
  }

  colormapSelector(options = {}) {
    return this.primitive("colormap_selector", { label: options.label ?? "Colormap", ...options }, []);
  }

  colormap_selector(options = {}) {
    return this.colormapSelector(options);
  }

  dragLineX(name, value, options = {}) {
    return this.primitive("drag_line_x", { name, value, thickness: options.thickness ?? 1, ...options }, []);
  }

  drag_line_x(name, value, options = {}) {
    return this.dragLineX(name, value, options);
  }

  dragLineY(name, value, options = {}) {
    return this.primitive("drag_line_y", { name, value, thickness: options.thickness ?? 1, ...options }, []);
  }

  drag_line_y(name, value, options = {}) {
    return this.dragLineY(name, value, options);
  }

  dragPoint(name, x, y, options = {}) {
    return this.primitive("drag_point", { name, x, y, size: options.size ?? 4, ...options }, []);
  }

  drag_point(name, x, y, options = {}) {
    return this.dragPoint(name, x, y, options);
  }

  dragRect(name, x1, y1, x2, y2, options = {}) {
    return this.primitive("drag_rect", { name, x1, y1, x2, y2, ...options }, []);
  }

  drag_rect(name, x1, y1, x2, y2, options = {}) {
    return this.dragRect(name, x1, y1, x2, y2, options);
  }

  dragDropPlot(options = {}) {
    return this.primitive("drag_drop_plot", {
      sourceEnabled: options.source ?? true,
      targetEnabled: options.target ?? true,
      ...options,
    }, []);
  }

  drag_drop_plot(options = {}) {
    return this.dragDropPlot(options);
  }

  dragDropAxis(axis, options = {}) {
    return this.primitive("drag_drop_axis", {
      sourceEnabled: options.source ?? true,
      targetEnabled: options.target ?? true,
      axisCode: axisCode(axis),
      ...options,
    }, []);
  }

  drag_drop_axis(axis, options = {}) {
    return this.dragDropAxis(axis, options);
  }

  dragDropLegend(options = {}) {
    return this.primitive("drag_drop_legend", {
      targetEnabled: options.target ?? true,
      ...options,
    }, []);
  }

  drag_drop_legend(options = {}) {
    return this.dragDropLegend(options);
  }

  setView(xMin, xMax, yMin, yMax) {
    this._assertReady();
    this.initialAutoFitActive = false;
    this.view = { xMin: Number(xMin), xMax: Number(xMax), yMin: Number(yMin), yMax: Number(yMax) };
    this.wasm.setView(this.view);
    this.requestRender();
    return this;
  }

  set_view(xMin, xMax, yMin, yMax) {
    return this.setView(xMin, xMax, yMin, yMax);
  }

  autoscale() {
    this._assertReady();
    const view = this.wasm.autoscale();
    if (view) this.view = view;
    this.requestRender();
    return this;
  }

  setColormap(name = "") {
    this._assertReady();
    this.colormapName = String(name || "");
    this.wasm.setColormap(this.colormapName);
    this.requestRender();
    return this;
  }

  set_colormap(name = "") {
    return this.setColormap(name);
  }

  setPlotFlags(options = {}) {
    this._assertReady();
    this.plotFlags = plotFlagsFromOptions(options);
    this.wasm.setPlotOptions(this.plotFlags, this.axisScaleX, this.axisScaleY);
    this.requestRender();
    return this;
  }

  setAxisScale(options = {}) {
    this._assertReady();
    this.axisScaleX = scaleCode(options.x || "linear");
    this.axisScaleY = scaleCode(options.y || "linear");
    this.axisState[AXES.x1] = { enabled: true, scale: this.axisScaleX };
    this.axisState[AXES.y1] = { enabled: true, scale: this.axisScaleY };
    this.wasm.setPlotOptions(this.plotFlags, this.axisScaleX, this.axisScaleY);
    this.wasm.setAxisState(AXES.x1, true, this.axisScaleX);
    this.wasm.setAxisState(AXES.y1, true, this.axisScaleY);
    this.requestRender();
    return this;
  }

  set_axis_scale(options = {}) {
    return this.setAxisScale(options);
  }

  setSecondaryAxes(options = {}) {
    this._assertReady();
    for (const [axis, enabled] of [
      [AXES.x2, Boolean(options.x2)],
      [AXES.x3, Boolean(options.x3)],
      [AXES.y2, Boolean(options.y2)],
      [AXES.y3, Boolean(options.y3)],
    ]) {
      const scale = this.axisState[axis]?.scale ?? AXIS_SCALES.linear;
      this.axisState[axis] = { enabled, scale };
      this.wasm.setAxisState(axis, enabled, scale);
    }
    this.requestRender();
    return this;
  }

  set_secondary_axes(options = {}) {
    return this.setSecondaryAxes(options);
  }

  setTimeAxis(axis = "x1") {
    this._assertReady();
    this.wasm.setAxisState(axisCode(axis), true, AXIS_SCALES.time);
    this.requestRender();
    return this;
  }

  set_time_axis(axis = "x1") {
    return this.setTimeAxis(axis);
  }

  setAxisState(axis, options = {}) {
    this._assertReady();
    const code = axisCode(axis);
    const enabled = code === AXES.x1 || code === AXES.y1 ? true : Boolean(options.enabled);
    const scale = scaleCode(options.scale || "linear");
    this.axisState[code] = { enabled, scale };
    if (code === AXES.x1) this.axisScaleX = scale;
    if (code === AXES.y1) this.axisScaleY = scale;
    this.wasm.setAxisState(code, enabled, scale);
    if (code === AXES.x1 || code === AXES.y1) {
      this.wasm.setPlotOptions(this.plotFlags, this.axisScaleX, this.axisScaleY);
    }
    this.requestRender();
    return this;
  }

  set_axis_state(axis, options = {}) {
    return this.setAxisState(axis, options);
  }

  setAxisLabel(axis, label = "") {
    this._assertReady();
    const code = axisCode(axis);
    this.axisLabels[code] = String(label || "");
    this.wasm.setAxisLabel(code, this.axisLabels[code]);
    this.requestRender();
    return this;
  }

  set_axis_label(axis, label = "") {
    return this.setAxisLabel(axis, label);
  }

  setAxisFormat(axis, format = "") {
    this._assertReady();
    const code = axisCode(axis);
    this.axisFormats[code] = String(format || "");
    this.wasm.setAxisFormat(code, this.axisFormats[code]);
    this.requestRender();
    return this;
  }

  set_axis_format(axis, format = "") {
    return this.setAxisFormat(axis, format);
  }

  setAxisTicks(axis, values, options = {}) {
    this._assertReady();
    const code = axisCode(axis);
    const ticks = ensureVector(values, "tick values");
    const labels = Array.isArray(options.labels) ? options.labels.map(String) : [];
    const keepDefault = Boolean(options.keepDefault ?? options.keep_default);
    this.axisTicks.set(code, { ticks, labels, keepDefault });
    this.wasm.setAxisTicks(code, ticks, labels, keepDefault);
    this.requestRender();
    return this;
  }

  set_axis_ticks(axis, values, options = {}) {
    return this.setAxisTicks(axis, values, options);
  }

  clearAxisTicks(axis) {
    this._assertReady();
    const code = axisCode(axis);
    this.axisTicks.delete(code);
    this.wasm.clearAxisTicks(code);
    this.requestRender();
    return this;
  }

  clear_axis_ticks(axis) {
    return this.clearAxisTicks(axis);
  }

  setAxisLimitsConstraints(axis, minValue, maxValue, options = {}) {
    this._assertReady();
    const code = axisCode(axis);
    this.axisLimitsConstraints.set(code, {
      enabled: options.enabled !== false,
      min: Number(minValue),
      max: Number(maxValue),
    });
    this.wasm.setAxisLimitsConstraints(code, options.enabled !== false, minValue, maxValue);
    this.requestRender();
    return this;
  }

  set_axis_limits_constraints(axis, minValue, maxValue, options = {}) {
    return this.setAxisLimitsConstraints(axis, minValue, maxValue, options);
  }

  setAxisZoomConstraints(axis, minZoom, maxZoom, options = {}) {
    this._assertReady();
    const code = axisCode(axis);
    this.axisZoomConstraints.set(code, {
      enabled: options.enabled !== false,
      min: Number(minZoom),
      max: Number(maxZoom),
    });
    this.wasm.setAxisZoomConstraints(code, options.enabled !== false, minZoom, maxZoom);
    this.requestRender();
    return this;
  }

  set_axis_zoom_constraints(axis, minZoom, maxZoom, options = {}) {
    return this.setAxisZoomConstraints(axis, minZoom, maxZoom, options);
  }

  setAxisLink(axis, targetAxis = null) {
    this._assertReady();
    const code = axisCode(axis);
    const target = targetAxis == null ? null : axisCode(targetAxis);
    if (target == null) this.axisLinks.delete(code);
    else this.axisLinks.set(code, target);
    this.wasm.setAxisLink(code, target);
    this.requestRender();
    return this;
  }

  set_axis_link(axis, targetAxis = null) {
    return this.setAxisLink(axis, targetAxis);
  }

  setSubplots(rows, cols, options = {}) {
    this._assertReady();
    this.subplotRows = Math.max(1, Number(rows) | 0);
    this.subplotCols = Math.max(1, Number(cols) | 0);
    this.subplotFlags = subplotFlagsFromOptions(options);
    this.wasm.setSubplots(this.subplotRows, this.subplotCols, this.subplotFlags);
    this._hideLinkedCrosshair();
    this.requestRender();
    return this;
  }

  set_subplots_config(options = {}) {
    return this.setSubplots(options.rows ?? 1, options.cols ?? 1, options);
  }

  setAlignedGroup(groupId, options = {}) {
    this._assertReady();
    this.alignedGroup = {
      groupId: String(groupId || ""),
      enabled: options.enabled !== false,
      vertical: options.vertical !== false,
    };
    this.wasm.setAlignedGroup(this.alignedGroup.groupId, this.alignedGroup.enabled, this.alignedGroup.vertical);
    this.requestRender();
    return this;
  }

  set_aligned_group(groupId, options = {}) {
    return this.setAlignedGroup(groupId, options);
  }

  setTheme(name = "nbimplot") {
    this.themeName = String(name || "nbimplot");
    if (this.wrapper) this.wrapper.dataset.nbimplotTheme = this.themeName;
    this.requestRender();
    return this;
  }

  set_theme(name = "nbimplot") {
    return this.setTheme(name);
  }

  setLinkedCrosshair(groupId = "default", options = {}) {
    const previous = this.linkedCrosshair.groupId;
    if (previous) {
      const group = linkedCrosshairGroups.get(previous);
      if (group) {
        group.delete(this);
        if (group.size === 0) linkedCrosshairGroups.delete(previous);
      }
    }
    const axis = String(options.axis || "x").toLowerCase();
    this.linkedCrosshair = {
      enabled: options.enabled !== false,
      groupId: String(groupId || "default"),
      axis: axis === "y" || axis === "xy" ? axis : "x",
    };
    if (this.linkedCrosshair.enabled) {
      linkedCrosshairGroup(this.linkedCrosshair.groupId).add(this);
    } else {
      this._hideLinkedCrosshair();
    }
    return this;
  }

  set_linked_crosshair(groupId = "default", options = {}) {
    return this.setLinkedCrosshair(groupId, options);
  }

  getState(options = {}) {
    const includeData = Boolean(options.includeData ?? options.include_data);
    const state = {
      version: 1,
      width: this.width,
      height: this.height,
      title: this.title,
      plotFlags: this.plotFlags,
      axisScaleX: resolveScaleName(this.axisScaleX),
      axisScaleY: resolveScaleName(this.axisScaleY),
      colormap: this.colormapName,
      theme: this.themeName,
      view: this.view ? { ...this.view } : null,
      subplots: { rows: this.subplotRows, cols: this.subplotCols, flags: this.subplotFlags },
      alignedGroup: this.alignedGroup ? { ...this.alignedGroup } : null,
      linkedCrosshair: { ...this.linkedCrosshair },
      axisState: this.axisState.map((item) => ({ enabled: item.enabled, scale: resolveScaleName(item.scale) })),
      axisLabels: [...this.axisLabels],
      axisFormats: [...this.axisFormats],
      axisLinks: Object.fromEntries([...this.axisLinks.entries()].map(([axis, target]) => [axis, target])),
      series: [],
      primitives: [],
    };
    if (includeData) {
      state.axisTicks = Object.fromEntries(
        [...this.axisTicks.entries()].map(([axis, cfg]) => [
          axis,
          { ticks: Array.from(cfg.ticks), labels: [...cfg.labels], keepDefault: cfg.keepDefault },
        ]),
      );
      state.axisLimitsConstraints = Object.fromEntries([...this.axisLimitsConstraints.entries()]);
      state.axisZoomConstraints = Object.fromEntries([...this.axisZoomConstraints.entries()]);
    }
    for (const record of this.seriesByToken.values()) {
      const item = {
        token: record.token,
        name: record.name,
        subplotIndex: record.subplotIndex,
        xAxis: resolveAxisName(record.xAxis),
        yAxis: resolveAxisName(record.yAxis),
        visible: record.visible,
        capacity: record.capacity || 0,
        style: {
          ...record.style,
          marker: resolveMarkerName(markerCode(record.style?.marker ?? "none")),
        },
        hasX: Boolean(record.xData),
        streamPaused: Boolean(record.streamPaused),
        streamAutoRender: Boolean(record.streamAutoRender),
        streamAutoscaleY: Boolean(record.streamAutoscaleY),
      };
      if (includeData) {
        item.y = Array.from(record.data || []);
        if (record.xData) item.x = Array.from(record.xData);
      }
      state.series.push(item);
    }
    if (includeData) {
      for (const record of this.primitiveRecords.values()) {
        state.primitives.push({
          kind: record.kind,
          payload: { ...record.payload, buffers: undefined },
          buffers: record.buffers.map((buffer) => Array.from(buffer)),
          visible: record.visible,
        });
      }
    }
    return state;
  }

  get_state(options = {}) {
    return this.getState(options);
  }

  setState(state = {}) {
    this._assertReady();
    if (state.title != null) this.title = String(state.title);
    if (state.width != null) this.width = Math.max(120, Number(state.width));
    if (state.height != null) this.height = Math.max(100, Number(state.height));
    if (state.subplots) {
      this.setSubplots(state.subplots.rows ?? 1, state.subplots.cols ?? 1, { flags: state.subplots.flags ?? 0 });
    }
    if (state.plotFlags != null || state.plot_flags != null) {
      this.plotFlags = Number(state.plotFlags ?? state.plot_flags) | 0;
      this.wasm.setPlotOptions(this.plotFlags, this.axisScaleX, this.axisScaleY);
    }
    if (state.axisState) {
      state.axisState.forEach((cfg, axis) => {
        if (!cfg) return;
        this.setAxisState(resolveAxisName(axis), { enabled: cfg.enabled, scale: cfg.scale || "linear" });
      });
    }
    if (state.axisLabels) {
      state.axisLabels.forEach((label, axis) => this.setAxisLabel(resolveAxisName(axis), label || ""));
    }
    if (state.axisFormats) {
      state.axisFormats.forEach((format, axis) => this.setAxisFormat(resolveAxisName(axis), format || ""));
    }
    if (state.axisLinks) {
      for (const [axis, target] of Object.entries(state.axisLinks)) {
        this.setAxisLink(resolveAxisName(axis), resolveAxisName(target));
      }
    }
    if (state.axisTicks) {
      for (const [axis, cfg] of Object.entries(state.axisTicks)) {
        if (!cfg) continue;
        this.setAxisTicks(resolveAxisName(axis), new Float32Array(cfg.ticks || []), {
          labels: cfg.labels || [],
          keepDefault: Boolean(cfg.keepDefault ?? cfg.keep_default),
        });
      }
    }
    if (state.axisLimitsConstraints) {
      for (const [axis, cfg] of Object.entries(state.axisLimitsConstraints)) {
        if (!cfg) continue;
        this.setAxisLimitsConstraints(resolveAxisName(axis), cfg.min, cfg.max, { enabled: cfg.enabled !== false });
      }
    }
    if (state.axisZoomConstraints) {
      for (const [axis, cfg] of Object.entries(state.axisZoomConstraints)) {
        if (!cfg) continue;
        this.setAxisZoomConstraints(resolveAxisName(axis), cfg.min, cfg.max, { enabled: cfg.enabled !== false });
      }
    }
    if (state.alignedGroup) {
      this.setAlignedGroup(state.alignedGroup.groupId, state.alignedGroup);
    }
    if (state.theme) this.setTheme(state.theme);
    if (state.colormap != null) this.setColormap(state.colormap);
    if (state.linkedCrosshair) {
      this.setLinkedCrosshair(state.linkedCrosshair.groupId || "default", state.linkedCrosshair);
    }
    if (state.view) {
      this.setView(state.view.xMin ?? state.view.x_min, state.view.xMax ?? state.view.x_max, state.view.yMin ?? state.view.y_min, state.view.yMax ?? state.view.y_max);
    }
    for (const item of state.series || []) {
      if (!item || !item.y) continue;
      const handle = this.line(item.name || "series", new Float32Array(item.y), {
        x: item.x ? new Float32Array(item.x) : undefined,
        subplotIndex: item.subplotIndex ?? item.subplot_index ?? 0,
        xAxis: item.xAxis ?? item.x_axis ?? "x1",
        yAxis: item.yAxis ?? item.y_axis ?? "y1",
        maxPoints: item.capacity ?? item.maxPoints ?? 0,
        visible: item.visible !== false,
        ...(item.style || {}),
      });
      handle.setStreamOptions({
        autoRender: Boolean(item.streamAutoRender ?? item.stream_auto_render),
        autoscaleY: Boolean(item.streamAutoscaleY ?? item.stream_autoscale_y),
      });
      if (item.streamPaused ?? item.stream_paused) handle.pause();
    }
    for (const item of state.primitives || []) {
      if (!item || !item.kind) continue;
      this.primitive(item.kind, item.payload || {}, (item.buffers || []).map((buffer) => new Float32Array(buffer)));
    }
    this._resize();
    this.requestRender();
    return this;
  }

  set_state(state = {}) {
    return this.setState(state);
  }

  exportJSONState(options = {}) {
    const text = JSON.stringify(this.getState(options), null, 2);
    if (options.download || options.filename) {
      downloadBlob(new Blob([text], { type: "application/json;charset=utf-8" }), sanitizeJsonFilename(options.filename));
    }
    return text;
  }

  export_json_state(options = {}) {
    return this.exportJSONState(options);
  }

  getView() {
    this._assertReady();
    return this.wasm.getView();
  }

  getPerfStats() {
    this._assertReady();
    return this.wasm.getPerfStats();
  }

  _syncPrimitive(token, payload) {
    const buffers = Array.isArray(payload.buffers) ? payload.buffers : [];
    const data0 = buffers[0] ? ensureFloat32(buffers[0], "data0") : new Float32Array(0);
    const data1 = buffers[1] ? ensureFloat32(buffers[1], "data1") : new Float32Array(0);
    const data2 = buffers[2] ? ensureFloat32(buffers[2], "data2") : new Float32Array(0);
    const [xAxis, yAxis] = axesCodes(payload.xAxis || payload.x_axis || "x1", payload.yAxis || payload.y_axis || "y1");

    const ints = [
      payload.hasX || payload.has_x ? 1 : 0,
      0,
      0,
      Math.max(1, Number(payload.version || 1) | 0),
      xAxis,
      yAxis,
      0,
      Math.max(0, Number(payload.subplotIndex ?? payload.subplot_index ?? 0) | 0),
    ];
    const floats = [0, 0, 0, 0, 0, 0, 0, 0];
    let text = String(payload.name || "");
    const labelFormatOrDefault = (value, fallback) => value == null ? fallback : String(value);

    switch (payload.kind) {
      case "bars":
        floats[1] = Number(payload.barWidth ?? payload.bar_width ?? 0.67);
        break;
      case "bar_groups":
        ints[1] = Number(payload.itemCount ?? payload.item_count ?? 0) | 0;
        ints[2] = Number(payload.groupCount ?? payload.group_count ?? 0) | 0;
        floats[1] = Number(payload.groupSize ?? payload.group_size ?? 0.67);
        floats[2] = Number(payload.shift ?? 0);
        text = Array.isArray(payload.labels) ? payload.labels.map(String).join(LABEL_SEP) : text;
        break;
      case "bars_h":
        floats[2] = Number(payload.barHeight ?? payload.bar_height ?? 0.67);
        break;
      case "shaded":
        floats[3] = Number(payload.alpha ?? 0.2);
        break;
      case "inf_lines":
        ints[1] = String(payload.axis || "x").toLowerCase() === "y" ? 1 : 0;
        break;
      case "error_bars":
      case "error_bars_h":
        ints[1] = payload.asymmetric ? 1 : 0;
        break;
      case "histogram2d":
      case "heatmap":
        ints[1] = Number(payload.rows || 0) | 0;
        ints[2] = Number(payload.cols || 0) | 0;
        ints[3] = Math.max(0, Number(payload.heatmapFlags ?? payload.heatmap_flags ?? 0) | 0);
        ints[0] = payload.showColorbar || payload.show_colorbar ? 1 : 0;
        ints[6] = Math.max(0, Number(payload.colorbarFlags ?? payload.colorbar_flags ?? 0) | 0);
        floats[0] = payload.scaleMin != null || payload.scale_min != null ? Number(payload.scaleMin ?? payload.scale_min) : Number.NaN;
        floats[1] = payload.scaleMax != null || payload.scale_max != null ? Number(payload.scaleMax ?? payload.scale_max) : Number.NaN;
        text = `${labelFormatOrDefault(payload.labelFmt ?? payload.label_fmt, payload.kind === "heatmap" ? "%.2f" : "%.0f")}${HEATMAP_META_SEP}${String(payload.colorbarLabel ?? payload.colorbar_label ?? "")}${HEATMAP_META_SEP}${labelFormatOrDefault(payload.colorbarFormat ?? payload.colorbar_format, "%g")}`;
        break;
      case "image":
        ints[0] = Math.max(0, Number(payload.imageFlags ?? payload.image_flags ?? 0) | 0);
        ints[1] = Number(payload.rows || 0) | 0;
        ints[2] = Number(payload.cols || 0) | 0;
        ints[3] = Math.max(1, Number(payload.version || 1) | 0);
        ints[6] = Math.max(1, Number(payload.channels || 1) | 0);
        floats[0] = Number(payload.boundsXMin ?? payload.bounds_x_min ?? 0);
        floats[1] = Number(payload.boundsXMax ?? payload.bounds_x_max ?? payload.cols ?? 0);
        floats[2] = Number(payload.boundsYMin ?? payload.bounds_y_min ?? 0);
        floats[3] = Number(payload.boundsYMax ?? payload.bounds_y_max ?? payload.rows ?? 0);
        floats[4] = Number(payload.uv0X ?? payload.uv0_x ?? 0);
        floats[5] = Number(payload.uv0Y ?? payload.uv0_y ?? 0);
        floats[6] = Number(payload.uv1X ?? payload.uv1_x ?? 1);
        floats[7] = Number(payload.uv1Y ?? payload.uv1_y ?? 1);
        break;
      case "tag_x":
        floats[4] = Number(payload.value || 0);
        ints[1] = payload.roundValue || payload.round_value ? 1 : 0;
        text = labelFormatOrDefault(payload.labelFmt ?? payload.label_fmt, "%g");
        break;
      case "tag_y":
        floats[5] = Number(payload.value || 0);
        ints[1] = payload.roundValue || payload.round_value ? 1 : 0;
        text = labelFormatOrDefault(payload.labelFmt ?? payload.label_fmt, "%g");
        break;
      case "colormap_slider":
        floats[4] = Number(payload.value ?? payload.t ?? 0.5);
        text = `${String(payload.label || "Colormap")}${HEATMAP_META_SEP}${labelFormatOrDefault(payload.labelFmt ?? payload.label_fmt, "")}`;
        break;
      case "colormap_button":
        floats[4] = Number(payload.x ?? payload.width ?? 0);
        floats[5] = Number(payload.y ?? payload.height ?? 0);
        text = String(payload.label || "Colormap");
        break;
      case "colormap_selector":
        text = String(payload.label || "Colormap");
        break;
      case "pie_chart": {
        floats[4] = Number(payload.x || 0);
        floats[5] = Number(payload.y || 0);
        floats[6] = Number(payload.radius || 1);
        floats[7] = Number(payload.angle0 || 90);
        const fmt = String(payload.labelFmt ?? payload.label_fmt ?? "%.1f");
        const labels = Array.isArray(payload.labels) ? payload.labels.map(String) : [];
        text = `${fmt}${PIE_FMT_SEP}${labels.join(LABEL_SEP)}`;
        break;
      }
      case "text":
        floats[4] = Number(payload.x || 0);
        floats[5] = Number(payload.y || 0);
        text = String(payload.label || "");
        break;
      case "annotation":
        floats[4] = Number(payload.x || 0);
        floats[5] = Number(payload.y || 0);
        floats[6] = Number(payload.offsetX ?? payload.offset_x ?? 8);
        floats[7] = Number(payload.offsetY ?? payload.offset_y ?? -8);
        text = String(payload.label || "");
        break;
      case "drag_line_x":
        floats[4] = Number(payload.value || 0);
        floats[6] = Number(payload.thickness || 1);
        text = String(payload.name || "drag_x");
        break;
      case "drag_line_y":
        floats[5] = Number(payload.value || 0);
        floats[6] = Number(payload.thickness || 1);
        text = String(payload.name || "drag_y");
        break;
      case "drag_point":
        floats[4] = Number(payload.x || 0);
        floats[5] = Number(payload.y || 0);
        floats[6] = Number(payload.size || 4);
        text = String(payload.name || "drag_point");
        break;
      case "drag_rect":
        floats[4] = Number(payload.x1 || 0);
        floats[5] = Number(payload.y1 || 0);
        floats[6] = Number(payload.x2 || 1);
        floats[7] = Number(payload.y2 || 1);
        text = String(payload.name || "drag_rect");
        break;
      case "drag_drop_plot": {
        const source = payload.sourceEnabled ?? payload.source ?? payload.hasX ?? true;
        const target = payload.targetEnabled ?? payload.target ?? true;
        ints[0] = source ? 1 : 0;
        ints[1] = target ? 1 : 0;
        break;
      }
      case "drag_drop_axis": {
        const source = payload.sourceEnabled ?? payload.source ?? true;
        const target = payload.targetEnabled ?? payload.target ?? true;
        ints[0] = source ? 1 : 0;
        ints[1] = target ? 1 : 0;
        ints[2] = Number(payload.axisCode ?? 0) | 0;
        break;
      }
      case "drag_drop_legend": {
        const target = payload.targetEnabled ?? payload.target ?? true;
        ints[1] = target ? 1 : 0;
        break;
      }
      default:
        break;
    }

    const ok = this.wasm.upsertPrimitive(token, PRIMITIVE_KIND_CODES[payload.kind], {
      data0,
      data1,
      data2,
      ints,
      floats,
      text,
    });
    if (!ok) {
      throw new Error(`Failed to upload primitive: ${payload.kind}`);
    }
    if (payload.hidden === true || payload.visible === false) {
      this.wasm.setPrimitiveVisible(token, false);
    }
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.rafId !== 0) {
      window.cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
    if (this.linkedCrosshair.groupId) {
      const group = linkedCrosshairGroups.get(this.linkedCrosshair.groupId);
      if (group) {
        group.delete(this);
        if (group.size === 0) linkedCrosshairGroups.delete(this.linkedCrosshair.groupId);
      }
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    this.canvas.removeEventListener("mousemove", this.onMouseMove);
    this.canvas.removeEventListener("mousedown", this.onMouseDown);
    window.removeEventListener("mouseup", this.onMouseUp);
    this.canvas.removeEventListener("mouseleave", this.onMouseLeave);
    this.canvas.removeEventListener("wheel", this.onWheel);
    this.canvas.removeEventListener("dblclick", this.onDoubleClick);
    this.canvas.removeEventListener("contextmenu", this.onContextMenu);
    window.removeEventListener("resize", this.onWindowResize);
    if (this.wasm) {
      this.wasm.destroy();
    }
    this.seriesByToken.clear();
    this.primitiveRecords.clear();
    this.wrapper.remove();
  }
}

export async function createPlot(target, options = {}) {
  const plot = new WebPlot(target, options);
  return plot.init();
}

export async function createDashboard(target, options = {}) {
  const rows = Math.max(1, Number(options.rows || 1) | 0);
  const cols = Math.max(1, Number(options.cols || 1) | 0);
  const plot = await createPlot(target, {
    ...options,
    title: options.title || "Dashboard",
    width: options.width || 1100,
    height: options.height || 650,
    subplotRows: rows,
    subplotCols: cols,
    linkAllX: options.linkX !== false,
    linkAllY: Boolean(options.linkY),
    crosshairs: options.crosshairs !== false,
  });
  if (options.theme !== false) {
    plot.setTheme(options.theme || "nbimplot");
  }
  if (options.linkedCrosshair !== false) {
    plot.setLinkedCrosshair(options.crosshairGroup || "dashboard", { axis: options.crosshairAxis || "x" });
  }
  return plot;
}

export const Plot = WebPlot;
export const Dashboard = createDashboard;
