import createNbImPlotModule from "../wasm/nbimplot_wasm.js";

const DEFAULT_WASM_URL = new URL("../wasm/nbimplot_wasm.wasm", import.meta.url);
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
  }

  setData(y) {
    this.plot._assertReady();
    if (!this.plot.wasm.upsertLine(this.token, ensureVector(y, "y"), false)) {
      throw new Error("Failed to update line data.");
    }
    this.plot._afterDataChange();
    return this;
  }

  append(y) {
    this.plot._assertReady();
    if (!this.plot.wasm.appendLineData(this.token, ensureVector(y, "y"), this.capacity)) {
      throw new Error("Failed to append line data.");
    }
    this.plot._afterDataChange();
    return this;
  }

  setVisible(visible) {
    this.plot._assertReady();
    this.plot.wasm.setSeriesVisible(this.token, Boolean(visible));
    this.plot.requestRender();
    return this;
  }

  setStyle(style = {}) {
    this.plot._assertReady();
    this.plot.wasm.setSeriesStyle(this.token, style);
    this.plot.requestRender();
    return this;
  }
}

class PrimitiveHandle {
  constructor(plot, token) {
    this.plot = plot;
    this.token = token >>> 0;
  }

  setVisible(visible) {
    this.plot._assertReady();
    this.plot.wasm.setPrimitiveVisible(this.token, Boolean(visible));
    this.plot.requestRender();
    return this;
  }

  remove() {
    this.plot._assertReady();
    this.plot.wasm.removePrimitive(this.token);
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
    this.perfCallbacks = new Set();
    this.lastInteractionHash = "";
    this.plotFlags = plotFlagsFromOptions(options);
    this.axisScaleX = scaleCode(options.axisScaleX || "linear");
    this.axisScaleY = scaleCode(options.axisScaleY || "linear");
    this.subplotRows = Math.max(1, Number(options.subplotRows || 1) | 0);
    this.subplotCols = Math.max(1, Number(options.subplotCols || 1) | 0);
    this.subplotFlags = subplotFlagsFromOptions(options);
    this.colormapName = options.colormap ? String(options.colormap) : "";
    this.alignedGroup = null;
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
    if (this.alignedGroup) {
      this.wasm.setAlignedGroup(
        this.alignedGroup.groupId,
        this.alignedGroup.enabled,
        this.alignedGroup.vertical,
      );
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
    if (this.interactionCallbacks.size === 0) return;
    const tuples = this.wasm.getInteractions();
    if (!(tuples instanceof Float32Array) || tuples.length === 0) return;
    const payload = [];
    for (let i = 0; i + 7 < tuples.length; i += 8) {
      payload.push({
        kind: tuples[i] | 0,
        id: tuples[i + 1] | 0,
        subplotIndex: tuples[i + 2] | 0,
        active: (tuples[i + 3] | 0) !== 0,
        v0: Number(tuples[i + 4]),
        v1: Number(tuples[i + 5]),
        v2: Number(tuples[i + 6]),
        v3: Number(tuples[i + 7]),
      });
    }
    const hash = JSON.stringify(payload);
    if (hash === this.lastInteractionHash) return;
    this.lastInteractionHash = hash;
    for (const callback of this.interactionCallbacks) {
      callback(payload, this);
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

  line(name, y, options = {}) {
    this._assertReady();
    const token = this.nextSeriesToken++;
    const data = ensureVector(y, "y");
    const [xAxis, yAxis] = axesCodes(options.xAxis || options.x_axis || "x1", options.yAxis || options.y_axis || "y1");
    const capacity = Math.max(0, Number(options.maxPoints || options.max_points || 0) | 0);
    const upload = capacity > 0 && data.length > capacity ? data.subarray(data.length - capacity) : data;
    if (!this.wasm.upsertLine(token, upload, true)) {
      throw new Error("Failed to upload line data.");
    }
    this.wasm.setSeriesName(token, name);
    this.wasm.setSeriesSubplot(token, options.subplotIndex ?? options.subplot_index ?? 0);
    this.wasm.setSeriesAxes(token, xAxis, yAxis);
    this.wasm.setSeriesStyle(token, {
      color: options.color,
      lineWeight: options.lineWeight ?? options.line_weight ?? 1,
      marker: options.marker ?? "none",
      markerSize: options.markerSize ?? options.marker_size ?? 4,
    });
    if (options.visible === false || options.hidden === true) {
      this.wasm.setSeriesVisible(token, false);
    }
    this._afterDataChange();
    return new LineHandle(this, token, { capacity });
  }

  streamLine(name, options = {}) {
    const capacity = Math.max(1, Number(options.capacity) | 0);
    const initial = options.initial ? ensureVector(options.initial, "initial") : new Float32Array([0]);
    return this.line(name, initial, { ...options, maxPoints: capacity });
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
    this._afterDataChange();
    return new PrimitiveHandle(this, token);
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
    this.wasm.setView({ xMin: Number(xMin), xMax: Number(xMax), yMin: Number(yMin), yMax: Number(yMax) });
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
    this.wasm.setPlotOptions(this.plotFlags, this.axisScaleX, this.axisScaleY);
    this.requestRender();
    return this;
  }

  set_axis_scale(options = {}) {
    return this.setAxisScale(options);
  }

  setSecondaryAxes(options = {}) {
    this._assertReady();
    this.wasm.setAxisState(AXES.x2, Boolean(options.x2), AXIS_SCALES.linear);
    this.wasm.setAxisState(AXES.x3, Boolean(options.x3), AXIS_SCALES.linear);
    this.wasm.setAxisState(AXES.y2, Boolean(options.y2), AXIS_SCALES.linear);
    this.wasm.setAxisState(AXES.y3, Boolean(options.y3), AXIS_SCALES.linear);
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
    this.wasm.setAxisState(axisCode(axis), Boolean(options.enabled), scaleCode(options.scale || "linear"));
    this.requestRender();
    return this;
  }

  set_axis_state(axis, options = {}) {
    return this.setAxisState(axis, options);
  }

  setAxisLabel(axis, label = "") {
    this._assertReady();
    this.wasm.setAxisLabel(axisCode(axis), label);
    this.requestRender();
    return this;
  }

  set_axis_label(axis, label = "") {
    return this.setAxisLabel(axis, label);
  }

  setAxisFormat(axis, format = "") {
    this._assertReady();
    this.wasm.setAxisFormat(axisCode(axis), format);
    this.requestRender();
    return this;
  }

  set_axis_format(axis, format = "") {
    return this.setAxisFormat(axis, format);
  }

  setAxisTicks(axis, values, options = {}) {
    this._assertReady();
    this.wasm.setAxisTicks(axisCode(axis), values, options.labels || [], Boolean(options.keepDefault ?? options.keep_default));
    this.requestRender();
    return this;
  }

  set_axis_ticks(axis, values, options = {}) {
    return this.setAxisTicks(axis, values, options);
  }

  clearAxisTicks(axis) {
    this._assertReady();
    this.wasm.clearAxisTicks(axisCode(axis));
    this.requestRender();
    return this;
  }

  clear_axis_ticks(axis) {
    return this.clearAxisTicks(axis);
  }

  setAxisLimitsConstraints(axis, minValue, maxValue, options = {}) {
    this._assertReady();
    this.wasm.setAxisLimitsConstraints(axisCode(axis), options.enabled !== false, minValue, maxValue);
    this.requestRender();
    return this;
  }

  set_axis_limits_constraints(axis, minValue, maxValue, options = {}) {
    return this.setAxisLimitsConstraints(axis, minValue, maxValue, options);
  }

  setAxisZoomConstraints(axis, minZoom, maxZoom, options = {}) {
    this._assertReady();
    this.wasm.setAxisZoomConstraints(axisCode(axis), options.enabled !== false, minZoom, maxZoom);
    this.requestRender();
    return this;
  }

  set_axis_zoom_constraints(axis, minZoom, maxZoom, options = {}) {
    return this.setAxisZoomConstraints(axis, minZoom, maxZoom, options);
  }

  setAxisLink(axis, targetAxis = null) {
    this._assertReady();
    this.wasm.setAxisLink(axisCode(axis), targetAxis == null ? null : axisCode(targetAxis));
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
    this.requestRender();
    return this;
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
    this.wrapper.remove();
  }
}

export async function createPlot(target, options = {}) {
  const plot = new WebPlot(target, options);
  return plot.init();
}

export const Plot = WebPlot;
