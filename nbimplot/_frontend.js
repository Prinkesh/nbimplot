const SERIES_COLORS = [
  "#1d4ed8",
  "#ea580c",
  "#16a34a",
  "#be123c",
  "#7c3aed",
  "#0f766e",
  "#ca8a04",
  "#0369a1",
];

const PRIMITIVE_KIND_CODES = Object.freeze({
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

const PLOT_FLAGS = Object.freeze({
  NO_LEGEND: 1 << 0,
  NO_MENUS: 1 << 1,
  NO_BOX_SELECT: 1 << 2,
  NO_MOUSE_POS: 1 << 3,
  CROSSHAIRS: 1 << 4,
  EQUAL: 1 << 5,
});

const SUBPLOT_FLAGS = Object.freeze({
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

const LABEL_SEP = "\x1f";
const PIE_FMT_SEP = "\x1e";
const HEATMAP_META_SEP = "\x1d";

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function rgbaToCss(r, g, b, a) {
  const rr = Math.round(clamp(Number(r), 0, 1) * 255);
  const gg = Math.round(clamp(Number(g), 0, 1) * 255);
  const bb = Math.round(clamp(Number(b), 0, 1) * 255);
  const aa = clamp(Number(a), 0, 1);
  return `rgba(${rr}, ${gg}, ${bb}, ${aa})`;
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

function toArrayBuffer(value) {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  if (ArrayBuffer.isView(value)) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  return null;
}

let modulePromise = null;
let moduleAssetKey = "";
let lastWasmLoadError = "";
const WASM_ASSET_EVENT = "nbimplot:wasm-assets-ready";
const globalWasmAssets = {
  state: "empty", // empty | requesting | ready | error
  jsSource: "",
  wasmBinary: null,
};

async function loadWasmModule(wasmJsSource, wasmBinary) {
  if (
    typeof wasmJsSource !== "string" ||
    wasmJsSource.length === 0 ||
    !(wasmBinary instanceof Uint8Array) ||
    wasmBinary.byteLength === 0
  ) {
    return null;
  }

  const assetKey = `${wasmJsSource.length}:${wasmBinary.byteLength}`;
  if (modulePromise && moduleAssetKey === assetKey) {
    return modulePromise;
  }

  moduleAssetKey = assetKey;
  modulePromise = (async () => {
    const errors = [];
    const finalizeError = () => {
      lastWasmLoadError =
        errors.length > 0
          ? errors.join(" | ")
          : "Unknown WASM module loading failure.";
      modulePromise = null;
      moduleAssetKey = "";
      return null;
    };

    const moduleOptions = {
      wasmBinary,
      locateFile: (path) => path,
    };

    const fromImported = async (imported) => {
      const factory = imported.default || imported.createNbImPlotModule;
      if (typeof factory !== "function") {
        throw new Error("WASM module factory is missing.");
      }
      return await factory(moduleOptions);
    };

    // Strategy 1: blob URL module import
    {
      let blobUrl = "";
      try {
        const blob = new Blob([wasmJsSource], { type: "text/javascript" });
        blobUrl = URL.createObjectURL(blob);
        const imported = await import(blobUrl);
        lastWasmLoadError = "";
        return await fromImported(imported);
      } catch (error) {
        errors.push(`blob import: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        if (blobUrl) {
          URL.revokeObjectURL(blobUrl);
        }
      }
    }

    // Strategy 2: data URL module import (for environments that block blob:)
    try {
      const encoded = encodeURIComponent(wasmJsSource);
      const dataUrl = `data:text/javascript;charset=utf-8,${encoded}`;
      const imported = await import(dataUrl);
      lastWasmLoadError = "";
      return await fromImported(imported);
    } catch (error) {
      errors.push(`data-url import: ${error instanceof Error ? error.message : String(error)}`);
    }

    // Strategy 3: transformed non-module eval fallback
    try {
      const transformed = wasmJsSource
        .replace(/import\.meta\.url/g, "\"http://localhost/\"")
        .replace(/export\s+default\s+createNbImPlotModule\s*;?/g, "");
      const factory = new Function(`${transformed}\nreturn createNbImPlotModule;`)();
      if (typeof factory !== "function") {
        throw new Error("Transformed loader did not yield a factory function.");
      }
      lastWasmLoadError = "";
      return await factory(moduleOptions);
    } catch (error) {
      errors.push(`eval fallback: ${error instanceof Error ? error.message : String(error)}`);
    }

    return finalizeError();
  })();

  return modulePromise;
}

class WasmCoreSession {
  constructor() {
    this.module = null;
    this.handle = 0;
    this.ready = false;
    this.lastError = "";
    this._encoder = new TextEncoder();
    this._perfPtr = 0;
  }

  async init(wasmJsSource, wasmBinary) {
    if (this.ready) {
      return true;
    }
    this.module = await loadWasmModule(wasmJsSource, wasmBinary);
    if (!this.module) {
      this.lastError = lastWasmLoadError || "Failed to load Emscripten module.";
      return false;
    }
    this.handle = this.module._nbp_create();
    this.ready = this.handle !== 0;
    if (!this.ready) {
      this.lastError = "WASM module loaded but _nbp_create returned 0.";
    } else {
      this.lastError = "";
      this._perfPtr = this.module._malloc(32);
    }
    return this.ready;
  }

  isReady() {
    return this.ready && this.module !== null && this.handle !== 0;
  }

  destroy() {
    if (!this.isReady()) {
      return;
    }
    if (this._perfPtr) {
      this.module._free(this._perfPtr);
      this._perfPtr = 0;
    }
    this.module._nbp_destroy(this.handle);
    this.handle = 0;
    this.ready = false;
  }

  setCanvas(width, height, dpr) {
    if (!this.isReady()) {
      return false;
    }
    this.module._nbp_set_canvas(
      this.handle,
      Math.max(1, width | 0),
      Math.max(1, height | 0),
      Math.max(1, Number(dpr)),
    );
    return true;
  }

  _withCString(text, fn) {
    if (!this.isReady()) {
      return false;
    }
    const encoded = this._encoder.encode(`${String(text || "")}\0`);
    const ptr = this.module._malloc(encoded.byteLength);
    if (ptr === 0) {
      return false;
    }
    this.module.HEAPU8.set(encoded, ptr);
    try {
      return fn(ptr);
    } finally {
      this.module._free(ptr);
    }
  }

  setCanvasSelector(selector) {
    if (!this.isReady()) {
      return false;
    }
    return this._withCString(selector, (ptr) => {
      const rc = this.module._nbp_set_canvas_selector(this.handle, ptr);
      return rc === 0;
    });
  }

  upsertLine(seriesToken, data, isNewSeries) {
    if (!this.isReady()) {
      return false;
    }
    const arrayBuffer = toArrayBuffer(data);
    if (!arrayBuffer || arrayBuffer.byteLength % 4 !== 0) {
      return false;
    }

    const view = new Float32Array(arrayBuffer);
    const bytes = view.byteLength;
    const ptr = this.module._malloc(bytes);
    if (ptr === 0) {
      return false;
    }
    this.module.HEAPF32.set(view, ptr >>> 2);
    this.module._nbp_line_set_data(
      this.handle,
      seriesToken >>> 0,
      ptr,
      view.length >>> 0,
      isNewSeries ? 1 : 0,
    );
    this.module._free(ptr);
    return true;
  }

  appendLineData(seriesToken, data, maxPoints) {
    if (!this.isReady()) {
      return false;
    }
    if (typeof this.module._nbp_line_append_data !== "function") {
      return false;
    }
    const arrayBuffer = toArrayBuffer(data);
    if (!arrayBuffer || arrayBuffer.byteLength % 4 !== 0) {
      return false;
    }
    const view = new Float32Array(arrayBuffer);
    if (view.length === 0) {
      return true;
    }
    const ptr = this.module._malloc(view.byteLength);
    if (ptr === 0) {
      return false;
    }
    this.module.HEAPF32.set(view, ptr >>> 2);
    const rc = this.module._nbp_line_append_data(
      this.handle,
      seriesToken >>> 0,
      ptr,
      view.length >>> 0,
      Math.max(0, Number(maxPoints || 0) | 0),
    );
    this.module._free(ptr);
    return rc === 0;
  }

  setSeriesVisible(seriesToken, visible) {
    if (!this.isReady()) {
      return false;
    }
    this.module._nbp_set_series_visible(this.handle, seriesToken >>> 0, visible ? 1 : 0);
    return true;
  }

  setSeriesName(seriesToken, name) {
    if (!this.isReady()) {
      return false;
    }
    return this._withCString(name, (ptr) => {
      return this.module._nbp_line_set_name(this.handle, seriesToken >>> 0, ptr) === 0;
    });
  }

  setSeriesSubplot(seriesToken, subplotIndex) {
    if (!this.isReady()) {
      return false;
    }
    return (
      this.module._nbp_set_series_subplot(
        this.handle,
        seriesToken >>> 0,
        Math.max(0, Number(subplotIndex) | 0),
      ) === 0
    );
  }

  setSeriesAxes(seriesToken, xAxis, yAxis) {
    if (!this.isReady()) {
      return false;
    }
    return (
      this.module._nbp_set_series_axes(
        this.handle,
        seriesToken >>> 0,
        Math.max(0, Number(xAxis) | 0),
        Math.max(0, Number(yAxis) | 0),
      ) === 0
    );
  }

  setSeriesStyle(seriesToken, style) {
    if (!this.isReady()) {
      return false;
    }
    if (typeof this.module._nbp_set_series_style !== "function") {
      return false;
    }
    const hasColor = Boolean(style?.hasColor);
    const colorR = hasColor ? Number(style?.colorR ?? 0) : 0;
    const colorG = hasColor ? Number(style?.colorG ?? 0) : 0;
    const colorB = hasColor ? Number(style?.colorB ?? 0) : 0;
    const colorA = hasColor ? Number(style?.colorA ?? 0) : 0;
    const lineWeight = Number(style?.lineWeight ?? 1.0);
    const marker = Number(style?.marker ?? -2) | 0;
    const markerSize = Number(style?.markerSize ?? 4.0);
    return (
      this.module._nbp_set_series_style(
        this.handle,
        seriesToken >>> 0,
        hasColor ? 1 : 0,
        colorR,
        colorG,
        colorB,
        colorA,
        lineWeight,
        marker,
        markerSize,
      ) === 0
    );
  }

  setPlotOptions(plotFlags, axisScaleX, axisScaleY) {
    if (!this.isReady()) {
      return false;
    }
    return (
      this.module._nbp_set_plot_options(
        this.handle,
        Math.max(0, Number(plotFlags) | 0),
        Math.max(0, Number(axisScaleX) | 0),
        Math.max(0, Number(axisScaleY) | 0),
      ) === 0
    );
  }

  setAxisState(axisIndex, enabled, scale) {
    if (!this.isReady()) {
      return false;
    }
    return (
      this.module._nbp_set_axis_state(
        this.handle,
        Math.max(0, Number(axisIndex) | 0),
        enabled ? 1 : 0,
        Math.max(0, Number(scale) | 0),
      ) === 0
    );
  }

  setAxisLabel(axisIndex, label) {
    if (!this.isReady() || typeof this.module._nbp_set_axis_label !== "function") {
      return false;
    }
    return this._withCString(label ?? "", (ptr) => {
      return this.module._nbp_set_axis_label(this.handle, Math.max(0, Number(axisIndex) | 0), ptr) === 0;
    });
  }

  setAxisFormat(axisIndex, format) {
    if (!this.isReady() || typeof this.module._nbp_set_axis_format !== "function") {
      return false;
    }
    return this._withCString(format ?? "", (ptr) => {
      return this.module._nbp_set_axis_format(this.handle, Math.max(0, Number(axisIndex) | 0), ptr) === 0;
    });
  }

  setAxisTicks(axisIndex, values, labels, keepDefault) {
    if (!this.isReady() || typeof this.module._nbp_set_axis_ticks !== "function") {
      return false;
    }
    const ticks = values instanceof Float32Array ? values : new Float32Array(0);
    let dataPtr = 0;
    if (ticks.length > 0) {
      dataPtr = this.module._malloc(ticks.byteLength);
      if (dataPtr === 0) {
        return false;
      }
      this.module.HEAPF32.set(ticks, dataPtr >>> 2);
    }
    const labelBlob = Array.isArray(labels) ? labels.map((s) => String(s)).join(LABEL_SEP) : "";
    try {
      return this._withCString(labelBlob, (labelPtr) => {
        return (
          this.module._nbp_set_axis_ticks(
            this.handle,
            Math.max(0, Number(axisIndex) | 0),
            dataPtr,
            ticks.length >>> 0,
            labelPtr,
            keepDefault ? 1 : 0,
          ) === 0
        );
      });
    } finally {
      if (dataPtr) {
        this.module._free(dataPtr);
      }
    }
  }

  clearAxisTicks(axisIndex) {
    if (!this.isReady() || typeof this.module._nbp_clear_axis_ticks !== "function") {
      return false;
    }
    return this.module._nbp_clear_axis_ticks(this.handle, Math.max(0, Number(axisIndex) | 0)) === 0;
  }

  setAxisLimitsConstraints(axisIndex, enabled, minValue, maxValue) {
    if (!this.isReady() || typeof this.module._nbp_set_axis_limits_constraints !== "function") {
      return false;
    }
    return (
      this.module._nbp_set_axis_limits_constraints(
        this.handle,
        Math.max(0, Number(axisIndex) | 0),
        enabled ? 1 : 0,
        Number(minValue || 0),
        Number(maxValue || 0),
      ) === 0
    );
  }

  setAxisZoomConstraints(axisIndex, enabled, minValue, maxValue) {
    if (!this.isReady() || typeof this.module._nbp_set_axis_zoom_constraints !== "function") {
      return false;
    }
    return (
      this.module._nbp_set_axis_zoom_constraints(
        this.handle,
        Math.max(0, Number(axisIndex) | 0),
        enabled ? 1 : 0,
        Number(minValue || 0),
        Number(maxValue || 0),
      ) === 0
    );
  }

  setAxisLink(axisIndex, targetAxis) {
    if (!this.isReady() || typeof this.module._nbp_set_axis_link !== "function") {
      return false;
    }
    return (
      this.module._nbp_set_axis_link(
        this.handle,
        Math.max(0, Number(axisIndex) | 0),
        Number(targetAxis) | 0,
      ) === 0
    );
  }

  setSubplots(rows, cols, subplotFlags) {
    if (!this.isReady()) {
      return false;
    }
    return (
      this.module._nbp_set_subplots(
        this.handle,
        Math.max(1, Number(rows) | 0),
        Math.max(1, Number(cols) | 0),
        Math.max(0, Number(subplotFlags) | 0),
      ) === 0
    );
  }

  setAlignedGroup(groupId, enabled, vertical) {
    if (!this.isReady() || typeof this.module._nbp_set_aligned_group !== "function") {
      return false;
    }
    return this._withCString(groupId ?? "", (ptr) => {
      return (
        this.module._nbp_set_aligned_group(
          this.handle,
          ptr,
          enabled ? 1 : 0,
          vertical ? 1 : 0,
        ) === 0
      );
    });
  }

  setColormap(name) {
    if (!this.isReady()) {
      return false;
    }
    if (typeof this.module._nbp_set_colormap !== "function") {
      return false;
    }
    const value = name === null || name === undefined ? "" : String(name);
    return this._withCString(value, (ptr) => this.module._nbp_set_colormap(this.handle, ptr) === 0);
  }

  setView(view) {
    if (!this.isReady()) {
      return false;
    }
    this.module._nbp_set_view(this.handle, view.xMin, view.xMax, view.yMin, view.yMax);
    return true;
  }

  setMousePos(x, y, inside) {
    if (!this.isReady()) {
      return false;
    }
    return this.module._nbp_set_mouse_pos(this.handle, x, y, inside ? 1 : 0) === 0;
  }

  setMouseButton(button, down) {
    if (!this.isReady()) {
      return false;
    }
    return this.module._nbp_set_mouse_button(this.handle, button | 0, down ? 1 : 0) === 0;
  }

  addMouseWheel(wheelX, wheelY) {
    if (!this.isReady()) {
      return false;
    }
    return this.module._nbp_add_mouse_wheel(this.handle, Number(wheelX), Number(wheelY)) === 0;
  }

  render(title) {
    if (!this.isReady()) {
      return false;
    }
    return this._withCString(title, (ptr) => this.module._nbp_render(this.handle, ptr) === 0);
  }

  getPerfStats() {
    if (!this.isReady()) {
      return null;
    }
    if (!this._perfPtr) {
      this._perfPtr = this.module._malloc(32);
      if (!this._perfPtr) {
        return null;
      }
    }
    if (this.module._nbp_get_perf_stats(this.handle, this._perfPtr) !== 0) {
      return null;
    }
    const base = this._perfPtr >>> 2;
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

  getView() {
    if (!this.isReady()) {
      return null;
    }
    const ptr = this.module._malloc(16);
    if (ptr === 0) {
      return null;
    }
    this.module._nbp_get_view(this.handle, ptr);
    const base = ptr >>> 2;
    const out = {
      xMin: this.module.HEAPF32[base],
      xMax: this.module.HEAPF32[base + 1],
      yMin: this.module.HEAPF32[base + 2],
      yMax: this.module.HEAPF32[base + 3],
      initialized: true,
    };
    this.module._free(ptr);
    return out;
  }

  autoscale() {
    if (!this.isReady()) {
      return null;
    }
    this.module._nbp_autoscale(this.handle);
    return this.getView();
  }

  buildDrawData(pixelWidth) {
    if (!this.isReady()) {
      return null;
    }
    this.module._nbp_build_draw_data(this.handle, Math.max(1, pixelWidth | 0));
    const len = this.module._nbp_get_draw_len(this.handle) >>> 0;
    if (len === 0) {
      return new Float32Array(0);
    }
    const ptr = this.module._nbp_get_draw_ptr(this.handle);
    if (!ptr) {
      return new Float32Array(0);
    }
    const start = ptr >>> 2;
    const end = start + len * 4;
    return this.module.HEAPF32.subarray(start, end);
  }

  getInteractions() {
    if (!this.isReady()) {
      return new Float32Array(0);
    }
    const len = this.module._nbp_get_interaction_len(this.handle) >>> 0;
    if (len === 0) {
      return new Float32Array(0);
    }
    const ptr = this.module._nbp_get_interaction_ptr(this.handle);
    if (!ptr) {
      return new Float32Array(0);
    }
    const start = ptr >>> 2;
    const end = start + len * 8;
    return this.module.HEAPF32.subarray(start, end);
  }

  isImPlotCompiled() {
    if (!this.isReady()) {
      return false;
    }
    return this.module._nbp_is_implot_compiled() === 1;
  }

  setImPlotEnabled(enabled) {
    if (!this.isReady()) {
      return false;
    }
    this.module._nbp_set_implot_enabled(this.handle, enabled ? 1 : 0);
    return this.module._nbp_is_implot_enabled(this.handle) === 1;
  }

  upsertPrimitive(primitiveToken, primitiveKind, payload) {
    if (!this.isReady()) {
      return false;
    }
    const token = primitiveToken >>> 0;
    const kind = primitiveKind | 0;
    if (token === 0 || kind <= 0) {
      return false;
    }

    const ints = Array.isArray(payload?.ints) ? payload.ints : [];
    const floats = Array.isArray(payload?.floats) ? payload.floats : [];
    const text = String(payload?.text || "");

    const view0 = payload?.data0 instanceof Float32Array ? payload.data0 : new Float32Array(0);
    const view1 = payload?.data1 instanceof Float32Array ? payload.data1 : new Float32Array(0);
    const view2 = payload?.data2 instanceof Float32Array ? payload.data2 : new Float32Array(0);

    const allocF32 = (view) => {
      if (!(view instanceof Float32Array) || view.length === 0) {
        return { ptr: 0, len: 0 };
      }
      const ptr = this.module._malloc(view.byteLength);
      if (ptr === 0) {
        return null;
      }
      this.module.HEAPF32.set(view, ptr >>> 2);
      return { ptr, len: view.length >>> 0 };
    };

    const b0 = allocF32(view0);
    if (b0 === null) {
      return false;
    }
    const b1 = allocF32(view1);
    if (b1 === null) {
      if (b0.ptr) this.module._free(b0.ptr);
      return false;
    }
    const b2 = allocF32(view2);
    if (b2 === null) {
      if (b1.ptr) this.module._free(b1.ptr);
      if (b0.ptr) this.module._free(b0.ptr);
      return false;
    }

    const i = new Array(8);
    for (let idx = 0; idx < 8; idx += 1) {
      i[idx] = Number.isFinite(Number(ints[idx])) ? Number(ints[idx]) | 0 : 0;
    }
    const f = new Array(8);
    for (let idx = 0; idx < 8; idx += 1) {
      const value = Number(floats[idx]);
      f[idx] = Number.isFinite(value) ? value : 0;
    }

    try {
      return this._withCString(text, (textPtr) => {
        const rc = this.module._nbp_primitive_set_data(
          this.handle,
          token,
          kind,
          b0.ptr,
          b0.len,
          b1.ptr,
          b1.len,
          b2.ptr,
          b2.len,
          i[0],
          i[1],
          i[2],
          i[3],
          i[4],
          i[5],
          i[6],
          i[7],
          f[0],
          f[1],
          f[2],
          f[3],
          f[4],
          f[5],
          f[6],
          f[7],
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

  removePrimitive(primitiveToken) {
    if (!this.isReady()) {
      return false;
    }
    return this.module._nbp_primitive_remove(this.handle, primitiveToken >>> 0) === 0;
  }

  setPrimitiveVisible(primitiveToken, visible) {
    if (!this.isReady() || typeof this.module._nbp_primitive_set_visible !== "function") {
      return false;
    }
    return (
      this.module._nbp_primitive_set_visible(this.handle, primitiveToken >>> 0, visible ? 1 : 0) === 0
    );
  }
}

class PlotRuntime {
  constructor({ model, el }) {
    this.model = model;
    this.el = el;
    this.disposed = false;
    this.strictWasm = Boolean(this.model.get("strict_wasm") ?? true);
    this.webgl2Probe = this._probeWebGL2Support();

    this.series = new Map();
    this.seriesOrder = [];
    this.primitives = new Map();
    this.primitiveOrder = [];
    this.primitiveTokenById = new Map();
    this.nextPrimitiveToken = 1;
    this.colorBySlot = [];
    this.nextColorIndex = 0;
    this.nextSeriesToken = 1;

    this.view = {
      xMin: 0,
      xMax: 1,
      yMin: -1,
      yMax: 1,
      initialized: false,
    };

    this.mouse = {
      x: 0,
      y: 0,
      inside: false,
    };

    this.drag = {
      active: false,
      lastX: 0,
      lastY: 0,
    };

    this.boxZoom = {
      active: false,
      startX: 0,
      startY: 0,
      endX: 0,
      endY: 0,
    };

    this.rightButton = {
      down: false,
      startX: 0,
      startY: 0,
      moved: false,
      region: "outside",
    };

    this.cssWidth = 900;
    this.cssHeight = 450;
    this.dpr = Math.max(1, window.devicePixelRatio || 1);
    this.rafId = 0;
    this.dirty = true;

    this.wasm = new WasmCoreSession();
    this.wasmReady = false;
    this.wasmStatus = "loading";
    this.wasmError = "";
    this.wasmInitInFlight = false;
    this.wasmJsSource = "";
    this.wasmBinary = null;
    this.implotCompiled = false;
    this.implotEnabled = false;
    this.implotEnableRetries = 0;
    this.implotEnableMaxRetries = 80;
    this.implotEnableRetryTimer = 0;
    this.initialAutoFitActive = true;
    this.initialAutoFitTimer = 0;
    this.initialAutoFitDebounceMs = 50;
    this.plotOptions = {
      plotFlags: 0,
      axisScaleX: 0,
      axisScaleY: 0,
    };
    this.colormapName = "";
    this.axisState = [
      { enabled: 1, scale: 0 }, // x1
      { enabled: 0, scale: 0 }, // x2
      { enabled: 0, scale: 0 }, // x3
      { enabled: 1, scale: 0 }, // y1
      { enabled: 0, scale: 0 }, // y2
      { enabled: 0, scale: 0 }, // y3
    ];
    this.axisLabels = new Array(6).fill("");
    this.axisFormats = new Array(6).fill("");
    this.axisTicks = new Array(6).fill(null);
    this.axisLimitsConstraints = new Array(6).fill(null);
    this.axisZoomConstraints = new Array(6).fill(null);
    this.axisLinks = new Array(6).fill(-1);
    this.subplotConfig = {
      rows: 1,
      cols: 1,
      flags: 0,
    };
    this.alignedGroup = {
      groupId: "",
      enabled: false,
      vertical: true,
    };
    this.perfReportingEnabled = false;
    this.perfReportingIntervalMs = 500;
    this.perfState = {
      fpsWindowStartMs: 0,
      fpsFrames: 0,
      fps: 0,
      lastEmitMs: 0,
      latestWasm: null,
    };
    this.lastInteractionHash = "";

    this._buildDom();
    this._bindModel();
    this._bindEvents();
    this.onWasmAssetsReady = () => {
      if (this.disposed || this.wasmReady) {
        return;
      }
      if (globalWasmAssets.state !== "ready") {
        return;
      }
      if (!this.wasmJsSource) {
        this.wasmJsSource = globalWasmAssets.jsSource;
      }
      if (!(this.wasmBinary instanceof Uint8Array)) {
        this.wasmBinary = globalWasmAssets.wasmBinary;
      }
      this._initWasm();
    };
    window.addEventListener(WASM_ASSET_EVENT, this.onWasmAssetsReady);
    this._resizeCanvas();
    this._markDirty();

    if (globalWasmAssets.state === "ready") {
      this.wasmJsSource = globalWasmAssets.jsSource;
      this.wasmBinary = globalWasmAssets.wasmBinary;
    }

    let needWasmAssets = false;
    if (globalWasmAssets.state === "empty" || globalWasmAssets.state === "error") {
      globalWasmAssets.state = "requesting";
      needWasmAssets = true;
    }

    if (typeof this.model.send === "function") {
      this.model.send({
        type: "frontend_ready",
        need_wasm_assets: needWasmAssets,
      });
    }
    this._initWasm();
  }

  _probeWebGL2Support() {
    try {
      const probeCanvas = document.createElement("canvas");
      if (!probeCanvas || typeof probeCanvas.getContext !== "function") {
        return { available: false, reason: "Canvas 2D/WebGL APIs are unavailable in this runtime." };
      }
      const webgl2 = probeCanvas.getContext("webgl2");
      if (webgl2) {
        return { available: true, reason: "" };
      }
      const webgl1 =
        probeCanvas.getContext("webgl") || probeCanvas.getContext("experimental-webgl");
      if (webgl1) {
        return { available: false, reason: "WebGL1 is available, but WebGL2 is required." };
      }
      return {
        available: false,
        reason: "Browser could not create a WebGL context (driver, GPU, or headless runtime issue).",
      };
    } catch (error) {
      return {
        available: false,
        reason: `WebGL probe failed: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  _setWasmError(reason) {
    this.wasmStatus = "error";
    this.wasmError = String(reason || "WASM runtime error.");
    this._markDirty();
  }

  _cancelInitialAutoFit() {
    if (this.initialAutoFitTimer !== 0) {
      window.clearTimeout(this.initialAutoFitTimer);
      this.initialAutoFitTimer = 0;
    }
  }

  _scheduleInitialAutoFit() {
    if (!this.initialAutoFitActive || this.disposed) {
      return;
    }
    this._cancelInitialAutoFit();
    this.initialAutoFitTimer = window.setTimeout(() => {
      this.initialAutoFitTimer = 0;
      if (this.disposed || !this.initialAutoFitActive) {
        return;
      }
      this._autoscale();
      this.initialAutoFitActive = false;
    }, this.initialAutoFitDebounceMs);
    this._markDirty();
  }

  async _initWasm() {
    if (this.wasmReady || this.wasmInitInFlight || this.disposed) {
      return;
    }
    if (this.strictWasm && !this.webgl2Probe.available) {
      const detail = this.webgl2Probe.reason ? ` ${this.webgl2Probe.reason}` : "";
      this._setWasmError(`Strict WASM mode requires WebGL2, but it is unavailable.${detail}`);
      return;
    }
    if (
      (!this.wasmJsSource || !(this.wasmBinary instanceof Uint8Array)) &&
      globalWasmAssets.state === "ready"
    ) {
      this.wasmJsSource = globalWasmAssets.jsSource;
      this.wasmBinary = globalWasmAssets.wasmBinary;
    }
    if (!this.wasmJsSource || !(this.wasmBinary instanceof Uint8Array)) {
      this.wasmStatus = "loading";
      this._markDirty();
      return;
    }

    this.wasmInitInFlight = true;
    const ready = await this.wasm.init(this.wasmJsSource, this.wasmBinary);
    this.wasmInitInFlight = false;
    if (this.disposed) {
      return;
    }
    if (!ready) {
      this._setWasmError(this.wasm.lastError || "Failed to initialize the nbimplot WASM module.");
      return;
    }

    this.wasmReady = true;
    this.wasm.setCanvas(this.cssWidth, this.cssHeight, this.dpr);
    if (!this.wasm.setCanvasSelector(`#${this.canvas.id}`)) {
      this._setWasmError("Failed to bind WASM renderer to the plot canvas.");
      return;
    }
    this._syncAllSeriesToWasm();
    this._syncAllPrimitivesToWasm();
    if (!this._syncImPlotPreference()) {
      this._scheduleImPlotEnableRetry();
      return;
    }
    this._finishWasmReady();
  }

  _syncAllSeriesToWasm() {
    if (!this.wasmReady) {
      return;
    }
    for (const seriesId of this.seriesOrder) {
      const record = this.series.get(seriesId);
      if (!record) {
        continue;
      }
      this.wasm.upsertLine(record.token, record.data, true);
      this.wasm.setSeriesName(record.token, record.name);
      this.wasm.setSeriesSubplot(record.token, record.subplotIndex || 0);
      this.wasm.setSeriesAxes(record.token, record.xAxis || 0, record.yAxis || 3);
      this._syncSeriesStyleToWasm(record);
      this.wasm.setSeriesVisible(record.token, record.visible);
    }
  }

  _syncAllPrimitivesToWasm() {
    if (!this.wasmReady) {
      return;
    }
    for (const primitiveId of this.primitiveOrder) {
      const payload = this.primitives.get(primitiveId);
      if (!payload) {
        continue;
      }
      this._syncPrimitiveToWasm(payload);
    }
  }

  _syncPrimitiveToWasm(payload) {
    if (!this.wasmReady || !payload) {
      return false;
    }
    let token = this.primitiveTokenById.get(payload.id);
    if (token == null) {
      token = this.nextPrimitiveToken >>> 0;
      this.nextPrimitiveToken += 1;
      this.primitiveTokenById.set(payload.id, token);
    }

    const kindCode = PRIMITIVE_KIND_CODES[payload.kind];
    if (!kindCode) {
      return false;
    }

    const buffers = Array.isArray(payload.buffers) ? payload.buffers : [];
    const data0 = buffers[0] instanceof Float32Array ? buffers[0] : new Float32Array(0);
    const data1 = buffers[1] instanceof Float32Array ? buffers[1] : new Float32Array(0);
    const data2 = buffers[2] instanceof Float32Array ? buffers[2] : new Float32Array(0);

    const ints = [
      payload.hasX ? 1 : 0,
      0,
      0,
      Math.max(1, Number(payload.version || 1) | 0),
      Math.min(2, Math.max(0, Number(payload.xAxis || 0) | 0)),
      Math.min(5, Math.max(3, Number(payload.yAxis || 3) | 0)),
      0,
      Math.max(0, Number(payload.subplotIndex || 0) | 0),
    ];
    const floats = [0, 0, 0, 0, 0, 0, 0, 0];
    let text = String(payload.name || "");
    const labelFormatOrDefault = (value, fallback) =>
      value === null || value === undefined ? fallback : String(value);

    switch (payload.kind) {
      case "bars":
        floats[1] = Number(payload.barWidth || 0.67);
        break;
      case "bar_groups":
        ints[1] = Number(payload.itemCount || 0) | 0;
        ints[2] = Number(payload.groupCount || 0) | 0;
        floats[1] = Number(payload.groupSize || 0.67);
        floats[2] = Number(payload.shift || 0.0);
        if (Array.isArray(payload.labels) && payload.labels.length > 0) {
          text = payload.labels.map((item) => String(item)).join(LABEL_SEP);
        }
        break;
      case "bars_h":
        floats[2] = Number(payload.barHeight || 0.67);
        break;
      case "shaded":
        floats[3] = Number(payload.alpha || 0.2);
        break;
      case "inf_lines":
        ints[1] = String(payload.axis || "x").toLowerCase() === "y" ? 1 : 0;
        break;
      case "error_bars":
        ints[1] = payload.asymmetric ? 1 : 0;
        break;
      case "error_bars_h":
        ints[1] = payload.asymmetric ? 1 : 0;
        break;
      case "histogram2d":
        ints[1] = Number(payload.rows || 0) | 0;
        ints[2] = Number(payload.cols || 0) | 0;
        ints[3] = Math.max(0, Number(payload.heatmapFlags || 0) | 0);
        ints[0] = payload.showColorbar ? 1 : 0;
        ints[6] = Math.max(0, Number(payload.colorbarFlags || 0) | 0);
        floats[0] =
          payload.scaleMin != null && Number.isFinite(Number(payload.scaleMin))
            ? Number(payload.scaleMin)
            : Number.NaN;
        floats[1] =
          payload.scaleMax != null && Number.isFinite(Number(payload.scaleMax))
            ? Number(payload.scaleMax)
            : Number.NaN;
        text = `${labelFormatOrDefault(payload.labelFmt, "%.0f")}${HEATMAP_META_SEP}${
          payload.colorbarLabel === null || payload.colorbarLabel === undefined
            ? ""
            : String(payload.colorbarLabel)
        }${HEATMAP_META_SEP}${labelFormatOrDefault(payload.colorbarFormat, "%g")}`;
        break;
      case "heatmap":
        ints[1] = Number(payload.rows || 0) | 0;
        ints[2] = Number(payload.cols || 0) | 0;
        ints[3] = Math.max(0, Number(payload.heatmapFlags || 0) | 0);
        ints[0] = payload.showColorbar ? 1 : 0;
        ints[6] = Math.max(0, Number(payload.colorbarFlags || 0) | 0);
        floats[0] =
          payload.scaleMin != null && Number.isFinite(Number(payload.scaleMin))
            ? Number(payload.scaleMin)
            : Number.NaN;
        floats[1] =
          payload.scaleMax != null && Number.isFinite(Number(payload.scaleMax))
            ? Number(payload.scaleMax)
            : Number.NaN;
        text = `${labelFormatOrDefault(payload.labelFmt, "%.2f")}${HEATMAP_META_SEP}${
          payload.colorbarLabel === null || payload.colorbarLabel === undefined
            ? ""
            : String(payload.colorbarLabel)
        }${HEATMAP_META_SEP}${labelFormatOrDefault(payload.colorbarFormat, "%g")}`;
        break;
      case "image":
        ints[0] = Math.max(0, Number(payload.imageFlags || 0) | 0);
        ints[1] = Number(payload.rows || 0) | 0;
        ints[2] = Number(payload.cols || 0) | 0;
        ints[3] = Math.max(1, Number(payload.version || 1) | 0);
        ints[6] = Math.max(1, Number(payload.channels || 1) | 0);
        floats[0] = Number.isFinite(Number(payload.boundsXMin))
          ? Number(payload.boundsXMin)
          : 0.0;
        floats[1] = Number.isFinite(Number(payload.boundsXMax))
          ? Number(payload.boundsXMax)
          : Number(payload.cols || 0);
        floats[2] = Number.isFinite(Number(payload.boundsYMin))
          ? Number(payload.boundsYMin)
          : 0.0;
        floats[3] = Number.isFinite(Number(payload.boundsYMax))
          ? Number(payload.boundsYMax)
          : Number(payload.rows || 0);
        floats[4] = Number.isFinite(Number(payload.uv0X)) ? Number(payload.uv0X) : 0.0;
        floats[5] = Number.isFinite(Number(payload.uv0Y)) ? Number(payload.uv0Y) : 0.0;
        floats[6] = Number.isFinite(Number(payload.uv1X)) ? Number(payload.uv1X) : 1.0;
        floats[7] = Number.isFinite(Number(payload.uv1Y)) ? Number(payload.uv1Y) : 1.0;
        break;
      case "tag_x":
        floats[4] = Number(payload.value || 0.0);
        ints[1] = payload.roundValue ? 1 : 0;
        text = labelFormatOrDefault(payload.labelFmt, "%g");
        break;
      case "tag_y":
        floats[5] = Number(payload.value || 0.0);
        ints[1] = payload.roundValue ? 1 : 0;
        text = labelFormatOrDefault(payload.labelFmt, "%g");
        break;
      case "colormap_slider":
        floats[4] = Number(payload.value || 0.0);
        text = `${String(payload.label || "Colormap")}${HEATMAP_META_SEP}${labelFormatOrDefault(
          payload.labelFmt,
          "",
        )}`;
        break;
      case "colormap_button":
        floats[4] = Number(payload.x || 0.0);
        floats[5] = Number(payload.y || 0.0);
        text = String(payload.label || "Colormap");
        break;
      case "colormap_selector":
        text = String(payload.label || "Colormap");
        break;
      case "drag_drop_plot":
        ints[0] = payload.sourceEnabled ? 1 : 0;
        ints[1] = payload.targetEnabled ? 1 : 0;
        break;
      case "drag_drop_axis":
        ints[0] = payload.sourceEnabled ? 1 : 0;
        ints[1] = payload.targetEnabled ? 1 : 0;
        ints[2] = Math.max(0, Number(payload.axisCode || 0) | 0);
        break;
      case "drag_drop_legend":
        ints[1] = payload.targetEnabled ? 1 : 0;
        break;
      case "pie_chart":
        floats[4] = Number(payload.x || 0.0);
        floats[5] = Number(payload.y || 0.0);
        floats[6] = Number(payload.radius || 1.0);
        floats[7] = Number(payload.angle0 || 90.0);
        {
          const labelFmt = String(payload.labelFmt || "%.1f");
          const labels = Array.isArray(payload.labels)
            ? payload.labels.map((item) => String(item))
            : [];
          text = `${labelFmt}${PIE_FMT_SEP}${labels.join(LABEL_SEP)}`;
        }
        break;
      case "text":
        floats[4] = Number(payload.x || 0.0);
        floats[5] = Number(payload.y || 0.0);
        text = String(payload.label || "");
        break;
      case "annotation":
        floats[4] = Number(payload.x || 0.0);
        floats[5] = Number(payload.y || 0.0);
        floats[6] = Number(payload.offsetX || 8.0);
        floats[7] = Number(payload.offsetY || -8.0);
        text = String(payload.label || "");
        break;
      case "drag_line_x":
        floats[4] = Number(payload.value || 0.0);
        floats[6] = Number(payload.thickness || 1.0);
        text = String(payload.name || "drag_x");
        break;
      case "drag_line_y":
        floats[5] = Number(payload.value || 0.0);
        floats[6] = Number(payload.thickness || 1.0);
        text = String(payload.name || "drag_y");
        break;
      case "drag_point":
        floats[4] = Number(payload.x || 0.0);
        floats[5] = Number(payload.y || 0.0);
        floats[6] = Number(payload.size || 4.0);
        text = String(payload.name || "drag_point");
        break;
      case "drag_rect":
        floats[4] = Number(payload.x1 || 0.0);
        floats[5] = Number(payload.y1 || 0.0);
        floats[6] = Number(payload.x2 || 1.0);
        floats[7] = Number(payload.y2 || 1.0);
        text = String(payload.name || "drag_rect");
        break;
      default:
        break;
    }

    const ok = this.wasm.upsertPrimitive(token, kindCode, {
      data0,
      data1,
      data2,
      ints,
      floats,
      text,
    });
    if (!ok) {
      return false;
    }
    if (payload.hidden !== undefined && payload.hidden !== null) {
      this.wasm.setPrimitiveVisible(token, !Boolean(payload.hidden));
    }
    return true;
  }

  _syncImPlotPreference() {
    if (!this.wasmReady) {
      return false;
    }
    const canvasAttached =
      Boolean(this.canvas && this.canvas.isConnected) &&
      Boolean(this.canvas.id) &&
      document.getElementById(this.canvas.id) === this.canvas;
    if (!canvasAttached) {
      this.implotEnabled = false;
      this.wasmError =
        "Unable to enable ImPlot in the WASM core (plot canvas is not attached to the document yet).";
      return false;
    }
    this.implotCompiled = this.wasm.isImPlotCompiled();
    if (!this.implotCompiled) {
      this.implotEnabled = false;
      this.wasmError =
        "WASM module was built without ImPlot. Rebuild with NBIMPLOT_WITH_IMPLOT=ON.";
      return false;
    }
    this.implotEnabled = this.wasm.setImPlotEnabled(true);
    if (!this.implotEnabled) {
      const webglUnavailable = !this.webgl2Probe.available;
      const detail = webglUnavailable
        ? ` ${this.webgl2Probe.reason || "WebGL2 is unavailable in this browser/runtime."}`
        : "";
      this.wasmError = `Unable to enable ImPlot in the WASM core.${detail}`;
      return false;
    }
    if (this.implotEnableRetryTimer !== 0) {
      window.clearTimeout(this.implotEnableRetryTimer);
      this.implotEnableRetryTimer = 0;
    }
    if (this.initialAutoFitTimer !== 0) {
      window.clearTimeout(this.initialAutoFitTimer);
      this.initialAutoFitTimer = 0;
    }
    this.implotEnableRetries = 0;
    if (this.legend) {
      this.legend.style.display = "none";
      this.legend.replaceChildren();
    }
    return true;
  }

  _finishWasmReady() {
    this._syncSubplotsToWasm();
    this._syncAlignedGroupToWasm();
    this._syncPlotOptionsToWasm();
    this._syncAxisStateToWasm();
    this._syncAxisConfigToWasm();
    this._syncColormapToWasm();
    this.wasmStatus = "ready";

    if (this.view.initialized) {
      this.wasm.setView(this.view);
    } else if (this._hasVisibleSeries() || this.primitives.size > 0) {
      const view = this.wasm.autoscale();
      if (view) {
        this.view = view;
      }
    }
    if (this.initialAutoFitActive && (this._hasVisibleSeries() || this.primitives.size > 0)) {
      this._scheduleInitialAutoFit();
    }

    this._markDirty();
  }

  _scheduleImPlotEnableRetry() {
    if (this.disposed || !this.wasmReady || this.implotEnabled) {
      return;
    }
    if (this.implotEnableRetryTimer !== 0) {
      return;
    }
    if (this.implotEnableRetries >= this.implotEnableMaxRetries) {
      this.wasmStatus = "error";
      this._markDirty();
      return;
    }
    const retryIndex = this.implotEnableRetries;
    this.implotEnableRetries += 1;
    const delayMs = Math.min(500, 25 * (retryIndex + 1));
    this.wasmStatus = "loading";
    this.implotEnableRetryTimer = window.setTimeout(() => {
      this.implotEnableRetryTimer = 0;
      if (this.disposed || !this.wasmReady || this.implotEnabled) {
        return;
      }
      if (this._syncImPlotPreference()) {
        this._finishWasmReady();
        return;
      }
      this._scheduleImPlotEnableRetry();
      this._markDirty();
    }, delayMs);
    this._markDirty();
  }

  _syncPlotOptionsToWasm() {
    if (!this.wasmReady) {
      return false;
    }
    return this.wasm.setPlotOptions(
      this.plotOptions.plotFlags,
      Math.min(2, Math.max(0, this.plotOptions.axisScaleX | 0)),
      Math.min(2, Math.max(0, this.plotOptions.axisScaleY | 0)),
    );
  }

  _syncAxisStateToWasm() {
    if (!this.wasmReady) {
      return false;
    }
    let ok = true;
    for (let axisIndex = 0; axisIndex < this.axisState.length; axisIndex += 1) {
      const state = this.axisState[axisIndex] || { enabled: 0, scale: 0 };
      const axisOk = this.wasm.setAxisState(
        axisIndex,
        state.enabled !== 0,
        Math.min(2, Math.max(0, state.scale | 0)),
      );
      ok = ok && axisOk;
    }
    return ok;
  }

  _syncAxisConfigToWasm() {
    if (!this.wasmReady) {
      return false;
    }
    let ok = true;
    for (let axisIndex = 0; axisIndex < 6; axisIndex += 1) {
      ok = this.wasm.setAxisLabel(axisIndex, this.axisLabels[axisIndex] || "") && ok;
      ok = this.wasm.setAxisFormat(axisIndex, this.axisFormats[axisIndex] || "") && ok;
      const ticks = this.axisTicks[axisIndex];
      if (ticks && ticks.values instanceof Float32Array) {
        ok = this.wasm.setAxisTicks(
          axisIndex,
          ticks.values,
          Array.isArray(ticks.labels) ? ticks.labels : [],
          Boolean(ticks.keepDefault),
        ) && ok;
      } else {
        ok = this.wasm.clearAxisTicks(axisIndex) && ok;
      }
      const lim = this.axisLimitsConstraints[axisIndex];
      if (lim && lim.enabled) {
        ok = this.wasm.setAxisLimitsConstraints(axisIndex, true, lim.min, lim.max) && ok;
      } else {
        ok = this.wasm.setAxisLimitsConstraints(axisIndex, false, 0, 0) && ok;
      }
      const zoom = this.axisZoomConstraints[axisIndex];
      if (zoom && zoom.enabled) {
        ok = this.wasm.setAxisZoomConstraints(axisIndex, true, zoom.min, zoom.max) && ok;
      } else {
        ok = this.wasm.setAxisZoomConstraints(axisIndex, false, 0, 0) && ok;
      }
      ok = this.wasm.setAxisLink(axisIndex, Number(this.axisLinks[axisIndex] ?? -1) | 0) && ok;
    }
    return ok;
  }

  _syncSubplotsToWasm() {
    if (!this.wasmReady) {
      return false;
    }
    return this.wasm.setSubplots(
      this.subplotConfig.rows,
      this.subplotConfig.cols,
      this.subplotConfig.flags,
    );
  }

  _syncAlignedGroupToWasm() {
    if (!this.wasmReady) {
      return false;
    }
    return this.wasm.setAlignedGroup(
      this.alignedGroup.groupId,
      this.alignedGroup.enabled,
      this.alignedGroup.vertical,
    );
  }

  _syncColormapToWasm() {
    if (!this.wasmReady) {
      return false;
    }
    return this.wasm.setColormap(this.colormapName);
  }

  _buildDom() {
    this.el.innerHTML = "";

    this.wrapper = document.createElement("div");
    this.wrapper.style.position = "relative";
    this.wrapper.style.border = "1px solid #d4d4d8";
    this.wrapper.style.borderRadius = "8px";
    this.wrapper.style.background = "#ffffff";
    this.wrapper.style.boxSizing = "border-box";
    this.wrapper.style.overflow = "hidden";

    this.canvas = document.createElement("canvas");
    this.canvas.id = `nbimplot-canvas-${Math.random().toString(36).slice(2)}`;
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    this.canvas.style.display = "block";
    this.canvas.style.touchAction = "none";

    this.overlay = document.createElement("canvas");
    this.overlay.style.position = "absolute";
    this.overlay.style.left = "0";
    this.overlay.style.top = "0";
    this.overlay.style.width = "100%";
    this.overlay.style.height = "100%";
    this.overlay.style.pointerEvents = "none";
    this.overlay.style.display = "block";

    this.legend = document.createElement("div");
    this.legend.style.position = "absolute";
    this.legend.style.top = "8px";
    this.legend.style.right = "8px";
    this.legend.style.background = "rgba(255, 255, 255, 0.92)";
    this.legend.style.border = "1px solid #d4d4d8";
    this.legend.style.borderRadius = "6px";
    this.legend.style.padding = "6px";
    this.legend.style.display = "flex";
    this.legend.style.flexDirection = "column";
    this.legend.style.gap = "4px";
    this.legend.style.maxWidth = "220px";
    this.legend.style.fontFamily = "ui-sans-serif, system-ui, sans-serif";
    this.legend.style.fontSize = "12px";

    this.contextMenu = document.createElement("div");
    this.contextMenu.style.position = "absolute";
    this.contextMenu.style.minWidth = "140px";
    this.contextMenu.style.background = "rgba(255, 255, 255, 0.98)";
    this.contextMenu.style.border = "1px solid #d4d4d8";
    this.contextMenu.style.borderRadius = "6px";
    this.contextMenu.style.boxShadow = "0 8px 24px rgba(24, 24, 27, 0.18)";
    this.contextMenu.style.padding = "4px";
    this.contextMenu.style.display = "none";
    this.contextMenu.style.zIndex = "20";
    this.contextMenu.style.fontFamily = "ui-sans-serif, system-ui, sans-serif";
    this.contextMenu.style.fontSize = "12px";

    this.wrapper.appendChild(this.canvas);
    this.wrapper.appendChild(this.overlay);
    this.wrapper.appendChild(this.legend);
    this.wrapper.appendChild(this.contextMenu);
    this.el.appendChild(this.wrapper);

    const maybeCtx = this.overlay.getContext("2d");
    if (!maybeCtx) {
      throw new Error("2D context is unavailable for nbimplot overlay.");
    }
    this.ctx = maybeCtx;

    if (typeof ResizeObserver !== "undefined") {
      this.resizeObserver = new ResizeObserver(() => this._resizeCanvas());
      this.resizeObserver.observe(this.el);
    } else {
      this.resizeObserver = null;
    }
  }

  _bindModel() {
    this.onWidthChange = () => this._resizeCanvas();
    this.onHeightChange = () => this._resizeCanvas();
    this.onTitleChange = () => this._markDirty();
    this.onCustomMessage = (content, buffers) => this._handleMessage(content, buffers);

    this.model.on("change:width", this.onWidthChange);
    this.model.on("change:height", this.onHeightChange);
    this.model.on("change:title", this.onTitleChange);
    this.model.on("msg:custom", this.onCustomMessage);
  }

  _bindEvents() {
    this.onMouseDown = (event) => this._handleMouseDown(event);
    this.onMouseMove = (event) => this._handleMouseMove(event);
    this.onMouseUp = (event) => this._handleMouseUp(event);
    this.onMouseLeave = () => this._handleMouseLeave();
    this.onWheel = (event) => this._handleWheel(event);
    this.onContextMenu = (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (this.implotEnabled && this.wasmReady) {
        const pos = this._pointerPosition(event);
        this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
        this._flushInputFrame();
      }
    };
    this.onDoubleClick = (event) => {
      const pos = this._pointerPosition(event);
      if (!this._insidePlot(pos)) {
        return;
      }
      event.preventDefault();
      this._cancelInitialAutoFit();
      this.initialAutoFitActive = false;
      if (this.implotEnabled && this.wasmReady) {
        this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      }
      this._autoscale();
    };
    this.onKeyDown = (event) => {
      if (event.key === "Escape") {
        this._hideContextMenu();
        if (this.boxZoom.active) {
          this.boxZoom.active = false;
          this.rightButton.down = false;
          this._markDirty();
        }
      }
    };
    this.onWindowMouseDown = (event) => {
      if (event.button !== 0) {
        return;
      }
      if (
        this.contextMenu &&
        this.contextMenu.style.display !== "none" &&
        this.contextMenu.contains(event.target)
      ) {
        return;
      }
      this._hideContextMenu();
    };
    this.onWindowResize = () => this._resizeCanvas();

    this.canvas.addEventListener("mousedown", this.onMouseDown);
    this.canvas.addEventListener("mousemove", this.onMouseMove);
    this.canvas.addEventListener("mouseleave", this.onMouseLeave);
    this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
    this.canvas.addEventListener("contextmenu", this.onContextMenu);
    this.canvas.addEventListener("dblclick", this.onDoubleClick);
    window.addEventListener("mousemove", this.onMouseMove);
    window.addEventListener("mouseup", this.onMouseUp);
    window.addEventListener("mousedown", this.onWindowMouseDown);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("resize", this.onWindowResize);
  }

  _handleMessage(content, buffers) {
    if (this.disposed || !content || typeof content.type !== "string") {
      return;
    }

    switch (content.type) {
      case "wasm_assets":
        this._handleWasmAssetsMessage(content, buffers);
        break;
      case "wasm_assets_missing":
        this._handleWasmAssetsMissingMessage(content);
        break;
      case "line":
        this._handleLineMessage(content, buffers);
        break;
      case "set_data":
        this._handleSetDataMessage(content, buffers);
        break;
      case "append_data":
        this._handleAppendDataMessage(content, buffers);
        break;
      case "series_style":
        this._handleSeriesStyleMessage(content);
        break;
      case "set_view":
        this._handleSetViewMessage(content);
        break;
      case "plot_options":
        this._handlePlotOptionsMessage(content);
        break;
      case "axis_state":
        this._handleAxisStateMessage(content);
        break;
      case "axis_label":
        this._handleAxisLabelMessage(content);
        break;
      case "axis_format":
        this._handleAxisFormatMessage(content);
        break;
      case "axis_ticks":
        this._handleAxisTicksMessage(content, buffers);
        break;
      case "axis_ticks_clear":
        this._handleAxisTicksClearMessage(content);
        break;
      case "axis_limits_constraints":
        this._handleAxisLimitsConstraintsMessage(content);
        break;
      case "axis_zoom_constraints":
        this._handleAxisZoomConstraintsMessage(content);
        break;
      case "axis_link":
        this._handleAxisLinkMessage(content);
        break;
      case "subplots_config":
        this._handleSubplotsConfigMessage(content);
        break;
      case "aligned_group":
        this._handleAlignedGroupMessage(content);
        break;
      case "colormap":
        this._handleColormapMessage(content);
        break;
      case "autoscale":
        this._autoscale();
        break;
      case "primitive_add":
        this._handlePrimitiveAddMessage(content, buffers);
        break;
      case "primitive_remove":
        this._handlePrimitiveRemoveMessage(content);
        break;
      case "render":
        this._markDirty();
        break;
      case "set_perf_reporting":
        this._handleSetPerfReportingMessage(content);
        break;
      case "dispose":
        this.dispose();
        break;
      default:
        break;
    }
  }

  _decodeFloat32Buffer(buffers) {
    if (!buffers || buffers.length === 0) {
      return null;
    }
    const buffer = toArrayBuffer(buffers[0]);
    if (!buffer || buffer.byteLength % 4 !== 0) {
      return null;
    }
    return new Float32Array(buffer);
  }

  _decodeUint8Buffer(buffers) {
    if (!buffers || buffers.length === 0) {
      return null;
    }
    const buffer = toArrayBuffer(buffers[0]);
    if (!buffer || buffer.byteLength === 0) {
      return null;
    }
    return new Uint8Array(buffer);
  }

  _handleWasmAssetsMessage(content, buffers) {
    const jsSource = typeof content.wasm_js_source === "string" ? content.wasm_js_source : "";
    const wasmBinary = this._decodeUint8Buffer(buffers);
    if (!jsSource || !(wasmBinary instanceof Uint8Array)) {
      this.wasmStatus = "error";
      this.wasmError = "Invalid WASM asset payload from Python.";
      globalWasmAssets.state = "error";
      this._markDirty();
      return;
    }
    this.wasmJsSource = jsSource;
    this.wasmBinary = wasmBinary;
    globalWasmAssets.state = "ready";
    globalWasmAssets.jsSource = jsSource;
    globalWasmAssets.wasmBinary = wasmBinary;
    window.dispatchEvent(new CustomEvent(WASM_ASSET_EVENT));
    this._initWasm();
  }

  _handleWasmAssetsMissingMessage(content) {
    this.wasmStatus = "error";
    this.wasmError = String(content.reason || "WASM assets are missing.");
    globalWasmAssets.state = "error";
    this._markDirty();
  }

  _parseSeriesStyleMessage(content) {
    const hasColor = Boolean(content.has_color);
    const colorR = hasColor && Number.isFinite(Number(content.color_r)) ? Number(content.color_r) : 0;
    const colorG = hasColor && Number.isFinite(Number(content.color_g)) ? Number(content.color_g) : 0;
    const colorB = hasColor && Number.isFinite(Number(content.color_b)) ? Number(content.color_b) : 0;
    const colorA = hasColor && Number.isFinite(Number(content.color_a)) ? Number(content.color_a) : 0;
    const rawWeight = Number(content.line_weight);
    const lineWeight = Number.isFinite(rawWeight) && rawWeight > 0 ? rawWeight : 1.0;
    const rawMarker = Number(content.marker) | 0;
    const marker = Math.max(-2, Math.min(9, rawMarker));
    const rawMarkerSize = Number(content.marker_size);
    const markerSize = Number.isFinite(rawMarkerSize) && rawMarkerSize > 0 ? rawMarkerSize : 4.0;
    return {
      hasColor,
      colorR,
      colorG,
      colorB,
      colorA,
      lineWeight,
      marker,
      markerSize,
    };
  }

  _syncSeriesStyleToWasm(record) {
    if (!this.wasmReady || !record) {
      return false;
    }
    return this.wasm.setSeriesStyle(record.token, record.style || null);
  }

  _handleLineMessage(content, buffers) {
    const data = this._decodeFloat32Buffer(buffers);
    if (!data) {
      return;
    }
    const seriesId = String(content.series_id || "");
    if (!seriesId) {
      return;
    }
    const name = String(content.name || seriesId);
    const subplotIndex = Math.max(0, Number(content.subplot_index || 0) | 0);
    const xAxis = Math.min(2, Math.max(0, Number(content.x_axis || 0) | 0));
    const yAxis = Math.min(5, Math.max(3, Number(content.y_axis || 3) | 0));
    const hidden = Boolean(content.hidden);
    const maxPoints = Math.max(0, Number(content.max_points || 0) | 0);
    const style = this._parseSeriesStyleMessage(content);

    const existing = this.series.get(seriesId);
    if (existing) {
      existing.name = name;
      existing.subplotIndex = subplotIndex;
      existing.xAxis = xAxis;
      existing.yAxis = yAxis;
      existing.style = style;
      existing.data = data;
      existing.maxPoints = maxPoints;
      existing.visible = !hidden;
      existing.version += 1;
      existing.lodCache = { key: "", points: [] };
      if (this.wasmReady) {
        this.wasm.upsertLine(existing.token, existing.data, false);
        this.wasm.setSeriesName(existing.token, existing.name);
        this.wasm.setSeriesSubplot(existing.token, existing.subplotIndex || 0);
        this.wasm.setSeriesAxes(existing.token, existing.xAxis || 0, existing.yAxis || 3);
        this._syncSeriesStyleToWasm(existing);
        this.wasm.setSeriesVisible(existing.token, existing.visible);
      }
      this._refreshLegend();
      if (this.initialAutoFitActive) {
        this._scheduleInitialAutoFit();
        return;
      }
      if (!this.view.initialized) {
        this._autoscale();
        return;
      }
      this._markDirty();
      return;
    }

    const color = SERIES_COLORS[this.nextColorIndex % SERIES_COLORS.length];
    this.nextColorIndex += 1;

    const slot = this.seriesOrder.length;
    const token = this.nextSeriesToken;
    this.nextSeriesToken += 1;

    const record = {
      id: seriesId,
      token,
      slot,
      name,
      data,
      color,
      subplotIndex,
      xAxis,
      yAxis,
      style,
      visible: !hidden,
      maxPoints,
      version: 1,
      lodCache: {
        key: "",
        points: [],
      },
    };

    this.series.set(record.id, record);
    this.seriesOrder.push(record.id);
    this.colorBySlot[slot] = color;
    this._refreshLegend();

    if (this.wasmReady) {
      this.wasm.upsertLine(record.token, record.data, true);
      this.wasm.setSeriesName(record.token, record.name);
      this.wasm.setSeriesSubplot(record.token, record.subplotIndex || 0);
      this.wasm.setSeriesAxes(record.token, record.xAxis || 0, record.yAxis || 3);
      this._syncSeriesStyleToWasm(record);
      this.wasm.setSeriesVisible(record.token, record.visible);
    }

    if (this.initialAutoFitActive) {
      this._scheduleInitialAutoFit();
      return;
    }

    if (!this.view.initialized) {
      this._autoscale();
      return;
    }

    this._markDirty();
  }

  _handleSetDataMessage(content, buffers) {
    const record = this.series.get(content.series_id);
    if (!record) {
      return;
    }

    const data = this._decodeFloat32Buffer(buffers);
    if (!data) {
      return;
    }

    record.data = data;
    record.version += 1;
    record.lodCache = { key: "", points: [] };

    if (this.wasmReady) {
      this.wasm.upsertLine(record.token, record.data, false);
    }

    if (this.initialAutoFitActive) {
      this._scheduleInitialAutoFit();
      return;
    }

    this._markDirty();
  }

  _handleAppendDataMessage(content, buffers) {
    const record = this.series.get(content.series_id);
    if (!record) {
      return;
    }
    const appended = this._decodeFloat32Buffer(buffers);
    if (!appended || appended.length === 0) {
      return;
    }
    const msgCap = Math.max(0, Number(content.max_points || 0) | 0);
    if (msgCap > 0) {
      record.maxPoints = msgCap;
    }
    const cap = Math.max(0, Number(record.maxPoints || 0) | 0);
    const total = record.data.length + appended.length;
    let merged = new Float32Array(total);
    merged.set(record.data, 0);
    merged.set(appended, record.data.length);
    if (cap > 0 && merged.length > cap) {
      merged = merged.slice(merged.length - cap);
    }
    record.data = merged;
    record.version += 1;
    record.lodCache = { key: "", points: [] };
    if (this.wasmReady) {
      this.wasm.appendLineData(record.token, appended, cap);
    }
    if (this.initialAutoFitActive) {
      this._scheduleInitialAutoFit();
      return;
    }
    this._markDirty();
  }

  _handleSeriesStyleMessage(content) {
    const record = this.series.get(content.series_id);
    if (!record) {
      return;
    }
    record.style = this._parseSeriesStyleMessage(content);
    if (this.wasmReady) {
      this._syncSeriesStyleToWasm(record);
    }
    this._refreshLegend();
    this._markDirty();
  }

  _handleSetViewMessage(content) {
    const xMin = Number(content.x_min);
    const xMax = Number(content.x_max);
    const yMin = Number(content.y_min);
    const yMax = Number(content.y_max);
    if (
      !Number.isFinite(xMin) ||
      !Number.isFinite(xMax) ||
      !Number.isFinite(yMin) ||
      !Number.isFinite(yMax) ||
      xMax <= xMin ||
      yMax <= yMin
    ) {
      return;
    }
    this.view.xMin = xMin;
    this.view.xMax = xMax;
    this.view.yMin = yMin;
    this.view.yMax = yMax;
    this.view.initialized = true;
    this._cancelInitialAutoFit();
    this.initialAutoFitActive = false;
    if (this.wasmReady) {
      this.wasm.setView(this.view);
    }
    this._markDirty();
  }

  _handlePlotOptionsMessage(content) {
    const flags = Math.max(0, Number(content.plot_flags) | 0);
    const axisScaleX = Math.max(0, Number(content.axis_scale_x) | 0);
    const axisScaleY = Math.max(0, Number(content.axis_scale_y) | 0);
    this.plotOptions.plotFlags = flags;
    this.plotOptions.axisScaleX = axisScaleX;
    this.plotOptions.axisScaleY = axisScaleY;
    this.axisState[0] = { enabled: 1, scale: axisScaleX };
    this.axisState[3] = { enabled: 1, scale: axisScaleY };
    if (this.wasmReady) {
      this._syncPlotOptionsToWasm();
    }
    this._markDirty();
  }

  _handleSubplotsConfigMessage(content) {
    const rows = Math.max(1, Number(content.rows) | 0);
    const cols = Math.max(1, Number(content.cols) | 0);
    const flags = Math.max(0, Number(content.flags) | 0);
    this.subplotConfig.rows = rows;
    this.subplotConfig.cols = cols;
    this.subplotConfig.flags = flags;
    if (this.wasmReady) {
      this._syncSubplotsToWasm();
      this._syncAllSeriesToWasm();
      this._syncAllPrimitivesToWasm();
    }
    this._markDirty();
  }

  _handleAlignedGroupMessage(content) {
    this.alignedGroup.groupId =
      content.group_id === undefined || content.group_id === null ? "" : String(content.group_id);
    this.alignedGroup.enabled = Boolean(content.enabled) && this.alignedGroup.groupId.length > 0;
    this.alignedGroup.vertical = content.vertical === undefined ? true : Boolean(content.vertical);
    if (this.wasmReady) {
      this._syncAlignedGroupToWasm();
    }
    this._markDirty();
  }

  _handleColormapMessage(content) {
    this.colormapName =
      content.name === undefined || content.name === null ? "" : String(content.name);
    if (this.wasmReady) {
      this._syncColormapToWasm();
    }
    this._markDirty();
  }

  _handleAxisStateMessage(content) {
    const axis = Math.max(0, Number(content.axis) | 0);
    if (axis < 0 || axis >= this.axisState.length) {
      return;
    }
    const enabled = axis === 0 || axis === 3 ? true : Boolean(content.enabled);
    const scale = Math.min(2, Math.max(0, Number(content.scale) | 0));
    this.axisState[axis] = { enabled: enabled ? 1 : 0, scale };
    if (axis === 0) {
      this.plotOptions.axisScaleX = scale;
    } else if (axis === 3) {
      this.plotOptions.axisScaleY = scale;
    }
    if (this.wasmReady) {
      this.wasm.setAxisState(axis, enabled, scale);
    }
    this._markDirty();
  }

  _handleAxisLabelMessage(content) {
    const axis = Math.max(0, Number(content.axis) | 0);
    if (axis < 0 || axis >= this.axisLabels.length) {
      return;
    }
    this.axisLabels[axis] =
      content.label === undefined || content.label === null ? "" : String(content.label);
    if (this.wasmReady) {
      this.wasm.setAxisLabel(axis, this.axisLabels[axis]);
    }
    this._markDirty();
  }

  _handleAxisFormatMessage(content) {
    const axis = Math.max(0, Number(content.axis) | 0);
    if (axis < 0 || axis >= this.axisFormats.length) {
      return;
    }
    this.axisFormats[axis] =
      content.format === undefined || content.format === null ? "" : String(content.format);
    if (this.wasmReady) {
      this.wasm.setAxisFormat(axis, this.axisFormats[axis]);
    }
    this._markDirty();
  }

  _handleAxisTicksMessage(content, buffers) {
    const axis = Math.max(0, Number(content.axis) | 0);
    if (axis < 0 || axis >= this.axisTicks.length) {
      return;
    }
    const values = this._decodeFloat32Buffer(buffers);
    if (!(values instanceof Float32Array)) {
      return;
    }
    const labels = Array.isArray(content.labels) ? content.labels.map((s) => String(s)) : [];
    const keepDefault = Boolean(content.keep_default);
    this.axisTicks[axis] = { values, labels, keepDefault };
    if (this.wasmReady) {
      this.wasm.setAxisTicks(axis, values, labels, keepDefault);
    }
    this._markDirty();
  }

  _handleAxisTicksClearMessage(content) {
    const axis = Math.max(0, Number(content.axis) | 0);
    if (axis < 0 || axis >= this.axisTicks.length) {
      return;
    }
    this.axisTicks[axis] = null;
    if (this.wasmReady) {
      this.wasm.clearAxisTicks(axis);
    }
    this._markDirty();
  }

  _handleAxisLimitsConstraintsMessage(content) {
    const axis = Math.max(0, Number(content.axis) | 0);
    if (axis < 0 || axis >= this.axisLimitsConstraints.length) {
      return;
    }
    const enabled = Boolean(content.enabled);
    const minValue = Number(content.min || 0);
    const maxValue = Number(content.max || 0);
    this.axisLimitsConstraints[axis] = enabled
      ? { enabled: true, min: minValue, max: maxValue }
      : { enabled: false, min: 0, max: 0 };
    if (this.wasmReady) {
      this.wasm.setAxisLimitsConstraints(axis, enabled, minValue, maxValue);
    }
    this._markDirty();
  }

  _handleAxisZoomConstraintsMessage(content) {
    const axis = Math.max(0, Number(content.axis) | 0);
    if (axis < 0 || axis >= this.axisZoomConstraints.length) {
      return;
    }
    const enabled = Boolean(content.enabled);
    const minValue = Number(content.min || 0);
    const maxValue = Number(content.max || 0);
    this.axisZoomConstraints[axis] = enabled
      ? { enabled: true, min: minValue, max: maxValue }
      : { enabled: false, min: 0, max: 0 };
    if (this.wasmReady) {
      this.wasm.setAxisZoomConstraints(axis, enabled, minValue, maxValue);
    }
    this._markDirty();
  }

  _handleAxisLinkMessage(content) {
    const axis = Math.max(0, Number(content.axis) | 0);
    if (axis < 0 || axis >= this.axisLinks.length) {
      return;
    }
    const targetAxis = Number(content.target_axis);
    this.axisLinks[axis] =
      Number.isFinite(targetAxis) && targetAxis >= 0 ? (targetAxis | 0) : -1;
    if (this.wasmReady) {
      this.wasm.setAxisLink(axis, this.axisLinks[axis]);
    }
    this._markDirty();
  }

  _handleSetPerfReportingMessage(content) {
    this.perfReportingEnabled = Boolean(content.enabled);
    const interval = Number(content.interval_ms);
    this.perfReportingIntervalMs =
      Number.isFinite(interval) && interval > 0 ? Math.max(100, interval) : 500;
    if (!this.perfReportingEnabled) {
      this.perfState.lastEmitMs = 0;
      return;
    }
    this.perfState.fpsWindowStartMs = 0;
    this.perfState.fpsFrames = 0;
    this.perfState.lastEmitMs = 0;
    this._markDirty();
  }

  _handlePrimitiveAddMessage(content, buffers) {
    const primitiveId = String(content.primitive_id || "");
    const kind = String(content.kind || "");
    if (!primitiveId || !kind) {
      return;
    }
    const prev = this.primitives.get(primitiveId);
    const version = prev ? Math.max(1, (Number(prev.version) | 0) + 1) : 1;

    const decoded = [];
    if (Array.isArray(buffers)) {
      for (const buffer of buffers) {
        const ab = toArrayBuffer(buffer);
        if (!ab || ab.byteLength % 4 !== 0) {
          decoded.push(new Float32Array(0));
        } else {
          decoded.push(new Float32Array(ab.slice(0)));
        }
      }
    }

    const payload = {
      id: primitiveId,
      kind,
      name: String(content.name || ""),
      label: String(content.label || ""),
      labels: Array.isArray(content.labels) ? content.labels.map((item) => String(item)) : [],
      axis: String(content.axis || "x"),
      axisCode: Number(content.length || 0),
      length: Number(content.length || 0),
      hasX: Boolean(content.has_x),
      hidden: Boolean(content.hidden),
      asymmetric: Boolean(content.asymmetric),
      size: Number(content.size || 2),
      barWidth: Number(content.bar_width || 0.67),
      barHeight: Number(content.bar_height || 0.67),
      groupSize: Number(content.group_size || 0.67),
      shift: Number(content.shift || 0),
      itemCount: Number(content.item_count || 0),
      groupCount: Number(content.group_count || 0),
      alpha: Number(content.alpha || 0.2),
      rows: Number(content.rows || 0),
      cols: Number(content.cols || 0),
      x: Number(content.x || 0),
      y: Number(content.y || 0),
      radius: Number(content.radius || 1),
      angle0: Number(content.angle0 || 90),
      labelFmt:
        content.label_fmt === undefined || content.label_fmt === null
          ? null
          : String(content.label_fmt),
      heatmapFlags: Math.max(0, Number(content.heatmap_flags || 0) | 0),
      imageFlags: Math.max(0, Number(content.image_flags || 0) | 0),
      scaleMin:
        content.scale_min === undefined || content.scale_min === null
          ? null
          : Number(content.scale_min),
      scaleMax:
        content.scale_max === undefined || content.scale_max === null
          ? null
          : Number(content.scale_max),
      showColorbar: Boolean(content.show_colorbar),
      colorbarLabel:
        content.colorbar_label === undefined || content.colorbar_label === null
          ? ""
          : String(content.colorbar_label),
      colorbarFormat:
        content.colorbar_format === undefined || content.colorbar_format === null
          ? "%g"
          : String(content.colorbar_format),
      colorbarFlags: Math.max(0, Number(content.colorbar_flags || 0) | 0),
      channels: Math.max(1, Number(content.channels || 1) | 0),
      boundsXMin:
        content.bounds_x_min === undefined || content.bounds_x_min === null
          ? null
          : Number(content.bounds_x_min),
      boundsXMax:
        content.bounds_x_max === undefined || content.bounds_x_max === null
          ? null
          : Number(content.bounds_x_max),
      boundsYMin:
        content.bounds_y_min === undefined || content.bounds_y_min === null
          ? null
          : Number(content.bounds_y_min),
      boundsYMax:
        content.bounds_y_max === undefined || content.bounds_y_max === null
          ? null
          : Number(content.bounds_y_max),
      uv0X:
        content.uv0_x === undefined || content.uv0_x === null ? null : Number(content.uv0_x),
      uv0Y:
        content.uv0_y === undefined || content.uv0_y === null ? null : Number(content.uv0_y),
      uv1X:
        content.uv1_x === undefined || content.uv1_x === null ? null : Number(content.uv1_x),
      uv1Y:
        content.uv1_y === undefined || content.uv1_y === null ? null : Number(content.uv1_y),
      offsetX: Number(content.offset_x || 8),
      offsetY: Number(content.offset_y || -8),
      value: Number(content.value || 0),
      roundValue: Boolean(content.round_value),
      sourceEnabled: Boolean(content.has_x),
      targetEnabled:
        content.value === undefined || content.value === null
          ? String(content.axis || "").toLowerCase() !== "none"
          : Boolean(content.value),
      x1: Number(content.x1 || 0),
      y1: Number(content.y1 || 0),
      x2: Number(content.x2 || 1),
      y2: Number(content.y2 || 1),
      thickness: Number(content.thickness || 1),
      xAxis: Math.min(2, Math.max(0, Number(content.x_axis || 0) | 0)),
      yAxis: Math.min(5, Math.max(3, Number(content.y_axis || 3) | 0)),
      subplotIndex: Math.max(0, Number(content.subplot_index || 0) | 0),
      version,
      buffers: decoded,
    };

    if (!this.primitives.has(primitiveId)) {
      this.primitiveOrder.push(primitiveId);
    }
    this.primitives.set(primitiveId, payload);
    if (this.wasmReady) {
      this._syncPrimitiveToWasm(payload);
    }
    if (!this.view.initialized) {
      if (this.initialAutoFitActive) {
        this._scheduleInitialAutoFit();
      } else {
        this._autoscale();
      }
      return;
    }
    if (this.initialAutoFitActive) {
      this._scheduleInitialAutoFit();
      return;
    }
    this._markDirty();
  }

  _handlePrimitiveRemoveMessage(content) {
    const primitiveId = String(content.primitive_id || "");
    if (!primitiveId) {
      return;
    }
    this.primitives.delete(primitiveId);
    this.primitiveOrder = this.primitiveOrder.filter((id) => id !== primitiveId);
    const token = this.primitiveTokenById.get(primitiveId);
    if (token != null && this.wasmReady) {
      this.wasm.removePrimitive(token);
    }
    this.primitiveTokenById.delete(primitiveId);
    if (this.initialAutoFitActive) {
      this._scheduleInitialAutoFit();
      return;
    }
    this._markDirty();
  }

  _refreshLegend() {
    if (this.implotEnabled || !this.legend) {
      if (this.legend) {
        this.legend.style.display = "none";
        this.legend.replaceChildren();
      }
      return;
    }
    this.legend.replaceChildren();

    for (const seriesId of this.seriesOrder) {
      const record = this.series.get(seriesId);
      if (!record) {
        continue;
      }
      const button = document.createElement("button");
      button.type = "button";
      button.style.display = "flex";
      button.style.alignItems = "center";
      button.style.gap = "8px";
      button.style.border = "1px solid #d4d4d8";
      button.style.borderRadius = "4px";
      button.style.padding = "2px 6px";
      button.style.background = record.visible ? "#ffffff" : "#f4f4f5";
      button.style.cursor = "pointer";
      button.style.fontSize = "12px";

      const swatch = document.createElement("span");
      swatch.style.width = "10px";
      swatch.style.height = "10px";
      swatch.style.borderRadius = "999px";
      swatch.style.display = "inline-block";
      const swatchColor =
        record.style && record.style.hasColor
          ? rgbaToCss(record.style.colorR, record.style.colorG, record.style.colorB, record.style.colorA)
          : record.color;
      swatch.style.background = swatchColor;
      swatch.style.opacity = record.visible ? "1" : "0.35";
      button.appendChild(swatch);

      const text = document.createElement("span");
      text.textContent = record.name;
      text.style.color = record.visible ? "#18181b" : "#71717a";
      button.appendChild(text);

      button.onclick = () => {
        record.visible = !record.visible;
        if (this.wasmReady) {
          this.wasm.setSeriesVisible(record.token, record.visible);
        }
        this._refreshLegend();
        this._markDirty();
      };
      this.legend.appendChild(button);
    }
  }

  _plotRect() {
    const left = 58;
    const top = 36;
    const width = Math.max(10, this.cssWidth - 78);
    const height = Math.max(10, this.cssHeight - 70);
    return { left, top, width, height };
  }

  _interactionRegion(pos) {
    const rect = this._plotRect();
    if (
      pos.x >= rect.left &&
      pos.x <= rect.left + rect.width &&
      pos.y >= rect.top &&
      pos.y <= rect.top + rect.height
    ) {
      return "plot";
    }
    if (
      pos.x >= rect.left &&
      pos.x <= rect.left + rect.width &&
      pos.y >= rect.top + rect.height &&
      pos.y <= rect.top + rect.height + 24
    ) {
      return "x-axis";
    }
    if (
      pos.x >= 0 &&
      pos.x <= rect.left &&
      pos.y >= rect.top &&
      pos.y <= rect.top + rect.height
    ) {
      return "y-axis";
    }
    return "outside";
  }

  _resizeCanvas() {
    if (this.disposed) {
      return;
    }

    this.cssWidth = Math.max(120, Number(this.model.get("width")) || 900);
    this.cssHeight = Math.max(100, Number(this.model.get("height")) || 450);
    this.dpr = Math.max(1, window.devicePixelRatio || 1);

    this.wrapper.style.width = `${this.cssWidth}px`;
    this.wrapper.style.height = `${this.cssHeight}px`;
    this.canvas.width = Math.max(1, Math.round(this.cssWidth * this.dpr));
    this.canvas.height = Math.max(1, Math.round(this.cssHeight * this.dpr));
    this.overlay.width = this.canvas.width;
    this.overlay.height = this.canvas.height;

    if (this.wasmReady) {
      this.wasm.setCanvas(this.cssWidth, this.cssHeight, this.dpr);
    }

    this._markDirty();
  }

  _markDirty() {
    if (this.disposed) {
      return;
    }
    this.dirty = true;
    if (this.rafId === 0) {
      this.rafId = window.requestAnimationFrame(() => this._frame());
    }
  }

  _flushInputFrame() {
    if (this.disposed || !this.wasmReady) {
      return;
    }
    this.dirty = true;
    if (this.rafId !== 0) {
      window.cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
    this._frame();
  }

  _frame() {
    this.rafId = 0;
    if (this.disposed) {
      return;
    }
    const perfLoopActive = this.perfReportingEnabled && this.wasmStatus === "ready";
    if (!this.dirty && !perfLoopActive) {
      return;
    }
    this.dirty = false;
    this._draw();
    if (!this.disposed && perfLoopActive && this.rafId === 0) {
      this.rafId = window.requestAnimationFrame(() => this._frame());
    }
  }

  _hasVisibleSeries() {
    for (const record of this.series.values()) {
      if (record.visible && record.data.length > 0) {
        return true;
      }
    }
    return false;
  }

  _hasRenderablePrimitives() {
    return this.primitives.size > 0;
  }

  _autoscale() {
    if (this.wasmReady && (this._hasVisibleSeries() || this.primitives.size > 0)) {
      const view = this.wasm.autoscale();
      if (view) {
        this.view = view;
        this._emitViewChange();
        this._markDirty();
        return;
      }
    }

    const bounds = this._computeAutoBounds();
    this.view.xMin = bounds.xMin;
    this.view.xMax = bounds.xMax;
    this.view.yMin = bounds.yMin;
    this.view.yMax = bounds.yMax;
    this.view.initialized = true;
    this._emitViewChange();
    this._markDirty();
  }

  _emitViewChange() {
    if (!this.view.initialized || typeof this.model.send !== "function") {
      return;
    }
    this.model.send({
      type: "view_change",
      x_min: this.view.xMin,
      x_max: this.view.xMax,
      y_min: this.view.yMin,
      y_max: this.view.yMax,
    });
  }

  _updatePerfStats() {
    const now = performance.now();
    if (!Number.isFinite(this.perfState.fpsWindowStartMs) || this.perfState.fpsWindowStartMs <= 0) {
      this.perfState.fpsWindowStartMs = now;
      this.perfState.fpsFrames = 0;
    }
    this.perfState.fpsFrames += 1;

    const elapsed = now - this.perfState.fpsWindowStartMs;
    if (elapsed >= 250) {
      this.perfState.fps = elapsed > 0 ? (this.perfState.fpsFrames * 1000.0) / elapsed : 0;
      this.perfState.fpsFrames = 0;
      this.perfState.fpsWindowStartMs = now;
    }

    this.perfState.latestWasm = this.wasm.getPerfStats();
    this._emitPerfStats(now);
  }

  _emitPerfStats(nowMs) {
    if (!this.perfReportingEnabled || typeof this.model.send !== "function") {
      return;
    }
    if (nowMs - this.perfState.lastEmitMs < this.perfReportingIntervalMs) {
      return;
    }
    const wasmStats = this.perfState.latestWasm;
    if (!wasmStats) {
      return;
    }
    this.perfState.lastEmitMs = nowMs;
    this.model.send({
      type: "perf_stats",
      fps: this.perfState.fps,
      lod_ms: wasmStats.lodMs,
      segment_build_ms: wasmStats.segmentBuildMs,
      render_ms: wasmStats.renderMs,
      frame_ms: wasmStats.frameMs,
      draw_points: wasmStats.drawPoints,
      draw_segments: wasmStats.drawSegments,
      primitive_count: wasmStats.primitiveCount,
      pixel_width: wasmStats.pixelWidth,
    });
  }

  _emitInteractionUpdate() {
    if (!this.wasmReady || typeof this.model.send !== "function") {
      return;
    }
    const tuples = this.wasm.getInteractions();
    if (!(tuples instanceof Float32Array) || tuples.length === 0) {
      this.lastInteractionHash = "";
      return;
    }

    const tools = [];
    let selection = null;
    for (let i = 0; i + 7 < tuples.length; i += 8) {
      const kind = tuples[i] | 0;
      const id = tuples[i + 1] | 0;
      const subplotIndex = tuples[i + 2] | 0;
      const active = tuples[i + 3] | 0;
      const v0 = Number(tuples[i + 4]);
      const v1 = Number(tuples[i + 5]);
      const v2 = Number(tuples[i + 6]);
      const v3 = Number(tuples[i + 7]);

      if (kind === 100) {
        selection = {
          subplot_index: subplotIndex,
          x_min: Math.min(v0, v1),
          x_max: Math.max(v0, v1),
          y_min: Math.min(v2, v3),
          y_max: Math.max(v2, v3),
        };
        continue;
      }

      let toolType = "";
      const payload = {
        tool_id: id,
        subplot_index: subplotIndex,
        active: active !== 0,
      };
      if (kind === 21) {
        toolType = "drag_line_x";
        payload.value = v0;
      } else if (kind === 22) {
        toolType = "drag_line_y";
        payload.value = v1;
      } else if (kind === 23) {
        toolType = "drag_point";
        payload.x = v0;
        payload.y = v1;
      } else if (kind === 24) {
        toolType = "drag_rect";
        payload.x1 = v0;
        payload.y1 = v1;
        payload.x2 = v2;
        payload.y2 = v3;
      } else if (kind === 27) {
        toolType = "colormap_slider";
        payload.value = v0;
      } else if (kind === 28) {
        toolType = "colormap_button";
      } else if (kind === 29) {
        toolType = "colormap_selector";
        payload.colormap_index = v0;
      } else if (kind === 30) {
        toolType = "drag_drop_plot";
        payload.x = v0;
        payload.y = v1;
        payload.source_kind = v2 | 0;
        payload.source_axis = v3 | 0;
      } else if (kind === 31) {
        toolType = "drag_drop_axis";
        payload.source_axis = v0 | 0;
        payload.target_axis = v1 | 0;
        payload.source_kind = v2 | 0;
        payload.source_subplot = v3 | 0;
      } else if (kind === 32) {
        toolType = "drag_drop_legend";
        payload.source_kind = v0 | 0;
        payload.source_axis = v1 | 0;
        payload.source_subplot = v2 | 0;
      } else {
        continue;
      }
      payload.type = toolType;
      tools.push(payload);
    }

    if (tools.length === 0 && selection == null) {
      this.lastInteractionHash = "";
      return;
    }
    const message = {
      type: "interaction_update",
      tools,
      selection,
    };
    const hash = JSON.stringify(message);
    if (hash === this.lastInteractionHash) {
      return;
    }
    this.lastInteractionHash = hash;
    this.model.send(message);
  }

  _draw() {
    const ctx = this.ctx;
    const title = String(this.model.get("title") || "");

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);

    this._drawBackendBadge(ctx);

    if (this.wasmStatus === "loading") {
      ctx.fillStyle = "#52525b";
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText("Loading WASM runtime...", 16, 24);
      return;
    }

    if (this.wasmStatus === "error") {
      this._drawStrictWasmErrorPanel(ctx);
      return;
    }

    if (!this.wasmReady) {
      ctx.fillStyle = "#991b1b";
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText("WASM runtime not ready.", 16, 24);
      return;
    }

    if (!this.view.initialized && this.primitives.size > 0 && !this._hasVisibleSeries()) {
      const view = this.wasm.autoscale();
      if (view) {
        this.view = view;
      }
    }

    const drawnByWasm = this.wasm.render(title);
    if (!drawnByWasm) {
      this.wasmStatus = "error";
      this.wasmError =
        "WASM draw pipeline error. WebGL context may be unavailable or lost for this notebook renderer.";
      this._drawStrictWasmErrorPanel(ctx);
      return;
    }
    const view = this.wasm.getView();
    if (view) {
      const changed =
        !this.view.initialized ||
        Math.abs(this.view.xMin - view.xMin) > 1e-9 ||
        Math.abs(this.view.xMax - view.xMax) > 1e-9 ||
        Math.abs(this.view.yMin - view.yMin) > 1e-9 ||
        Math.abs(this.view.yMax - view.yMax) > 1e-9;
      this.view = view;
      if (changed) {
        this._emitViewChange();
      }
    }

    this._emitInteractionUpdate();
    this._updatePerfStats();

  }

  _wrapText(ctx, text, maxWidth) {
    const src = String(text || "").trim();
    if (!src) {
      return [""];
    }
    const words = src.split(/\s+/);
    const lines = [];
    let line = "";
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (!line || ctx.measureText(candidate).width <= maxWidth) {
        line = candidate;
      } else {
        lines.push(line);
        line = word;
      }
    }
    if (line) {
      lines.push(line);
    }
    return lines;
  }

  _drawStrictWasmErrorPanel(ctx) {
    const reason = String(this.wasmError || "WASM runtime error.");
    const reasonLower = reason.toLowerCase();
    const hasWebGLHint =
      !this.webgl2Probe.available ||
      reasonLower.includes("webgl") ||
      reasonLower.includes("feature_failure") ||
      reasonLower.includes("egl_create") ||
      reasonLower.includes("driver");

    const lines = [reason];
    if (this.strictWasm) {
      lines.push("Strict WASM mode is enabled; JS fallback is disabled.");
    }
    if (hasWebGLHint) {
      lines.push("Detected issue: WebGL2 context is unavailable for this browser/runtime.");
      if (this.webgl2Probe.reason) {
        lines.push(`Probe detail: ${this.webgl2Probe.reason}`);
      }
      lines.push("Open this notebook in a local desktop browser session.");
      lines.push("Console check: !!document.createElement('canvas').getContext('webgl2')");
      lines.push("If false: enable hardware acceleration or update GPU drivers.");
    }

    const x = 12;
    const y = 14;
    const lineHeight = 16;
    const innerPad = 10;
    const maxTextWidth = Math.max(120, this.cssWidth - x * 2 - innerPad * 2);
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";

    const wrapped = [];
    for (const line of lines) {
      const pieces = this._wrapText(ctx, line, maxTextWidth);
      for (const piece of pieces) {
        wrapped.push(piece);
      }
    }
    const panelHeight = 30 + wrapped.length * lineHeight + 10;

    ctx.fillStyle = "rgba(254, 242, 242, 0.98)";
    ctx.strokeStyle = "rgba(220, 38, 38, 0.45)";
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, this.cssWidth - x * 2, panelHeight);
    ctx.strokeRect(x, y, this.cssWidth - x * 2, panelHeight);

    ctx.fillStyle = "#7f1d1d";
    ctx.font = "bold 12px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText("nbimplot runtime error", x + innerPad, y + 18);

    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    let yLine = y + 36;
    for (const line of wrapped) {
      ctx.fillText(line, x + innerPad, yLine);
      yLine += lineHeight;
    }
  }

  _drawBackendBadge(ctx) {
    let label = "WASM";
    if (this.wasmStatus === "loading") {
      label = "WASM loading";
    } else if (this.wasmStatus === "error") {
      label = "WASM error";
    } else if (this.wasmReady) {
      label = "WASM + ImPlot";
      if (this.perfReportingEnabled && this.perfState.fps > 0) {
        label += ` ${this.perfState.fps.toFixed(1)} FPS`;
      }
    }

    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    const width = ctx.measureText(label).width + 10;
    const x = this.cssWidth - width - 8;
    const y = 8;
    ctx.fillStyle = "rgba(24, 24, 27, 0.88)";
    ctx.fillRect(x, y, width, 16);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 5, y + 12);
  }

  _drawSeriesFromWasm(ctx, rect) {
    this.wasm.setView(this.view);
    const tuples = this.wasm.buildDrawData(Math.max(1, Math.floor(rect.width)));
    if (tuples === null) {
      return false;
    }

    const xSpan = Math.max(1e-9, this.view.xMax - this.view.xMin);
    const ySpan = Math.max(1e-9, this.view.yMax - this.view.yMin);

    let currentSlot = -1;
    let pathStarted = false;
    let needsMove = true;

    for (let i = 0; i < tuples.length; i += 4) {
      const slot = tuples[i] | 0;
      const x = tuples[i + 1];
      const y = tuples[i + 2];
      const penDown = tuples[i + 3] > 0.5;

      if (slot !== currentSlot) {
        if (pathStarted) {
          ctx.stroke();
          pathStarted = false;
        }
        currentSlot = slot;
        needsMove = true;
        ctx.strokeStyle = this._colorForSlot(slot);
        ctx.lineWidth = 1;
        ctx.beginPath();
      }

      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        needsMove = true;
        continue;
      }

      const xPx = rect.left + ((x - this.view.xMin) / xSpan) * rect.width;
      const yPx = rect.top + rect.height - ((y - this.view.yMin) / ySpan) * rect.height;

      if (!pathStarted || !penDown || needsMove) {
        ctx.moveTo(xPx, yPx);
        pathStarted = true;
      } else {
        ctx.lineTo(xPx, yPx);
      }

      needsMove = false;
    }

    if (pathStarted) {
      ctx.stroke();
    }

    return true;
  }

  _colorForSlot(slot) {
    if (slot >= 0 && slot < this.colorBySlot.length && this.colorBySlot[slot]) {
      return this.colorBySlot[slot];
    }
    return SERIES_COLORS[Math.abs(slot) % SERIES_COLORS.length];
  }

  _drawGrid(ctx, rect) {
    ctx.strokeStyle = "#ededf0";
    ctx.lineWidth = 1;

    const cols = 6;
    const rows = 4;

    for (let c = 1; c < cols; c += 1) {
      const x = rect.left + (c / cols) * rect.width;
      ctx.beginPath();
      ctx.moveTo(x, rect.top);
      ctx.lineTo(x, rect.top + rect.height);
      ctx.stroke();
    }
    for (let r = 1; r < rows; r += 1) {
      const y = rect.top + (r / rows) * rect.height;
      ctx.beginPath();
      ctx.moveTo(rect.left, y);
      ctx.lineTo(rect.left + rect.width, y);
      ctx.stroke();
    }
  }

  _drawAxesLabels(ctx, rect) {
    const xMin = this.view.xMin;
    const xMax = this.view.xMax;
    const yMin = this.view.yMin;
    const yMax = this.view.yMax;

    ctx.fillStyle = "#52525b";
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(xMin.toFixed(1), rect.left, rect.top + rect.height + 16);
    ctx.fillText(xMax.toFixed(1), rect.left + rect.width - 28, rect.top + rect.height + 16);
    ctx.fillText(yMax.toPrecision(4), 8, rect.top + 10);
    ctx.fillText(yMin.toPrecision(4), 8, rect.top + rect.height);
  }

  _drawPrimitives(ctx, rect) {
    if (this.primitives.size === 0) {
      return;
    }
    const xSpan = Math.max(1e-9, this.view.xMax - this.view.xMin);
    const ySpan = Math.max(1e-9, this.view.yMax - this.view.yMin);
    const toX = (x) => rect.left + ((x - this.view.xMin) / xSpan) * rect.width;
    const toY = (y) => rect.top + rect.height - ((y - this.view.yMin) / ySpan) * rect.height;
    const pxPerX = Math.abs(rect.width / xSpan);
    const pxPerY = Math.abs(rect.height / ySpan);
    const baseY = toY(0);
    const baseX = toX(0);

    const heatColor = (tValue) => {
      const t = clamp(tValue, 0, 1);
      const red = Math.round(255 * t);
      const blue = Math.round(255 * (1 - t));
      const green = Math.round(110 + 80 * (1 - Math.abs(2 * t - 1)));
      return `rgba(${red}, ${green}, ${blue}, 0.72)`;
    };

    for (let idx = 0; idx < this.primitiveOrder.length; idx += 1) {
      const primitive = this.primitives.get(this.primitiveOrder[idx]);
      if (!primitive) {
        continue;
      }
      if (primitive.hidden) {
        continue;
      }
      const color = SERIES_COLORS[idx % SERIES_COLORS.length];
      const kind = primitive.kind;
      const bufs = primitive.buffers;

      if (kind === "dummy") {
        continue;
      }

      if (kind === "scatter" || kind === "bubbles") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const sVals =
          kind === "bubbles" ? (primitive.hasX ? bufs[2] : bufs[1]) : null;
        const n = yVals ? yVals.length : 0;
        ctx.fillStyle = color;
        const markerSize = Math.max(1, Number(primitive.size || 2));
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          const y = yVals[i];
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            continue;
          }
          let radius = markerSize;
          if (sVals) {
            const s = sVals[i];
            if (!Number.isFinite(s)) {
              continue;
            }
            radius = Math.max(1, Math.abs(s) * Math.min(pxPerX, pxPerY) * 0.08);
          }
          const xPx = toX(x);
          const yPx = toY(y);
          ctx.beginPath();
          ctx.arc(xPx, yPx, radius, 0, Math.PI * 2);
          ctx.fill();
        }
        continue;
      }

      if (kind === "stairs" || kind === "digital") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const n = yVals ? yVals.length : 0;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          const y = yVals[i];
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            started = false;
            continue;
          }
          const xPx = toX(x);
          const yPx = toY(y);
          if (!started) {
            ctx.moveTo(xPx, yPx);
            started = true;
          } else {
            ctx.lineTo(xPx, toY(yVals[i - 1]));
            ctx.lineTo(xPx, yPx);
          }
        }
        ctx.stroke();
        continue;
      }

      if (kind === "stems") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const n = yVals ? yVals.length : 0;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1;
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          const y = yVals[i];
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            continue;
          }
          const xPx = toX(x);
          const yPx = toY(y);
          ctx.beginPath();
          ctx.moveTo(xPx, baseY);
          ctx.lineTo(xPx, yPx);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(xPx, yPx, 2, 0, Math.PI * 2);
          ctx.fill();
        }
        continue;
      }

      if (kind === "bars") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const n = yVals ? yVals.length : 0;
        const barHalfPx = Math.max(1, Math.abs(((Number(primitive.barWidth || 0.67) / xSpan) * rect.width) / 2));
        ctx.fillStyle = `${color}99`;
        ctx.strokeStyle = color;
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          const y = yVals[i];
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            continue;
          }
          const xPx = toX(x);
          const yPx = toY(y);
          const top = Math.min(yPx, baseY);
          const h = Math.abs(baseY - yPx);
          ctx.fillRect(xPx - barHalfPx, top, barHalfPx * 2, h);
          ctx.strokeRect(xPx - barHalfPx, top, barHalfPx * 2, h);
        }
        continue;
      }

      if (kind === "bars_h") {
        const xVals = bufs[0];
        const yVals = bufs[1];
        const n = Math.min(xVals ? xVals.length : 0, yVals ? yVals.length : 0);
        const barHalfPx = Math.max(1, Math.abs(((Number(primitive.barHeight || 0.67) / ySpan) * rect.height) / 2));
        ctx.fillStyle = `${color}99`;
        ctx.strokeStyle = color;
        for (let i = 0; i < n; i += 1) {
          const x = xVals[i];
          const y = yVals[i];
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            continue;
          }
          const xPx = toX(x);
          const yPx = toY(y);
          const left = Math.min(xPx, baseX);
          const w = Math.abs(baseX - xPx);
          ctx.fillRect(left, yPx - barHalfPx, w, barHalfPx * 2);
          ctx.strokeRect(left, yPx - barHalfPx, w, barHalfPx * 2);
        }
        continue;
      }

      if (kind === "bar_groups") {
        const flat = bufs[0];
        const itemCount = Math.max(1, primitive.itemCount | 0);
        const groupCount = Math.max(0, primitive.groupCount | 0);
        if (!flat || groupCount <= 0) {
          continue;
        }
        const groupSize = Math.abs(Number(primitive.groupSize || 0.67));
        const shift = Number(primitive.shift || 0);
        const itemWidth = groupSize / itemCount;
        for (let group = 0; group < groupCount; group += 1) {
          const groupCenter = group + shift;
          for (let item = 0; item < itemCount; item += 1) {
            const flatIdx = item * groupCount + group;
            if (flatIdx >= flat.length) {
              continue;
            }
            const value = flat[flatIdx];
            if (!Number.isFinite(value)) {
              continue;
            }
            const xCenter = groupCenter - groupSize / 2 + (item + 0.5) * itemWidth;
            const xPx = toX(xCenter);
            const yPx = toY(value);
            const top = Math.min(yPx, baseY);
            const h = Math.abs(baseY - yPx);
            const half = Math.max(1, Math.abs(itemWidth * pxPerX * 0.45));
            const itemColor = SERIES_COLORS[item % SERIES_COLORS.length];
            ctx.fillStyle = `${itemColor}99`;
            ctx.strokeStyle = itemColor;
            ctx.fillRect(xPx - half, top, half * 2, h);
            ctx.strokeRect(xPx - half, top, half * 2, h);
          }
        }
        continue;
      }

      if (kind === "shaded") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const y1 = primitive.hasX ? bufs[1] : bufs[0];
        const y2 = primitive.hasX ? bufs[2] : bufs[1];
        const n = Math.min(y1 ? y1.length : 0, y2 ? y2.length : 0);
        if (n === 0) {
          continue;
        }
        ctx.fillStyle = `${color}${Math.round(clamp(primitive.alpha * 255, 0, 255)).toString(16).padStart(2, "0")}`;
        ctx.beginPath();
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          const y = y1[i];
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            continue;
          }
          const xPx = toX(x);
          const yPx = toY(y);
          if (i === 0) ctx.moveTo(xPx, yPx);
          else ctx.lineTo(xPx, yPx);
        }
        for (let i = n - 1; i >= 0; i -= 1) {
          const x = xVals ? xVals[i] : i;
          const y = y2[i];
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            continue;
          }
          ctx.lineTo(toX(x), toY(y));
        }
        ctx.closePath();
        ctx.fill();
        continue;
      }

      if (kind === "error_bars") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const errVals = primitive.hasX ? bufs[2] : bufs[1];
        const n = Math.min(yVals ? yVals.length : 0, errVals ? errVals.length : 0);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          const y = yVals[i];
          const err = Math.abs(errVals[i]);
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(err)) {
            continue;
          }
          const xPx = toX(x);
          const yMinPx = toY(y - err);
          const yMaxPx = toY(y + err);
          ctx.beginPath();
          ctx.moveTo(xPx, yMinPx);
          ctx.lineTo(xPx, yMaxPx);
          ctx.moveTo(xPx - 3, yMinPx);
          ctx.lineTo(xPx + 3, yMinPx);
          ctx.moveTo(xPx - 3, yMaxPx);
          ctx.lineTo(xPx + 3, yMaxPx);
          ctx.stroke();
        }
        continue;
      }

      if (kind === "error_bars_h") {
        const xVals = bufs[0];
        const errVals = bufs[1];
        const yVals = bufs[2];
        const n = Math.min(xVals ? xVals.length : 0, errVals ? errVals.length : 0, yVals ? yVals.length : 0);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        for (let i = 0; i < n; i += 1) {
          const x = xVals[i];
          const y = yVals[i];
          const err = Math.abs(errVals[i]);
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(err)) {
            continue;
          }
          const yPx = toY(y);
          const xMinPx = toX(x - err);
          const xMaxPx = toX(x + err);
          ctx.beginPath();
          ctx.moveTo(xMinPx, yPx);
          ctx.lineTo(xMaxPx, yPx);
          ctx.moveTo(xMinPx, yPx - 3);
          ctx.lineTo(xMinPx, yPx + 3);
          ctx.moveTo(xMaxPx, yPx - 3);
          ctx.lineTo(xMaxPx, yPx + 3);
          ctx.stroke();
        }
        continue;
      }

      if (kind === "inf_lines") {
        const vals = bufs[0];
        const n = vals ? vals.length : 0;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        for (let i = 0; i < n; i += 1) {
          const value = vals[i];
          if (!Number.isFinite(value)) {
            continue;
          }
          ctx.beginPath();
          if (primitive.axis === "y") {
            const yPx = toY(value);
            ctx.moveTo(rect.left, yPx);
            ctx.lineTo(rect.left + rect.width, yPx);
          } else {
            const xPx = toX(value);
            ctx.moveTo(xPx, rect.top);
            ctx.lineTo(xPx, rect.top + rect.height);
          }
          ctx.stroke();
        }
        continue;
      }

      if (kind === "histogram") {
        const edges = bufs[0];
        const counts = bufs[1];
        const n = Math.min(counts ? counts.length : 0, edges ? Math.max(0, edges.length - 1) : 0);
        ctx.fillStyle = `${color}99`;
        ctx.strokeStyle = color;
        const zeroY = toY(0);
        for (let i = 0; i < n; i += 1) {
          const x0 = edges[i];
          const x1 = edges[i + 1];
          const c = counts[i];
          if (!Number.isFinite(x0) || !Number.isFinite(x1) || !Number.isFinite(c)) {
            continue;
          }
          const left = toX(Math.min(x0, x1));
          const right = toX(Math.max(x0, x1));
          const yPx = toY(c);
          const top = Math.min(yPx, zeroY);
          const w = Math.max(1, right - left);
          const h = Math.abs(zeroY - yPx);
          ctx.fillRect(left, top, w, h);
          ctx.strokeRect(left, top, w, h);
        }
        continue;
      }

      if (kind === "histogram2d") {
        const xEdges = bufs[0];
        const yEdges = bufs[1];
        const counts = bufs[2];
        const rows = Math.max(0, primitive.rows | 0);
        const cols = Math.max(0, primitive.cols | 0);
        if (
          !xEdges ||
          !yEdges ||
          !counts ||
          xEdges.length < 2 ||
          yEdges.length < 2 ||
          rows <= 0 ||
          cols <= 0
        ) {
          continue;
        }
        let min = Number.POSITIVE_INFINITY;
        let max = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < counts.length; i += 1) {
          const v = counts[i];
          if (!Number.isFinite(v)) {
            continue;
          }
          if (v < min) min = v;
          if (v > max) max = v;
        }
        if (!Number.isFinite(min) || !Number.isFinite(max)) {
          continue;
        }
        const denom = Math.max(1e-9, max - min);
        for (let row = 0; row < rows; row += 1) {
          for (let col = 0; col < cols; col += 1) {
            const idxFlat = row * cols + col;
            if (idxFlat >= counts.length || row + 1 >= xEdges.length || col + 1 >= yEdges.length) {
              continue;
            }
            const value = counts[idxFlat];
            if (!Number.isFinite(value)) {
              continue;
            }
            const x0 = xEdges[row];
            const x1 = xEdges[row + 1];
            const y0 = yEdges[col];
            const y1 = yEdges[col + 1];
            ctx.fillStyle = heatColor((value - min) / denom);
            const px0 = toX(Math.min(x0, x1));
            const px1 = toX(Math.max(x0, x1));
            const py0 = toY(Math.min(y0, y1));
            const py1 = toY(Math.max(y0, y1));
            const left = Math.min(px0, px1);
            const top = Math.min(py0, py1);
            const w = Math.max(1, Math.abs(px1 - px0));
            const h = Math.max(1, Math.abs(py1 - py0));
            ctx.fillRect(left, top, w, h);
          }
        }
        continue;
      }

      if (kind === "heatmap" || kind === "image") {
        const flat = bufs[0];
        const rows = Math.max(0, primitive.rows | 0);
        const cols = Math.max(0, primitive.cols | 0);
        if (!flat || rows <= 0 || cols <= 0) {
          continue;
        }
        let min = Number.POSITIVE_INFINITY;
        let max = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < flat.length; i += 1) {
          const v = flat[i];
          if (!Number.isFinite(v)) continue;
          if (v < min) min = v;
          if (v > max) max = v;
        }
        if (!Number.isFinite(min) || !Number.isFinite(max)) {
          continue;
        }
        const denom = Math.max(1e-9, max - min);
        for (let r = 0; r < rows; r += 1) {
          for (let c = 0; c < cols; c += 1) {
            const idxFlat = r * cols + c;
            if (idxFlat >= flat.length) {
              continue;
            }
            const v = flat[idxFlat];
            if (!Number.isFinite(v)) {
              continue;
            }
            ctx.fillStyle = heatColor((v - min) / denom);
            const x0 = toX(c);
            const x1 = toX(c + 1);
            const y0 = toY(r);
            const y1 = toY(r + 1);
            const left = Math.min(x0, x1);
            const top = Math.min(y0, y1);
            const w = Math.max(1, Math.abs(x1 - x0));
            const h = Math.max(1, Math.abs(y1 - y0));
            ctx.fillRect(left, top, w, h);
          }
        }
        continue;
      }

      if (kind === "pie_chart") {
        const vals = bufs[0];
        if (!vals || vals.length === 0) {
          continue;
        }
        const cx = toX(Number(primitive.x || 0));
        const cy = toY(Number(primitive.y || 0));
        const radius = Math.max(
          2,
          Math.abs(Number(primitive.radius || 1)) * 0.5 * (pxPerX + pxPerY),
        );
        let total = 0;
        for (let i = 0; i < vals.length; i += 1) {
          const value = vals[i];
          if (Number.isFinite(value) && value > 0) {
            total += value;
          }
        }
        if (total <= 0) {
          continue;
        }

        let angle = (Number(primitive.angle0 || 90) * Math.PI) / 180;
        for (let i = 0; i < vals.length; i += 1) {
          const value = vals[i];
          if (!Number.isFinite(value) || value <= 0) {
            continue;
          }
          const ratio = value / total;
          const span = ratio * Math.PI * 2;
          const start = angle;
          const end = angle + span;
          const itemColor = SERIES_COLORS[i % SERIES_COLORS.length];
          ctx.fillStyle = `${itemColor}d9`;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.arc(cx, cy, radius, start, end);
          ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1;
          ctx.stroke();

          const labels = primitive.labels || [];
          if (labels.length > 0 && radius > 18) {
            const mid = start + span * 0.5;
            const tx = cx + Math.cos(mid) * radius * 0.62;
            const ty = cy + Math.sin(mid) * radius * 0.62;
            ctx.fillStyle = "#111827";
            ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
            const text = String(labels[i] || i);
            ctx.fillText(text, tx, ty);
          }
          angle = end;
        }
        continue;
      }

      if (kind === "text" || kind === "annotation") {
        const x = Number(primitive.x);
        const y = Number(primitive.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          continue;
        }
        const xPx = toX(x);
        const yPx = toY(y);
        ctx.fillStyle = color;
        ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
        if (kind === "annotation") {
          ctx.beginPath();
          ctx.arc(xPx, yPx, 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillText(String(primitive.label || ""), xPx + primitive.offsetX, yPx + primitive.offsetY);
        } else {
          ctx.fillText(String(primitive.label || ""), xPx, yPx);
        }
      }
    }
  }

  _drawTooltip(ctx, rect) {
    if (!this.mouse.inside || !this._hasVisibleSeries()) {
      return;
    }

    const xSpan = Math.max(1e-9, this.view.xMax - this.view.xMin);
    const ySpan = Math.max(1e-9, this.view.yMax - this.view.yMin);
    const xNorm = clamp((this.mouse.x - rect.left) / rect.width, 0, 1);
    const xData = this.view.xMin + xNorm * xSpan;
    const nearestIndex = Math.round(xData);

    let best = null;
    for (const seriesId of this.seriesOrder) {
      const record = this.series.get(seriesId);
      if (!record || !record.visible) {
        continue;
      }
      if (nearestIndex < 0 || nearestIndex >= record.data.length) {
        continue;
      }
      const y = record.data[nearestIndex];
      if (!Number.isFinite(y)) {
        continue;
      }
      const yPx = rect.top + rect.height - ((y - this.view.yMin) / ySpan) * rect.height;
      const dist = Math.abs(yPx - this.mouse.y);
      if (best === null || dist < best.dist) {
        best = {
          name: record.name,
          color: record.color,
          index: nearestIndex,
          y,
          xPx: rect.left + ((nearestIndex - this.view.xMin) / xSpan) * rect.width,
          yPx,
          dist,
        };
      }
    }

    if (!best || best.dist > 40) {
      return;
    }

    ctx.strokeStyle = "rgba(24, 24, 27, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(best.xPx, rect.top);
    ctx.lineTo(best.xPx, rect.top + rect.height);
    ctx.stroke();

    ctx.fillStyle = best.color;
    ctx.beginPath();
    ctx.arc(best.xPx, best.yPx, 3, 0, Math.PI * 2);
    ctx.fill();

    const label = `${best.name}: ${best.y.toPrecision(6)} @ ${best.index}`;
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    const padding = 6;
    const textWidth = ctx.measureText(label).width;
    const boxX = clamp(this.mouse.x + 10, rect.left + 4, rect.left + rect.width - textWidth - 20);
    const boxY = clamp(this.mouse.y - 28, rect.top + 4, rect.top + rect.height - 24);
    const boxW = textWidth + padding * 2;
    const boxH = 20;

    ctx.fillStyle = "rgba(24, 24, 27, 0.9)";
    ctx.fillRect(boxX, boxY, boxW, boxH);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, boxX + padding, boxY + 14);
  }

  _drawBoxZoomOverlay(ctx, rect) {
    if (!this.boxZoom.active) {
      return;
    }
    const x0 = clamp(Math.min(this.boxZoom.startX, this.boxZoom.endX), rect.left, rect.left + rect.width);
    const x1 = clamp(Math.max(this.boxZoom.startX, this.boxZoom.endX), rect.left, rect.left + rect.width);
    const y0 = clamp(Math.min(this.boxZoom.startY, this.boxZoom.endY), rect.top, rect.top + rect.height);
    const y1 = clamp(Math.max(this.boxZoom.startY, this.boxZoom.endY), rect.top, rect.top + rect.height);
    const w = Math.max(0, x1 - x0);
    const h = Math.max(0, y1 - y0);
    if (w < 1 || h < 1) {
      return;
    }

    ctx.fillStyle = "rgba(59, 130, 246, 0.12)";
    ctx.fillRect(x0, y0, w, h);
    ctx.strokeStyle = "rgba(37, 99, 235, 0.85)";
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x0, y0, w, h);
    ctx.setLineDash([]);
  }

  _computeAutoBounds() {
    let xMin = Number.POSITIVE_INFINITY;
    let xMax = Number.NEGATIVE_INFINITY;
    let yMin = Number.POSITIVE_INFINITY;
    let yMax = Number.NEGATIVE_INFINITY;
    let hasData = false;

    for (const record of this.series.values()) {
      if (!record.visible || record.data.length === 0) {
        continue;
      }
      hasData = true;
      xMin = Math.min(xMin, 0);
      xMax = Math.max(xMax, record.data.length - 1);
      for (let i = 0; i < record.data.length; i += 1) {
        const value = record.data[i];
        if (!Number.isFinite(value)) {
          continue;
        }
        if (value < yMin) yMin = value;
        if (value > yMax) yMax = value;
      }
    }

    for (const primitive of this.primitives.values()) {
      if (primitive.hidden) {
        continue;
      }
      const kind = primitive.kind;
      const bufs = primitive.buffers;
      const updatePoint = (x, y) => {
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        hasData = true;
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      };

      if (
        kind === "scatter" ||
        kind === "stairs" ||
        kind === "stems" ||
        kind === "digital"
      ) {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const n = yVals ? yVals.length : 0;
        for (let i = 0; i < n; i += 1) {
          updatePoint(xVals ? xVals[i] : i, yVals[i]);
        }
      } else if (kind === "bubbles") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const sVals = primitive.hasX ? bufs[2] : bufs[1];
        const n = Math.min(yVals ? yVals.length : 0, sVals ? sVals.length : 0);
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          const y = yVals[i];
          const s = Math.abs(sVals[i]);
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(s)) {
            continue;
          }
          updatePoint(x - s, y - s);
          updatePoint(x + s, y + s);
        }
      } else if (kind === "bars") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const n = yVals ? yVals.length : 0;
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          updatePoint(x, yVals[i]);
          updatePoint(x, 0);
        }
      } else if (kind === "bars_h") {
        const xVals = bufs[0];
        const yVals = bufs[1];
        const n = Math.min(xVals ? xVals.length : 0, yVals ? yVals.length : 0);
        for (let i = 0; i < n; i += 1) {
          updatePoint(xVals[i], yVals[i]);
          updatePoint(0, yVals[i]);
        }
      } else if (kind === "shaded") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const y1 = primitive.hasX ? bufs[1] : bufs[0];
        const y2 = primitive.hasX ? bufs[2] : bufs[1];
        const n = Math.min(y1 ? y1.length : 0, y2 ? y2.length : 0);
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          updatePoint(x, y1[i]);
          updatePoint(x, y2[i]);
        }
      } else if (kind === "bar_groups") {
        const flat = bufs[0];
        const itemCount = Math.max(1, primitive.itemCount | 0);
        const groupCount = Math.max(0, primitive.groupCount | 0);
        const groupSize = Math.abs(Number(primitive.groupSize || 0.67));
        const shift = Number(primitive.shift || 0);
        if (!flat || groupCount <= 0) {
          continue;
        }
        for (let group = 0; group < groupCount; group += 1) {
          const groupCenter = group + shift;
          updatePoint(groupCenter - groupSize * 0.5, 0);
          updatePoint(groupCenter + groupSize * 0.5, 0);
          for (let item = 0; item < itemCount; item += 1) {
            const flatIdx = item * groupCount + group;
            if (flatIdx >= flat.length) {
              continue;
            }
            updatePoint(groupCenter, flat[flatIdx]);
          }
        }
      } else if (kind === "error_bars") {
        const xVals = primitive.hasX ? bufs[0] : null;
        const yVals = primitive.hasX ? bufs[1] : bufs[0];
        const eVals = primitive.hasX ? bufs[2] : bufs[1];
        const n = Math.min(yVals ? yVals.length : 0, eVals ? eVals.length : 0);
        for (let i = 0; i < n; i += 1) {
          const x = xVals ? xVals[i] : i;
          const y = yVals[i];
          const e = Math.abs(eVals[i]);
          updatePoint(x, y - e);
          updatePoint(x, y + e);
        }
      } else if (kind === "error_bars_h") {
        const xVals = bufs[0];
        const eVals = bufs[1];
        const yVals = bufs[2];
        const n = Math.min(
          xVals ? xVals.length : 0,
          eVals ? eVals.length : 0,
          yVals ? yVals.length : 0,
        );
        for (let i = 0; i < n; i += 1) {
          const x = xVals[i];
          const y = yVals[i];
          const e = Math.abs(eVals[i]);
          updatePoint(x - e, y);
          updatePoint(x + e, y);
        }
      } else if (kind === "histogram") {
        const edges = bufs[0];
        const counts = bufs[1];
        if (edges && edges.length > 0) {
          for (let i = 0; i < edges.length; i += 1) {
            const x = edges[i];
            if (!Number.isFinite(x)) continue;
            hasData = true;
            if (x < xMin) xMin = x;
            if (x > xMax) xMax = x;
          }
        }
        if (counts) {
          for (let i = 0; i < counts.length; i += 1) {
            const c = counts[i];
            if (!Number.isFinite(c)) continue;
            hasData = true;
            if (0 < yMin) yMin = 0;
            if (c > yMax) yMax = c;
          }
        }
      } else if (kind === "histogram2d") {
        const xEdges = bufs[0];
        const yEdges = bufs[1];
        if (xEdges && xEdges.length > 0 && yEdges && yEdges.length > 0) {
          hasData = true;
          xMin = Math.min(xMin, xEdges[0]);
          xMax = Math.max(xMax, xEdges[xEdges.length - 1]);
          yMin = Math.min(yMin, yEdges[0]);
          yMax = Math.max(yMax, yEdges[yEdges.length - 1]);
        }
      } else if (kind === "heatmap") {
        const rows = Math.max(0, primitive.rows | 0);
        const cols = Math.max(0, primitive.cols | 0);
        if (rows > 0 && cols > 0) {
          hasData = true;
          xMin = Math.min(xMin, 0);
          xMax = Math.max(xMax, cols);
          yMin = Math.min(yMin, 0);
          yMax = Math.max(yMax, rows);
        }
      } else if (kind === "image") {
        const rows = Math.max(0, primitive.rows | 0);
        const cols = Math.max(0, primitive.cols | 0);
        if (rows > 0 && cols > 0) {
          const bx0 = Number.isFinite(Number(primitive.boundsXMin))
            ? Number(primitive.boundsXMin)
            : 0;
          const bx1 = Number.isFinite(Number(primitive.boundsXMax))
            ? Number(primitive.boundsXMax)
            : cols;
          const by0 = Number.isFinite(Number(primitive.boundsYMin))
            ? Number(primitive.boundsYMin)
            : 0;
          const by1 = Number.isFinite(Number(primitive.boundsYMax))
            ? Number(primitive.boundsYMax)
            : rows;
          hasData = true;
          xMin = Math.min(xMin, bx0, bx1);
          xMax = Math.max(xMax, bx0, bx1);
          yMin = Math.min(yMin, by0, by1);
          yMax = Math.max(yMax, by0, by1);
        }
      } else if (kind === "inf_lines") {
        const vals = bufs[0];
        if (!vals) continue;
        for (let i = 0; i < vals.length; i += 1) {
          const v = vals[i];
          if (!Number.isFinite(v)) continue;
          hasData = true;
          if (primitive.axis === "y") {
            yMin = Math.min(yMin, v);
            yMax = Math.max(yMax, v);
          } else {
            xMin = Math.min(xMin, v);
            xMax = Math.max(xMax, v);
          }
        }
      } else if (kind === "pie_chart") {
        const cx = Number(primitive.x || 0);
        const cy = Number(primitive.y || 0);
        const r = Math.abs(Number(primitive.radius || 1));
        updatePoint(cx - r, cy - r);
        updatePoint(cx + r, cy + r);
      } else if (kind === "text" || kind === "annotation") {
        updatePoint(Number(primitive.x), Number(primitive.y));
      }
    }

    if (!hasData) {
      yMin = -1;
      yMax = 1;
      xMin = 0;
      xMax = 1;
    }

    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
      yMin = -1;
      yMax = 1;
    } else if (yMin === yMax) {
      const pad = Math.abs(yMin) * 0.05 || 1;
      yMin -= pad;
      yMax += pad;
    }

    if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) {
      xMin = 0;
      xMax = 1;
    } else if (xMin === xMax) {
      const pad = Math.abs(xMin) * 0.05 || 1;
      xMin -= pad;
      xMax += pad;
    }

    return {
      xMin,
      xMax,
      yMin,
      yMax,
    };
  }

  _makeContextAction(label, action) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.style.display = "block";
    button.style.width = "100%";
    button.style.textAlign = "left";
    button.style.background = "#ffffff";
    button.style.border = "1px solid transparent";
    button.style.borderRadius = "4px";
    button.style.padding = "6px 8px";
    button.style.cursor = "pointer";
    button.style.color = "#18181b";
    button.style.fontSize = "12px";
    button.onmouseenter = () => {
      button.style.background = "#f4f4f5";
      button.style.borderColor = "#e4e4e7";
    };
    button.onmouseleave = () => {
      button.style.background = "#ffffff";
      button.style.borderColor = "transparent";
    };
    button.onclick = () => {
      this._hideContextMenu();
      action();
      this._emitViewChange();
      this._markDirty();
    };
    return button;
  }

  _showContextMenu(pos, region) {
    if (!this.contextMenu) {
      return;
    }
    this.contextMenu.replaceChildren();

    const bounds = this._computeAutoBounds();
    const actions = [
      this._makeContextAction("Autoscale", () => this._autoscale()),
    ];

    if (region === "plot" || region === "x-axis") {
      actions.push(
        this._makeContextAction("Reset X Axis", () => {
          this.view.xMin = bounds.xMin;
          this.view.xMax = bounds.xMax;
          this.view.initialized = true;
        }),
      );
    }
    if (region === "plot" || region === "y-axis") {
      actions.push(
        this._makeContextAction("Reset Y Axis", () => {
          this.view.yMin = bounds.yMin;
          this.view.yMax = bounds.yMax;
          this.view.initialized = true;
        }),
      );
    }

    for (const action of actions) {
      this.contextMenu.appendChild(action);
    }

    this.contextMenu.style.display = "block";
    const wrapperRect = this.wrapper.getBoundingClientRect();
    const menuRect = this.contextMenu.getBoundingClientRect();
    const left = clamp(pos.x, 4, Math.max(4, wrapperRect.width - menuRect.width - 4));
    const top = clamp(pos.y, 4, Math.max(4, wrapperRect.height - menuRect.height - 4));
    this.contextMenu.style.left = `${left}px`;
    this.contextMenu.style.top = `${top}px`;
  }

  _hideContextMenu() {
    if (!this.contextMenu || this.contextMenu.style.display === "none") {
      return;
    }
    this.contextMenu.style.display = "none";
    this.contextMenu.replaceChildren();
  }

  _pointerPosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  _insidePlot(pos) {
    const rect = this._plotRect();
    return (
      pos.x >= rect.left &&
      pos.x <= rect.left + rect.width &&
      pos.y >= rect.top &&
      pos.y <= rect.top + rect.height
    );
  }

  _insideCanvas(pos) {
    return pos.x >= 0 && pos.x <= this.cssWidth && pos.y >= 0 && pos.y <= this.cssHeight;
  }

  _handleMouseDown(event) {
    const pos = this._pointerPosition(event);
    const region = this._interactionRegion(pos);
    if (region !== "outside" && (event.button === 0 || event.button === 1 || event.button === 2)) {
      this._cancelInitialAutoFit();
      this.initialAutoFitActive = false;
    }

    if (this.implotEnabled && this.wasmReady) {
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      const btn = domButtonToImGuiButton(event.button);
      if (btn >= 0) {
        this.wasm.setMouseButton(btn, true);
      }
      if (event.button === 2) {
        event.preventDefault();
      }
      this._flushInputFrame();
      return;
    }

    this._hideContextMenu();
    if (event.button === 2) {
      event.preventDefault();
      this.rightButton.down = true;
      this.rightButton.startX = pos.x;
      this.rightButton.startY = pos.y;
      this.rightButton.moved = false;
      this.rightButton.region = region;
      this.boxZoom.active = false;
      this.drag.active = false;
      return;
    }
    if (region !== "plot") {
      return;
    }
    if (event.button !== 0) {
      return;
    }

    this.drag.active = true;
    this.drag.lastX = pos.x;
    this.drag.lastY = pos.y;
  }

  _handleMouseMove(event) {
    if (this.implotEnabled && this.wasmReady) {
      const pos = this._pointerPosition(event);
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      this._markDirty();
      return;
    }

    const pos = this._pointerPosition(event);
    this.mouse.x = pos.x;
    this.mouse.y = pos.y;
    this.mouse.inside = this._insidePlot(pos);

    if (this.rightButton.down) {
      const moveX = Math.abs(pos.x - this.rightButton.startX);
      const moveY = Math.abs(pos.y - this.rightButton.startY);
      if (moveX > 3 || moveY > 3) {
        this.rightButton.moved = true;
      }

      if (this.rightButton.region === "plot" && this.rightButton.moved) {
        if (!this.boxZoom.active) {
          this.boxZoom.active = true;
          this.boxZoom.startX = this.rightButton.startX;
          this.boxZoom.startY = this.rightButton.startY;
        }
        this.boxZoom.endX = pos.x;
        this.boxZoom.endY = pos.y;
      }
      this._markDirty();
      return;
    }

    if (this.boxZoom.active) {
      this.boxZoom.endX = pos.x;
      this.boxZoom.endY = pos.y;
      this._markDirty();
      return;
    }

    if (!this.drag.active) {
      this._markDirty();
      return;
    }

    const rect = this._plotRect();
    const dx = pos.x - this.drag.lastX;
    const dy = pos.y - this.drag.lastY;
    this.drag.lastX = pos.x;
    this.drag.lastY = pos.y;

    const xRange = this.view.xMax - this.view.xMin;
    const yRange = this.view.yMax - this.view.yMin;
    this.view.xMin -= (dx / rect.width) * xRange;
    this.view.xMax -= (dx / rect.width) * xRange;
    this.view.yMin += (dy / rect.height) * yRange;
    this.view.yMax += (dy / rect.height) * yRange;
    this.view.initialized = true;

    this._markDirty();
  }

  _handleMouseUp(event) {
    if (this.implotEnabled && this.wasmReady) {
      const pos = this._pointerPosition(event);
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      if (event) {
        const btn = domButtonToImGuiButton(event.button);
        if (btn >= 0) {
          this.wasm.setMouseButton(btn, false);
        }
        if (event.button === 2) {
          event.preventDefault();
        }
      }
      this._flushInputFrame();
      return;
    }

    if (event && event.button === 2 && this.rightButton.down) {
      event.preventDefault();
      const pos = this._pointerPosition(event);
      const region = this.rightButton.region === "outside" ? this._interactionRegion(pos) : this.rightButton.region;
      const shouldBoxZoom = this.boxZoom.active && this.rightButton.moved;
      this.rightButton.down = false;
      if (shouldBoxZoom) {
        if (this._applyBoxZoom()) {
          this._emitViewChange();
        }
      } else if (region !== "outside") {
        this._showContextMenu(pos, region);
      }
      this.boxZoom.active = false;
      this._markDirty();
      return;
    }

    if (this.boxZoom.active) {
      if (event && event.button === 2) {
        event.preventDefault();
      }
      if (this._applyBoxZoom()) {
        this._emitViewChange();
      }
      this.boxZoom.active = false;
      this._markDirty();
      return;
    }
    if (this.drag.active) {
      this._emitViewChange();
    }
    this.drag.active = false;
  }

  _handleMouseLeave() {
    if (this.implotEnabled && this.wasmReady) {
      this.wasm.setMousePos(0, 0, false);
      this._markDirty();
      return;
    }

    this.mouse.inside = false;
    if (this.rightButton.down && this.boxZoom.active) {
      this._markDirty();
      return;
    }
    this._markDirty();
  }

  _handleWheel(event) {
    if (this.implotEnabled && this.wasmReady) {
      event.preventDefault();
      const pos = this._pointerPosition(event);
      this._cancelInitialAutoFit();
      this.initialAutoFitActive = false;
      this.wasm.setMousePos(pos.x, pos.y, this._insideCanvas(pos));
      const wheelScale = event.deltaMode === 1 ? 1.0 : event.deltaMode === 2 ? 12.0 : 0.01;
      this.wasm.addMouseWheel(-event.deltaX * wheelScale, -event.deltaY * wheelScale);
      this._flushInputFrame();
      return;
    }

    this._hideContextMenu();
    const pos = this._pointerPosition(event);
    const region = this._interactionRegion(pos);
    if (region === "outside") {
      return;
    }
    this._cancelInitialAutoFit();
    this.initialAutoFitActive = false;
    event.preventDefault();

    const rect = this._plotRect();
    const xRange = Math.max(1e-9, this.view.xMax - this.view.xMin);
    const yRange = Math.max(1e-9, this.view.yMax - this.view.yMin);
    const xNorm = clamp((pos.x - rect.left) / rect.width, 0, 1);
    const yNorm = clamp((pos.y - rect.top) / rect.height, 0, 1);
    const xAnchor = this.view.xMin + xNorm * xRange;
    const yAnchor = this.view.yMax - yNorm * yRange;
    const zoom = Math.exp(event.deltaY * 0.0012);

    if (region === "plot" || region === "x-axis") {
      this.view.xMin = xAnchor + (this.view.xMin - xAnchor) * zoom;
      this.view.xMax = xAnchor + (this.view.xMax - xAnchor) * zoom;
    }
    if (region === "plot" || region === "y-axis") {
      this.view.yMin = yAnchor + (this.view.yMin - yAnchor) * zoom;
      this.view.yMax = yAnchor + (this.view.yMax - yAnchor) * zoom;
    }
    this.view.initialized = true;

    this._emitViewChange();
    this._markDirty();
  }

  _applyBoxZoom() {
    const rect = this._plotRect();
    const x0 = clamp(Math.min(this.boxZoom.startX, this.boxZoom.endX), rect.left, rect.left + rect.width);
    const x1 = clamp(Math.max(this.boxZoom.startX, this.boxZoom.endX), rect.left, rect.left + rect.width);
    const y0 = clamp(Math.min(this.boxZoom.startY, this.boxZoom.endY), rect.top, rect.top + rect.height);
    const y1 = clamp(Math.max(this.boxZoom.startY, this.boxZoom.endY), rect.top, rect.top + rect.height);
    if (x1 - x0 < 4 || y1 - y0 < 4) {
      return false;
    }

    const xSpan = Math.max(1e-9, this.view.xMax - this.view.xMin);
    const ySpan = Math.max(1e-9, this.view.yMax - this.view.yMin);

    const newXMin = this.view.xMin + ((x0 - rect.left) / rect.width) * xSpan;
    const newXMax = this.view.xMin + ((x1 - rect.left) / rect.width) * xSpan;
    const newYMax = this.view.yMax - ((y0 - rect.top) / rect.height) * ySpan;
    const newYMin = this.view.yMax - ((y1 - rect.top) / rect.height) * ySpan;

    if (!Number.isFinite(newXMin) || !Number.isFinite(newXMax) || !Number.isFinite(newYMin) || !Number.isFinite(newYMax)) {
      return false;
    }
    if (newXMax - newXMin < 1e-9 || newYMax - newYMin < 1e-9) {
      return false;
    }

    this.view.xMin = newXMin;
    this.view.xMax = newXMax;
    this.view.yMin = newYMin;
    this.view.yMax = newYMax;
    this.view.initialized = true;
    return true;
  }

  dispose() {
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    if (this.rafId !== 0) {
      window.cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
    if (this.implotEnableRetryTimer !== 0) {
      window.clearTimeout(this.implotEnableRetryTimer);
      this.implotEnableRetryTimer = 0;
    }

    this.model.off("change:width", this.onWidthChange);
    this.model.off("change:height", this.onHeightChange);
    this.model.off("change:title", this.onTitleChange);
    this.model.off("msg:custom", this.onCustomMessage);

    this.canvas.removeEventListener("mousedown", this.onMouseDown);
    this.canvas.removeEventListener("mousemove", this.onMouseMove);
    this.canvas.removeEventListener("mouseleave", this.onMouseLeave);
    this.canvas.removeEventListener("wheel", this.onWheel);
    this.canvas.removeEventListener("contextmenu", this.onContextMenu);
    this.canvas.removeEventListener("dblclick", this.onDoubleClick);
    window.removeEventListener("mousemove", this.onMouseMove);
    window.removeEventListener("mouseup", this.onMouseUp);
    window.removeEventListener("mousedown", this.onWindowMouseDown);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("resize", this.onWindowResize);
    window.removeEventListener(WASM_ASSET_EVENT, this.onWasmAssetsReady);

    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    this.wasm.destroy();

    this.series.clear();
    this.seriesOrder = [];
    this.primitives.clear();
    this.primitiveOrder = [];
    this.primitiveTokenById.clear();
    this.colorBySlot = [];
    this.el.innerHTML = "";
  }
}

function render({ model, el }) {
  const runtime = new PlotRuntime({ model, el });
  return () => runtime.dispose();
}

export default { render };
