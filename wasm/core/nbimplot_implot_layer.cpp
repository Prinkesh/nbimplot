#include "nbimplot_implot_layer.h"

#if NBIMPLOT_WITH_IMPLOT
#include <GLES3/gl3.h>
#include <emscripten/html5.h>
#include "imgui.h"
#include "implot.h"
#include "backends/imgui_impl_opengl3.h"
#endif

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nbimplot {

#if NBIMPLOT_WITH_IMPLOT
namespace {

struct LayerState {
  ImGuiContext *imgui_ctx = nullptr;
  ImPlotContext *implot_ctx = nullptr;
  EMSCRIPTEN_WEBGL_CONTEXT_HANDLE gl_ctx = 0;
  bool initialized = false;
  std::int32_t width = 900;
  std::int32_t height = 450;
  float dpr = 1.0f;
  float mouse_x = 0.0f;
  float mouse_y = 0.0f;
  bool mouse_inside = false;
  std::array<bool, 5> mouse_down = {false, false, false, false, false};
  float wheel_x = 0.0f;
  float wheel_y = 0.0f;
  struct ImageTextureEntry {
    GLuint texture = 0;
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::int32_t channels = 1;
    std::int32_t version = 0;
    std::uint32_t data_len = 0;
    std::int32_t colormap_idx = -1;
  };
  std::unordered_map<std::uint32_t, ImageTextureEntry> image_textures;
  std::unordered_set<std::uint32_t> image_textures_touched;
};

std::unordered_map<const ImPlotLayer *, LayerState> g_states;

LayerState *get_state(const ImPlotLayer *layer) {
  const auto it = g_states.find(layer);
  if (it == g_states.end()) {
    return nullptr;
  }
  return &it->second;
}

ImVec4 color_for_slot(std::int32_t slot) {
  static const std::array<ImVec4, 8> kSeriesColors = {
      ImVec4(0.114f, 0.306f, 0.847f, 1.0f), // #1d4ed8
      ImVec4(0.918f, 0.345f, 0.047f, 1.0f), // #ea580c
      ImVec4(0.086f, 0.639f, 0.290f, 1.0f), // #16a34a
      ImVec4(0.745f, 0.071f, 0.235f, 1.0f), // #be123c
      ImVec4(0.486f, 0.227f, 0.918f, 1.0f), // #7c3aed
      ImVec4(0.059f, 0.463f, 0.431f, 1.0f), // #0f766e
      ImVec4(0.792f, 0.541f, 0.016f, 1.0f), // #ca8a04
      ImVec4(0.012f, 0.412f, 0.631f, 1.0f), // #0369a1
  };
  const std::size_t idx = static_cast<std::size_t>(std::abs(slot)) % kSeriesColors.size();
  return kSeriesColors[idx];
}

std::vector<std::string> split_labels(const char *text, char delim) {
  std::vector<std::string> out;
  if (text == nullptr || text[0] == '\0') {
    return out;
  }
  std::string current;
  for (const char *p = text; *p != '\0'; ++p) {
    if (*p == delim) {
      out.push_back(current);
      current.clear();
    } else {
      current.push_back(*p);
    }
  }
  out.push_back(current);
  return out;
}

bool compute_value_range(const float *values, std::uint32_t len, double &out_min,
                         double &out_max) {
  if (values == nullptr || len == 0U) {
    return false;
  }
  double min_v = std::numeric_limits<double>::infinity();
  double max_v = -std::numeric_limits<double>::infinity();
  bool has = false;
  for (std::uint32_t i = 0; i < len; ++i) {
    const double v = static_cast<double>(values[i]);
    if (!std::isfinite(v)) {
      continue;
    }
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
    has = true;
  }
  if (!has || !std::isfinite(min_v) || !std::isfinite(max_v)) {
    return false;
  }
  out_min = min_v;
  out_max = max_v;
  return true;
}

std::uint8_t to_u8_channel(float v) {
  if (!std::isfinite(static_cast<double>(v))) {
    return 0U;
  }
  if (v >= 0.0f && v <= 1.0f) {
    return static_cast<std::uint8_t>(
        std::lround(static_cast<double>(v) * 255.0));
  }
  return static_cast<std::uint8_t>(
      std::lround(std::clamp(static_cast<double>(v), 0.0, 255.0)));
}

void destroy_image_texture(LayerState::ImageTextureEntry &entry) {
  if (entry.texture != 0U) {
    glDeleteTextures(1, &entry.texture);
    entry.texture = 0;
  }
  entry.rows = 0;
  entry.cols = 0;
  entry.channels = 1;
  entry.version = 0;
  entry.data_len = 0U;
  entry.colormap_idx = -1;
}

GLuint ensure_image_texture(LayerState &state, const PrimitiveView &p) {
  if (p.data0 == nullptr || p.i1 <= 0 || p.i2 <= 0) {
    return 0U;
  }
  const std::int32_t rows = p.i1;
  const std::int32_t cols = p.i2;
  std::int32_t channels = p.i6;
  if (channels != 1 && channels != 3 && channels != 4) {
    channels = 1;
  }
  const std::uint32_t expected_len =
      static_cast<std::uint32_t>(std::max(0, rows)) *
      static_cast<std::uint32_t>(std::max(0, cols)) *
      static_cast<std::uint32_t>(channels);
  if (expected_len == 0U || p.len0 < expected_len) {
    return 0U;
  }

  LayerState::ImageTextureEntry &entry = state.image_textures[p.id];
  state.image_textures_touched.insert(p.id);
  const std::int32_t active_colormap_idx =
      static_cast<std::int32_t>(ImPlot::GetStyle().Colormap);
  const bool colormap_changed_for_scalar =
      (channels == 1) && (entry.colormap_idx != active_colormap_idx);
  const bool needs_upload =
      entry.texture == 0U || entry.rows != rows || entry.cols != cols ||
      entry.channels != channels || entry.version != p.i3 ||
      entry.data_len != p.len0 || colormap_changed_for_scalar;
  if (!needs_upload) {
    return entry.texture;
  }

  if (entry.texture == 0U) {
    glGenTextures(1, &entry.texture);
    if (entry.texture == 0U) {
      return 0U;
    }
  }
  glBindTexture(GL_TEXTURE_2D, entry.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  const std::size_t pixel_count =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  std::vector<std::uint8_t> rgba(pixel_count * 4U, 0U);
  const float *src = p.data0;

  if (channels == 1) {
    double min_v = 0.0;
    double max_v = 1.0;
    const bool has_range = compute_value_range(src, expected_len, min_v, max_v);
    const double denom =
        has_range ? std::max(1e-12, max_v - min_v) : 1.0;
    for (std::size_t i = 0; i < pixel_count; ++i) {
      const float v = src[i];
      if (!std::isfinite(static_cast<double>(v))) {
        rgba[i * 4U + 3U] = 0U;
        continue;
      }
      double t = has_range ? (static_cast<double>(v) - min_v) / denom : 0.5;
      t = std::clamp(t, 0.0, 1.0);
      const ImVec4 mapped = ImPlot::SampleColormap(static_cast<float>(t));
      rgba[i * 4U + 0U] = to_u8_channel(mapped.x);
      rgba[i * 4U + 1U] = to_u8_channel(mapped.y);
      rgba[i * 4U + 2U] = to_u8_channel(mapped.z);
      rgba[i * 4U + 3U] = to_u8_channel(mapped.w);
    }
  } else {
    const std::size_t src_stride = static_cast<std::size_t>(channels);
    for (std::size_t i = 0; i < pixel_count; ++i) {
      const std::size_t src_idx = i * src_stride;
      const float r = src[src_idx + 0U];
      const float g = src[src_idx + 1U];
      const float b = src[src_idx + 2U];
      const float a = channels == 4 ? src[src_idx + 3U] : 1.0f;
      const bool finite_rgb =
          std::isfinite(static_cast<double>(r)) &&
          std::isfinite(static_cast<double>(g)) &&
          std::isfinite(static_cast<double>(b));
      if (!finite_rgb) {
        rgba[i * 4U + 3U] = 0U;
        continue;
      }
      rgba[i * 4U + 0U] = to_u8_channel(r);
      rgba[i * 4U + 1U] = to_u8_channel(g);
      rgba[i * 4U + 2U] = to_u8_channel(b);
      rgba[i * 4U + 3U] = to_u8_channel(a);
    }
  }

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols, rows, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, rgba.data());

  entry.rows = rows;
  entry.cols = cols;
  entry.channels = channels;
  entry.version = p.i3;
  entry.data_len = p.len0;
  entry.colormap_idx = channels == 1 ? active_colormap_idx : -1;
  return entry.texture;
}

std::string format_text(const char *fmt, ...) {
  char buffer[256];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return std::string(buffer);
}

ImPlotColormap resolve_colormap_name(const char *name) {
  if (name == nullptr || name[0] == '\0') {
    return static_cast<ImPlotColormap>(-1);
  }
  ImPlotColormap idx = ImPlot::GetColormapIndex(name);
  if (static_cast<int>(idx) >= 0) {
    return idx;
  }
  std::string lower;
  for (const char *p = name; *p != '\0'; ++p) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
  }
  const int count = ImPlot::GetColormapCount();
  for (int i = 0; i < count; ++i) {
    const char *candidate = ImPlot::GetColormapName(i);
    if (candidate == nullptr || candidate[0] == '\0') {
      continue;
    }
    std::string cand_lower;
    for (const char *p = candidate; *p != '\0'; ++p) {
      cand_lower.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
    }
    if (cand_lower == lower) {
      return static_cast<ImPlotColormap>(i);
    }
  }
  return static_cast<ImPlotColormap>(-1);
}

int find_edge_bin(const float *edges, std::uint32_t len, double value) {
  if (edges == nullptr || len < 2U || !std::isfinite(value)) {
    return -1;
  }
  const double lo = static_cast<double>(edges[0]);
  const double hi = static_cast<double>(edges[len - 1U]);
  if (value < lo || value > hi) {
    return -1;
  }
  if (value == hi) {
    return static_cast<int>(len - 2U);
  }
  const auto it =
      std::upper_bound(edges, edges + static_cast<std::ptrdiff_t>(len),
                       static_cast<float>(value));
  if (it == edges) {
    return 0;
  }
  const int idx = static_cast<int>(it - edges - 1);
  return std::clamp(idx, 0, static_cast<int>(len - 2U));
}

bool is_finite_vec2(const ImVec2 &v) {
  return std::isfinite(static_cast<double>(v.x)) &&
         std::isfinite(static_cast<double>(v.y));
}

double distance2(const ImVec2 &a, const ImVec2 &b) {
  const double dx = static_cast<double>(a.x - b.x);
  const double dy = static_cast<double>(a.y - b.y);
  return dx * dx + dy * dy;
}

double point_segment_distance2(const ImVec2 &p, const ImVec2 &a,
                               const ImVec2 &b) {
  const double abx = static_cast<double>(b.x - a.x);
  const double aby = static_cast<double>(b.y - a.y);
  const double apx = static_cast<double>(p.x - a.x);
  const double apy = static_cast<double>(p.y - a.y);
  const double denom = abx * abx + aby * aby;
  if (denom <= 1e-12) {
    return distance2(p, a);
  }
  const double t = std::clamp((apx * abx + apy * aby) / denom, 0.0, 1.0);
  const ImVec2 q(static_cast<float>(static_cast<double>(a.x) + t * abx),
                 static_cast<float>(static_cast<double>(a.y) + t * aby));
  return distance2(p, q);
}

ImU32 with_alpha(ImU32 color, std::uint8_t alpha) {
  return (color & ~IM_COL32_A_MASK) | (static_cast<ImU32>(alpha) << IM_COL32_A_SHIFT);
}

struct TupleGetterData {
  const float *base = nullptr;
};

ImPlotPoint tuple_getter(int idx, void *user_data) {
  const TupleGetterData *data = static_cast<const TupleGetterData *>(user_data);
  const float *p = data->base + static_cast<std::size_t>(idx) * 4U;
  return ImPlotPoint(static_cast<double>(p[1]), static_cast<double>(p[2]));
}

constexpr std::int32_t kPrimScatter = 1;
constexpr std::int32_t kPrimBubbles = 2;
constexpr std::int32_t kPrimStairs = 3;
constexpr std::int32_t kPrimStems = 4;
constexpr std::int32_t kPrimDigital = 5;
constexpr std::int32_t kPrimBars = 6;
constexpr std::int32_t kPrimBarGroups = 7;
constexpr std::int32_t kPrimBarsH = 8;
constexpr std::int32_t kPrimShaded = 9;
constexpr std::int32_t kPrimErrorBars = 10;
constexpr std::int32_t kPrimErrorBarsH = 11;
constexpr std::int32_t kPrimInfLines = 12;
constexpr std::int32_t kPrimHistogram = 13;
constexpr std::int32_t kPrimHistogram2D = 14;
constexpr std::int32_t kPrimHeatmap = 15;
constexpr std::int32_t kPrimImage = 16;
constexpr std::int32_t kPrimPieChart = 17;
constexpr std::int32_t kPrimText = 18;
constexpr std::int32_t kPrimAnnotation = 19;
constexpr std::int32_t kPrimDummy = 20;
constexpr std::int32_t kPrimDragLineX = 21;
constexpr std::int32_t kPrimDragLineY = 22;
constexpr std::int32_t kPrimDragPoint = 23;
constexpr std::int32_t kPrimDragRect = 24;
constexpr std::int32_t kPrimTagX = 25;
constexpr std::int32_t kPrimTagY = 26;
constexpr std::int32_t kPrimColormapSlider = 27;
constexpr std::int32_t kPrimColormapButton = 28;
constexpr std::int32_t kPrimColormapSelector = 29;
constexpr std::int32_t kPrimDragDropPlot = 30;
constexpr std::int32_t kPrimDragDropAxis = 31;
constexpr std::int32_t kPrimDragDropLegend = 32;

const char *primitive_kind_name(std::int32_t kind) {
  switch (kind) {
  case kPrimScatter:
    return "scatter";
  case kPrimBubbles:
    return "bubbles";
  case kPrimStairs:
    return "stairs";
  case kPrimStems:
    return "stems";
  case kPrimDigital:
    return "digital";
  case kPrimBars:
    return "bars";
  case kPrimBarGroups:
    return "bar_groups";
  case kPrimBarsH:
    return "bars_h";
  case kPrimShaded:
    return "shaded";
  case kPrimErrorBars:
    return "error_bars";
  case kPrimErrorBarsH:
    return "error_bars_h";
  case kPrimInfLines:
    return "inf_lines";
  case kPrimHistogram:
    return "histogram";
  case kPrimHistogram2D:
    return "histogram2d";
  case kPrimHeatmap:
    return "heatmap";
  case kPrimImage:
    return "image";
  case kPrimPieChart:
    return "pie_chart";
  case kPrimText:
    return "text";
  case kPrimAnnotation:
    return "annotation";
  case kPrimDummy:
    return "dummy";
  case kPrimDragLineX:
    return "drag_line_x";
  case kPrimDragLineY:
    return "drag_line_y";
  case kPrimDragPoint:
    return "drag_point";
  case kPrimDragRect:
    return "drag_rect";
  case kPrimTagX:
    return "tag_x";
  case kPrimTagY:
    return "tag_y";
  case kPrimColormapSlider:
    return "colormap_slider";
  case kPrimColormapButton:
    return "colormap_button";
  case kPrimColormapSelector:
    return "colormap_selector";
  case kPrimDragDropPlot:
    return "drag_drop_plot";
  case kPrimDragDropAxis:
    return "drag_drop_axis";
  case kPrimDragDropLegend:
    return "drag_drop_legend";
  default:
    return "primitive";
  }
}

void set_contexts(const LayerState &state) {
  ImGui::SetCurrentContext(state.imgui_ctx);
  ImPlot::SetCurrentContext(state.implot_ctx);
}

void destroy_state(LayerState &state) {
  if (!state.initialized) {
    return;
  }
  set_contexts(state);
  for (auto &it : state.image_textures) {
    destroy_image_texture(it.second);
  }
  state.image_textures.clear();
  state.image_textures_touched.clear();
  ImGui_ImplOpenGL3_Shutdown();
  if (state.gl_ctx > 0) {
    emscripten_webgl_destroy_context(state.gl_ctx);
    state.gl_ctx = 0;
  }
  ImPlot::DestroyContext(state.implot_ctx);
  ImGui::DestroyContext(state.imgui_ctx);
  state.implot_ctx = nullptr;
  state.imgui_ctx = nullptr;
  state.initialized = false;
}

} // namespace
#endif

bool ImPlotLayer::is_compiled() const noexcept {
#if NBIMPLOT_WITH_IMPLOT
  return true;
#else
  return false;
#endif
}

bool ImPlotLayer::initialize(const char *canvas_selector) noexcept {
#if NBIMPLOT_WITH_IMPLOT
  if (canvas_selector == nullptr || canvas_selector[0] == '\0') {
    return false;
  }

  LayerState &state = g_states[this];
  if (state.initialized) {
    return true;
  }

  IMGUI_CHECKVERSION();
  state.imgui_ctx = ImGui::CreateContext();
  if (state.imgui_ctx == nullptr) {
    return false;
  }
  ImGui::SetCurrentContext(state.imgui_ctx);

  state.implot_ctx = ImPlot::CreateContext();
  if (state.implot_ctx == nullptr) {
    ImGui::DestroyContext(state.imgui_ctx);
    state.imgui_ctx = nullptr;
    return false;
  }
  ImPlot::SetCurrentContext(state.implot_ctx);

  EmscriptenWebGLContextAttributes attrs;
  emscripten_webgl_init_context_attributes(&attrs);
  attrs.alpha = true;
  attrs.depth = true;
  attrs.stencil = true;
  attrs.antialias = true;
  attrs.majorVersion = 2;
  attrs.minorVersion = 0;
  attrs.enableExtensionsByDefault = true;
  attrs.premultipliedAlpha = false;

  state.gl_ctx = emscripten_webgl_create_context(canvas_selector, &attrs);
  if (state.gl_ctx <= 0) {
    ImPlot::DestroyContext(state.implot_ctx);
    ImGui::DestroyContext(state.imgui_ctx);
    state.implot_ctx = nullptr;
    state.imgui_ctx = nullptr;
    return false;
  }

  if (emscripten_webgl_make_context_current(state.gl_ctx) != EMSCRIPTEN_RESULT_SUCCESS) {
    emscripten_webgl_destroy_context(state.gl_ctx);
    state.gl_ctx = 0;
    ImPlot::DestroyContext(state.implot_ctx);
    ImGui::DestroyContext(state.imgui_ctx);
    state.implot_ctx = nullptr;
    state.imgui_ctx = nullptr;
    return false;
  }

  ImGuiIO &io = ImGui::GetIO();
  io.BackendPlatformName = "nbimplot_emscripten";
  io.BackendRendererName = "imgui_impl_opengl3";
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  if (!ImGui_ImplOpenGL3_Init("#version 300 es")) {
    destroy_state(state);
    return false;
  }

  ImGui::StyleColorsLight();
  state.initialized = true;
  return true;
#else
  (void)canvas_selector;
  return false;
#endif
}

void ImPlotLayer::shutdown() noexcept {
#if NBIMPLOT_WITH_IMPLOT
  const auto it = g_states.find(this);
  if (it == g_states.end()) {
    return;
  }
  LayerState &state = it->second;
  destroy_state(state);
  g_states.erase(it);
#endif
}

void ImPlotLayer::set_canvas(std::int32_t width, std::int32_t height,
                             float dpr) noexcept {
#if NBIMPLOT_WITH_IMPLOT
  LayerState *state = get_state(this);
  if (state == nullptr) {
    return;
  }
  state->width = std::max(1, width);
  state->height = std::max(1, height);
  state->dpr = std::max(1.0f, dpr);
#else
  (void)width;
  (void)height;
  (void)dpr;
#endif
}

void ImPlotLayer::set_mouse_pos(float x, float y, bool inside) noexcept {
#if NBIMPLOT_WITH_IMPLOT
  LayerState *state = get_state(this);
  if (state == nullptr) {
    return;
  }
  state->mouse_x = x;
  state->mouse_y = y;
  state->mouse_inside = inside;
#else
  (void)x;
  (void)y;
  (void)inside;
#endif
}

void ImPlotLayer::set_mouse_button(std::int32_t button, bool down) noexcept {
#if NBIMPLOT_WITH_IMPLOT
  LayerState *state = get_state(this);
  if (state == nullptr || button < 0 || button >= static_cast<std::int32_t>(state->mouse_down.size())) {
    return;
  }
  state->mouse_down[static_cast<std::size_t>(button)] = down;
#else
  (void)button;
  (void)down;
#endif
}

void ImPlotLayer::add_mouse_wheel(float wheel_x, float wheel_y) noexcept {
#if NBIMPLOT_WITH_IMPLOT
  LayerState *state = get_state(this);
  if (state == nullptr) {
    return;
  }
  state->wheel_x += wheel_x;
  state->wheel_y += wheel_y;
#else
  (void)wheel_x;
  (void)wheel_y;
#endif
}

bool ImPlotLayer::render(const float *draw_points, std::uint32_t point_count,
                         const DrawSegmentView *segments,
                         std::uint32_t segment_count,
                         const SeriesView *series_views,
                         std::uint32_t series_count, float *x_min, float *x_max,
                         float *y_min, float *y_max, const char *title_id,
                         bool force_view, std::int32_t plot_flags,
                         const std::int32_t *axis_enabled6,
                         const std::int32_t *axis_scale6,
                         const char *const *axis_labels6,
                         const char *const *axis_formats6,
                         const std::int32_t *axis_limits_constraints_enabled6,
                         const double *axis_limits_constraints_min6,
                         const double *axis_limits_constraints_max6,
                         const std::int32_t *axis_zoom_constraints_enabled6,
                         const double *axis_zoom_constraints_min6,
                         const double *axis_zoom_constraints_max6,
                         const std::int32_t *axis_links6,
                         double *axis_view_min6, double *axis_view_max6,
                         const double *const *axis_ticks_values6,
                         const std::int32_t *axis_ticks_counts6,
                         const char *const *axis_ticks_labels6,
                         const std::int32_t *axis_ticks_keep_default6,
                         std::int32_t subplot_rows, std::int32_t subplot_cols,
                         std::int32_t subplot_flags,
                         std::int32_t aligned_group_enabled,
                         const char *aligned_group_id,
                         std::int32_t aligned_group_vertical,
                         const char *colormap_name, PrimitiveView *primitives,
                         float *selection_out6,
                         std::uint32_t primitive_count) noexcept {
#if NBIMPLOT_WITH_IMPLOT
  constexpr std::int32_t kPlotFlagNoLegend = 1 << 0;
  constexpr std::int32_t kPlotFlagNoMenus = 1 << 1;
  constexpr std::int32_t kPlotFlagNoBoxSelect = 1 << 2;
  constexpr std::int32_t kPlotFlagNoMousePos = 1 << 3;
  constexpr std::int32_t kPlotFlagCrosshairs = 1 << 4;
  constexpr std::int32_t kPlotFlagEqual = 1 << 5;

  constexpr std::int32_t kSubplotFlagNoLegend = 1 << 0;
  constexpr std::int32_t kSubplotFlagNoMenus = 1 << 1;
  constexpr std::int32_t kSubplotFlagNoResize = 1 << 2;
  constexpr std::int32_t kSubplotFlagNoAlign = 1 << 3;
  constexpr std::int32_t kSubplotFlagShareItems = 1 << 4;
  constexpr std::int32_t kSubplotFlagLinkRows = 1 << 5;
  constexpr std::int32_t kSubplotFlagLinkCols = 1 << 6;
  constexpr std::int32_t kSubplotFlagLinkAllX = 1 << 7;
  constexpr std::int32_t kSubplotFlagLinkAllY = 1 << 8;
  constexpr std::int32_t kSubplotFlagColMajor = 1 << 9;

  LayerState *state = get_state(this);
  if (state == nullptr || !state->initialized) {
    return false;
  }
  state->image_textures_touched.clear();
  if (selection_out6 != nullptr) {
    for (int i = 0; i < 6; ++i) {
      selection_out6[i] = 0.0f;
    }
  }
  if (emscripten_webgl_make_context_current(state->gl_ctx) != EMSCRIPTEN_RESULT_SUCCESS) {
    return false;
  }

  set_contexts(*state);
  ImGuiIO &io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(state->width),
                          static_cast<float>(state->height));
  io.DisplayFramebufferScale = ImVec2(state->dpr, state->dpr);
  io.DeltaTime = 1.0f / 60.0f;
  io.MousePos =
      state->mouse_inside ? ImVec2(state->mouse_x, state->mouse_y)
                          : ImVec2(-FLT_MAX, -FLT_MAX);
  for (std::size_t i = 0; i < state->mouse_down.size(); ++i) {
    io.MouseDown[i] = state->mouse_down[i];
  }
  io.MouseWheel += state->wheel_y;
  io.MouseWheelH += state->wheel_x;
  state->wheel_x = 0.0f;
  state->wheel_y = 0.0f;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui::NewFrame();

  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
  ImGui::SetNextWindowSize(
      ImVec2(static_cast<float>(state->width), static_cast<float>(state->height)),
      ImGuiCond_Always);
  const ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
      ImGuiWindowFlags_NoBackground;
  ImGui::Begin("##nbimplot_root", nullptr, window_flags);

  const char *plot_title =
      (title_id != nullptr && title_id[0] != '\0') ? title_id : "##nbimplot_plot";
  ImPlotFlags implot_flags = ImPlotFlags_None;
  if ((plot_flags & kPlotFlagNoLegend) != 0) {
    implot_flags |= ImPlotFlags_NoLegend;
  }
  if ((plot_flags & kPlotFlagNoMenus) != 0) {
    implot_flags |= ImPlotFlags_NoMenus;
  }
  if ((plot_flags & kPlotFlagNoBoxSelect) != 0) {
    implot_flags |= ImPlotFlags_NoBoxSelect;
  }
  if ((plot_flags & kPlotFlagNoMousePos) != 0) {
    implot_flags |= ImPlotFlags_NoMouseText;
  }
  if ((plot_flags & kPlotFlagCrosshairs) != 0) {
    implot_flags |= ImPlotFlags_Crosshairs;
  }
  if ((plot_flags & kPlotFlagEqual) != 0) {
    implot_flags |= ImPlotFlags_Equal;
  }

  ImPlotSubplotFlags implot_subplot_flags = ImPlotSubplotFlags_None;
  if ((subplot_flags & kSubplotFlagNoLegend) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_NoLegend;
  }
  if ((subplot_flags & kSubplotFlagNoMenus) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_NoMenus;
  }
  if ((subplot_flags & kSubplotFlagNoResize) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_NoResize;
  }
  if ((subplot_flags & kSubplotFlagNoAlign) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_NoAlign;
  }
  if ((subplot_flags & kSubplotFlagShareItems) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_ShareItems;
  }
  if ((subplot_flags & kSubplotFlagLinkRows) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_LinkRows;
  }
  if ((subplot_flags & kSubplotFlagLinkCols) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_LinkCols;
  }
  if ((subplot_flags & kSubplotFlagLinkAllX) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_LinkAllX;
  }
  if ((subplot_flags & kSubplotFlagLinkAllY) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_LinkAllY;
  }
  if ((subplot_flags & kSubplotFlagColMajor) != 0) {
    implot_subplot_flags |= ImPlotSubplotFlags_ColMajor;
  }

  bool pushed_colormap = false;
  if (colormap_name != nullptr && colormap_name[0] != '\0') {
    const ImPlotColormap cmap = resolve_colormap_name(colormap_name);
    if (static_cast<int>(cmap) >= 0) {
      ImPlot::PushColormap(cmap);
      pushed_colormap = true;
    }
  }

  std::unordered_map<std::int32_t, std::pair<std::uint32_t, std::uint32_t>>
      slot_segment_ranges;
  if (segments != nullptr && draw_points != nullptr) {
    std::int32_t current_slot = std::numeric_limits<std::int32_t>::min();
    std::uint32_t range_begin = 0;
    for (std::uint32_t si = 0; si < segment_count; ++si) {
      const DrawSegmentView &seg = segments[si];
      if (seg.count < 2 || seg.start >= point_count ||
          seg.count > point_count - seg.start) {
        continue;
      }
      if (seg.slot != current_slot) {
        if (current_slot != std::numeric_limits<std::int32_t>::min()) {
          slot_segment_ranges[current_slot] = {range_begin, si};
        }
        current_slot = seg.slot;
        range_begin = si;
      }
    }
    if (current_slot != std::numeric_limits<std::int32_t>::min()) {
      slot_segment_ranges[current_slot] = {range_begin, segment_count};
    }
  }

  auto draw_series_for_subplot =
      [&](std::int32_t subplot_index, std::vector<float> &tmp_x,
          std::vector<float> &tmp_y) {
        if (series_views == nullptr || draw_points == nullptr ||
            segments == nullptr) {
          return;
        }
        for (std::uint32_t svi = 0; svi < series_count; ++svi) {
          const SeriesView &series = series_views[svi];
          if (series.subplot_index != subplot_index) {
            continue;
          }
          const auto range_it = slot_segment_ranges.find(series.slot);
          if (range_it == slot_segment_ranges.end()) {
            continue;
          }
          const auto [seg_begin, seg_end] = range_it->second;
          tmp_x.clear();
          tmp_y.clear();
          for (std::uint32_t si = seg_begin; si < seg_end; ++si) {
            const DrawSegmentView &seg = segments[si];
            if (seg.slot != series.slot) {
              continue;
            }
            if (seg.count < 2 || seg.start >= point_count ||
                seg.count > point_count - seg.start) {
              continue;
            }
            if (!tmp_x.empty()) {
              const float nan = std::numeric_limits<float>::quiet_NaN();
              tmp_x.push_back(nan);
              tmp_y.push_back(nan);
            }
            const float *base =
                draw_points + static_cast<std::size_t>(seg.start) * 4U;
            for (std::uint32_t i = 0; i < seg.count; ++i) {
              const float *p = base + static_cast<std::size_t>(i) * 4U;
              tmp_x.push_back(p[1]);
              tmp_y.push_back(p[2]);
            }
          }
          if (tmp_x.empty()) {
            continue;
          }
          std::string display_label =
              (series.label != nullptr && series.label[0] != '\0')
                  ? std::string(series.label)
                  : ("series_" + std::to_string(series.slot));
          std::string item_label =
              display_label + "###s" + std::to_string(series.slot);
          const std::int32_t x_axis = std::clamp(series.x_axis, 0, 2);
          const std::int32_t y_axis = std::clamp(series.y_axis, 3, 5);
          ImPlot::SetAxes(static_cast<ImAxis>(x_axis),
                          static_cast<ImAxis>(y_axis));
          ImPlotSpec spec;
          if (series.has_custom_color != 0) {
            spec.LineColor = ImVec4(series.color_r, series.color_g, series.color_b,
                                    series.color_a);
            spec.MarkerLineColor = spec.LineColor;
            spec.MarkerFillColor = spec.LineColor;
          }
          if (std::isfinite(static_cast<double>(series.line_weight)) &&
              series.line_weight > 0.0f) {
            spec.LineWeight = series.line_weight;
          }
          spec.Marker = static_cast<ImPlotMarker>(std::clamp(series.marker, -2, 9));
          if (std::isfinite(static_cast<double>(series.marker_size)) &&
              series.marker_size > 0.0f) {
            spec.MarkerSize = series.marker_size;
          }
          ImPlot::PlotLine(item_label.c_str(), tmp_x.data(), tmp_y.data(),
                           static_cast<int>(tmp_x.size()), spec);
        }
      };

  auto draw_primitives_for_subplot = [&](std::int32_t subplot_index,
                                         std::vector<float> &tmp_x,
                                         std::vector<float> &tmp_y,
                                         std::vector<const char *> &label_ptrs,
                                         std::vector<std::string> &label_storage) {
    for (std::uint32_t pi = 0; pi < primitive_count; ++pi) {
      PrimitiveView &p = primitives[pi];
      if (p.i7 != subplot_index) {
        continue;
      }
      const bool has_x = p.i0 != 0;
      const std::int32_t x_axis = std::clamp(p.i4, 0, 2);
      const std::int32_t y_axis = std::clamp(p.i5, 3, 5);
      ImPlot::SetAxes(static_cast<ImAxis>(x_axis),
                      static_cast<ImAxis>(y_axis));
      char internal_id[64];
      std::snprintf(internal_id, sizeof(internal_id), "##p%u", p.id);
      const char *label = (p.text != nullptr && p.text[0] != '\0') ? p.text : internal_id;

      switch (p.kind) {
      case kPrimScatter: {
        if (has_x && p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            ImPlot::PlotScatter(label, p.data0, p.data1, n);
          }
        } else if (p.data0 != nullptr && p.len0 > 0) {
          ImPlot::PlotScatter(label, p.data0, static_cast<int>(p.len0), 1.0, 0.0);
        }
        break;
      }
      case kPrimBubbles: {
        if (has_x && p.data0 != nullptr && p.data1 != nullptr && p.data2 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, std::min(p.len1, p.len2)));
          if (n > 0) {
            ImPlot::PlotBubbles(label, p.data0, p.data1, p.data2, n);
          }
        } else if (p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            ImPlot::PlotBubbles(label, p.data0, p.data1, n, 1.0, 0.0);
          }
        }
        break;
      }
      case kPrimStairs: {
        if (has_x && p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            ImPlot::PlotStairs(label, p.data0, p.data1, n);
          }
        } else if (p.data0 != nullptr && p.len0 > 0) {
          ImPlot::PlotStairs(label, p.data0, static_cast<int>(p.len0), 1.0, 0.0);
        }
        break;
      }
      case kPrimStems: {
        if (has_x && p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            ImPlot::PlotStems(label, p.data0, p.data1, n, 0.0);
          }
        } else if (p.data0 != nullptr && p.len0 > 0) {
          ImPlot::PlotStems(label, p.data0, static_cast<int>(p.len0), 0.0, 1.0, 0.0);
        }
        break;
      }
      case kPrimDigital: {
        if (has_x && p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            ImPlot::PlotDigital(label, p.data0, p.data1, n);
          }
        } else if (p.data0 != nullptr && p.len0 > 0) {
          const int n = static_cast<int>(p.len0);
          tmp_x.resize(n);
          for (int i = 0; i < n; ++i) {
            tmp_x[static_cast<std::size_t>(i)] = static_cast<float>(i);
          }
          ImPlot::PlotDigital(label, tmp_x.data(), p.data0, n);
        }
        break;
      }
      case kPrimBars: {
        const double bar_size = std::max(1e-6f, p.f1 > 0.0f ? p.f1 : 0.67f);
        if (has_x && p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            ImPlot::PlotBars(label, p.data0, p.data1, n, bar_size);
          }
        } else if (p.data0 != nullptr && p.len0 > 0) {
          ImPlot::PlotBars(label, p.data0, static_cast<int>(p.len0), bar_size, 0.0);
        }
        break;
      }
      case kPrimBarGroups: {
        if (p.data0 == nullptr || p.i1 <= 0 || p.i2 <= 0) {
          break;
        }
        label_storage.clear();
        label_ptrs.clear();
        label_storage.reserve(static_cast<std::size_t>(p.i1));
        label_ptrs.reserve(static_cast<std::size_t>(p.i1));
        std::vector<std::string> custom_labels = split_labels(p.text, '\x1f');
        for (int i = 0; i < p.i1; ++i) {
          if (static_cast<std::size_t>(i) < custom_labels.size() &&
              !custom_labels[static_cast<std::size_t>(i)].empty()) {
            label_storage.push_back(custom_labels[static_cast<std::size_t>(i)]);
          } else {
            label_storage.push_back("g" + std::to_string(i));
          }
        }
        for (int i = 0; i < p.i1; ++i) {
          label_ptrs.push_back(label_storage[static_cast<std::size_t>(i)].c_str());
        }
        ImPlot::PlotBarGroups(label_ptrs.data(), p.data0, p.i1, p.i2,
                              p.f1 > 0.0f ? p.f1 : 0.67, p.f2);
        break;
      }
      case kPrimBarsH: {
        if (p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            const double bar_size = std::max(1e-6f, p.f2 > 0.0f ? p.f2 : 0.67f);
            ImPlot::PlotBars(label, p.data1, p.data0, n, bar_size);
          }
        }
        break;
      }
      case kPrimShaded: {
        if (has_x && p.data0 != nullptr && p.data1 != nullptr && p.data2 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, std::min(p.len1, p.len2)));
          if (n > 0) {
            ImPlot::PlotShaded(label, p.data0, p.data1, p.data2, n);
          }
        } else if (p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            tmp_x.resize(n);
            for (int i = 0; i < n; ++i) {
              tmp_x[static_cast<std::size_t>(i)] = static_cast<float>(i);
            }
            ImPlot::PlotShaded(label, tmp_x.data(), p.data0, p.data1, n);
          }
        }
        break;
      }
      case kPrimErrorBars: {
        const bool asym = p.i1 != 0;
        if (asym) {
          if (has_x && p.data0 != nullptr && p.data1 != nullptr && p.data2 != nullptr) {
            const int n =
                static_cast<int>(std::min(p.len0, std::min(p.len1, p.len2 / 2U)));
            if (n > 0) {
              tmp_x.resize(static_cast<std::size_t>(n));
              tmp_y.resize(static_cast<std::size_t>(n));
              for (int i = 0; i < n; ++i) {
                tmp_x[static_cast<std::size_t>(i)] = p.data2[static_cast<std::size_t>(i) * 2U];
                tmp_y[static_cast<std::size_t>(i)] =
                    p.data2[static_cast<std::size_t>(i) * 2U + 1U];
              }
              ImPlot::PlotErrorBars(label, p.data0, p.data1, tmp_x.data(),
                                    tmp_y.data(), n);
            }
          } else if (p.data0 != nullptr && p.data1 != nullptr) {
            const int n = static_cast<int>(std::min(p.len0, p.len1 / 2U));
            if (n > 0) {
              tmp_x.resize(static_cast<std::size_t>(n));
              tmp_y.resize(static_cast<std::size_t>(n));
              std::vector<float> idx_x(static_cast<std::size_t>(n));
              for (int i = 0; i < n; ++i) {
                idx_x[static_cast<std::size_t>(i)] = static_cast<float>(i);
                tmp_x[static_cast<std::size_t>(i)] = p.data1[static_cast<std::size_t>(i) * 2U];
                tmp_y[static_cast<std::size_t>(i)] =
                    p.data1[static_cast<std::size_t>(i) * 2U + 1U];
              }
              ImPlot::PlotErrorBars(label, idx_x.data(), p.data0, tmp_x.data(),
                                    tmp_y.data(), n);
            }
          }
        } else if (has_x && p.data0 != nullptr && p.data1 != nullptr &&
                   p.data2 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, std::min(p.len1, p.len2)));
          if (n > 0) {
            ImPlot::PlotErrorBars(label, p.data0, p.data1, p.data2, n);
          }
        } else if (p.data0 != nullptr && p.data1 != nullptr) {
          const int n = static_cast<int>(std::min(p.len0, p.len1));
          if (n > 0) {
            tmp_x.resize(n);
            for (int i = 0; i < n; ++i) {
              tmp_x[static_cast<std::size_t>(i)] = static_cast<float>(i);
            }
            ImPlot::PlotErrorBars(label, tmp_x.data(), p.data0, p.data1, n);
          }
        }
        break;
      }
      case kPrimErrorBarsH: {
        if (p.data0 != nullptr && p.data1 != nullptr && p.data2 != nullptr) {
          const bool asym = p.i1 != 0;
          if (asym) {
            const int n =
                static_cast<int>(std::min(p.len0, std::min(p.len2, p.len1 / 2U)));
            if (n > 0) {
              tmp_x.resize(static_cast<std::size_t>(n));
              tmp_y.resize(static_cast<std::size_t>(n));
              for (int i = 0; i < n; ++i) {
                tmp_x[static_cast<std::size_t>(i)] = p.data1[static_cast<std::size_t>(i) * 2U];
                tmp_y[static_cast<std::size_t>(i)] =
                    p.data1[static_cast<std::size_t>(i) * 2U + 1U];
              }
              ImPlot::PlotErrorBars(label, p.data2, p.data0, tmp_x.data(),
                                    tmp_y.data(), n);
            }
          } else {
            const int n = static_cast<int>(std::min(p.len0, std::min(p.len1, p.len2)));
            if (n > 0) {
              ImPlot::PlotErrorBars(label, p.data2, p.data0, p.data1, n);
            }
          }
        }
        break;
      }
      case kPrimInfLines: {
        if (p.data0 != nullptr && p.len0 > 0) {
          if (p.i1 == 0) {
            ImPlot::PlotInfLines(label, p.data0, static_cast<int>(p.len0));
          } else {
            const ImPlotRect limits = ImPlot::GetPlotLimits();
            tmp_x.clear();
            tmp_y.clear();
            tmp_x.reserve(static_cast<std::size_t>(p.len0) * 3U);
            tmp_y.reserve(static_cast<std::size_t>(p.len0) * 3U);
            const float nan = std::numeric_limits<float>::quiet_NaN();
            for (std::uint32_t i = 0; i < p.len0; ++i) {
              const float yy = p.data0[static_cast<std::size_t>(i)];
              if (!std::isfinite(yy)) {
                continue;
              }
              tmp_x.push_back(static_cast<float>(limits.X.Min));
              tmp_y.push_back(yy);
              tmp_x.push_back(static_cast<float>(limits.X.Max));
              tmp_y.push_back(yy);
              if (i + 1U < p.len0) {
                tmp_x.push_back(nan);
                tmp_y.push_back(nan);
              }
            }
            if (!tmp_x.empty()) {
              ImPlot::PlotLine(label, tmp_x.data(), tmp_y.data(),
                               static_cast<int>(tmp_x.size()));
            }
          }
        }
        break;
      }
      case kPrimHistogram: {
        if (p.data0 != nullptr && p.data1 != nullptr && p.len0 > 1 && p.len1 > 0) {
          const int n = static_cast<int>(std::min(p.len1, p.len0 - 1));
          tmp_x.resize(static_cast<std::size_t>(n));
          float mean_w = 1.0f;
          for (int i = 0; i < n; ++i) {
            const float x0 = p.data0[static_cast<std::size_t>(i)];
            const float x1 = p.data0[static_cast<std::size_t>(i + 1)];
            tmp_x[static_cast<std::size_t>(i)] = 0.5f * (x0 + x1);
            mean_w += std::fabs(x1 - x0);
          }
          mean_w /= static_cast<float>(n + 1);
          ImPlot::PlotBars(label, tmp_x.data(), p.data1, n, std::max(1e-6f, mean_w));
        }
        break;
      }
      case kPrimHistogram2D: {
        if (p.data2 != nullptr && p.i1 > 0 && p.i2 > 0) {
          const double x0 = (p.data0 != nullptr && p.len0 > 0) ? p.data0[0] : 0.0;
          const double x1 = (p.data0 != nullptr && p.len0 > 0)
                                ? p.data0[static_cast<std::size_t>(p.len0 - 1)]
                                : static_cast<double>(p.i2);
          const double y0 = (p.data1 != nullptr && p.len1 > 0) ? p.data1[0] : 0.0;
          const double y1 = (p.data1 != nullptr && p.len1 > 0)
                                ? p.data1[static_cast<std::size_t>(p.len1 - 1)]
                                : static_cast<double>(p.i1);
          std::string label_fmt = "%.0f";
          std::string colorbar_label;
          std::string colorbar_fmt = "%g";
          if (p.text != nullptr) {
            const std::vector<std::string> parts = split_labels(p.text, '\x1d');
            if (!parts.empty()) {
              label_fmt = parts[0];
            }
            if (parts.size() > 1U) {
              colorbar_label = parts[1];
            }
            if (parts.size() > 2U && !parts[2].empty()) {
              colorbar_fmt = parts[2];
            }
          }
          const bool has_manual_scale =
              std::isfinite(static_cast<double>(p.f0)) &&
              std::isfinite(static_cast<double>(p.f1));
          const double scale_min = has_manual_scale ? static_cast<double>(p.f0) : 0.0;
          const double scale_max = has_manual_scale ? static_cast<double>(p.f1) : 0.0;
          const char *fmt = label_fmt.empty() ? nullptr : label_fmt.c_str();
          ImPlotSpec hm_spec;
          hm_spec.Flags = std::max(0, p.i3);
          ImPlot::PlotHeatmap(label, p.data2, p.i1, p.i2, scale_min, scale_max, fmt,
                              ImPlotPoint(x0, y0), ImPlotPoint(x1, y1), hm_spec);
          if (p.i0 != 0) {
            double cbar_min = scale_min;
            double cbar_max = scale_max;
            if (!has_manual_scale || !std::isfinite(cbar_min) ||
                !std::isfinite(cbar_max) || cbar_min == cbar_max) {
              if (!compute_value_range(p.data2, p.len2, cbar_min, cbar_max)) {
                cbar_min = 0.0;
                cbar_max = 1.0;
              }
            }
            if (cbar_min == cbar_max) {
              const double pad = std::max(1e-6, std::fabs(cbar_min) * 1e-3);
              cbar_min -= pad;
              cbar_max += pad;
            }
            const std::string scale_label =
                colorbar_label.empty()
                    ? ("##cbar_h2d_" + std::to_string(p.id))
                    : (colorbar_label + "###cbar_h2d_" + std::to_string(p.id));
            const char *scale_fmt =
                colorbar_fmt.empty() ? "%g" : colorbar_fmt.c_str();
            ImPlot::ColormapScale(
                scale_label.c_str(), cbar_min, cbar_max, ImVec2(48.0f, 0.0f),
                scale_fmt,
                static_cast<ImPlotColormapScaleFlags>(std::max(0, p.i6)));
          }
        }
        break;
      }
      case kPrimHeatmap: {
        if (p.data0 != nullptr && p.i1 > 0 && p.i2 > 0) {
          std::string label_fmt = "%.2f";
          std::string colorbar_label;
          std::string colorbar_fmt = "%g";
          if (p.text != nullptr) {
            const std::vector<std::string> parts = split_labels(p.text, '\x1d');
            if (!parts.empty()) {
              label_fmt = parts[0];
            }
            if (parts.size() > 1U) {
              colorbar_label = parts[1];
            }
            if (parts.size() > 2U && !parts[2].empty()) {
              colorbar_fmt = parts[2];
            }
          }
          const bool has_manual_scale =
              std::isfinite(static_cast<double>(p.f0)) &&
              std::isfinite(static_cast<double>(p.f1));
          const double scale_min = has_manual_scale ? static_cast<double>(p.f0) : 0.0;
          const double scale_max = has_manual_scale ? static_cast<double>(p.f1) : 0.0;
          const char *fmt = label_fmt.empty() ? nullptr : label_fmt.c_str();
          ImPlotSpec hm_spec;
          hm_spec.Flags = std::max(0, p.i3);
          ImPlot::PlotHeatmap(label, p.data0, p.i1, p.i2, scale_min, scale_max, fmt,
                              ImPlotPoint(0, 0), ImPlotPoint(p.i2, p.i1), hm_spec);
          if (p.i0 != 0) {
            double cbar_min = scale_min;
            double cbar_max = scale_max;
            if (!has_manual_scale || !std::isfinite(cbar_min) ||
                !std::isfinite(cbar_max) || cbar_min == cbar_max) {
              if (!compute_value_range(p.data0, p.len0, cbar_min, cbar_max)) {
                cbar_min = 0.0;
                cbar_max = 1.0;
              }
            }
            if (cbar_min == cbar_max) {
              const double pad = std::max(1e-6, std::fabs(cbar_min) * 1e-3);
              cbar_min -= pad;
              cbar_max += pad;
            }
            const std::string scale_label =
                colorbar_label.empty()
                    ? ("##cbar_hm_" + std::to_string(p.id))
                    : (colorbar_label + "###cbar_hm_" + std::to_string(p.id));
            const char *scale_fmt =
                colorbar_fmt.empty() ? "%g" : colorbar_fmt.c_str();
            ImPlot::ColormapScale(
                scale_label.c_str(), cbar_min, cbar_max, ImVec2(48.0f, 0.0f),
                scale_fmt,
                static_cast<ImPlotColormapScaleFlags>(std::max(0, p.i6)));
          }
        }
        break;
      }
      case kPrimImage: {
        if (state != nullptr && p.data0 != nullptr && p.i1 > 0 && p.i2 > 0) {
          const GLuint tex = ensure_image_texture(*state, p);
          if (tex != 0U) {
            double x_min =
                std::isfinite(static_cast<double>(p.f0)) ? static_cast<double>(p.f0)
                                                         : 0.0;
            double x_max =
                std::isfinite(static_cast<double>(p.f1)) ? static_cast<double>(p.f1)
                                                         : static_cast<double>(p.i2);
            double y_min =
                std::isfinite(static_cast<double>(p.f2)) ? static_cast<double>(p.f2)
                                                         : 0.0;
            double y_max =
                std::isfinite(static_cast<double>(p.f3)) ? static_cast<double>(p.f3)
                                                         : static_cast<double>(p.i1);
            if (x_min == x_max) {
              x_max = x_min + 1.0;
            }
            if (y_min == y_max) {
              y_max = y_min + 1.0;
            }
            const float uv0_x =
                std::isfinite(static_cast<double>(p.f4)) ? p.f4 : 0.0f;
            const float uv0_y =
                std::isfinite(static_cast<double>(p.f5)) ? p.f5 : 0.0f;
            const float uv1_x =
                std::isfinite(static_cast<double>(p.f6)) ? p.f6 : 1.0f;
            const float uv1_y =
                std::isfinite(static_cast<double>(p.f7)) ? p.f7 : 1.0f;
            ImVec4 tint(1.0f, 1.0f, 1.0f, 1.0f);
            if (p.data1 != nullptr && p.len1 >= 4U) {
              tint = ImVec4(std::clamp(p.data1[0], 0.0f, 1.0f),
                            std::clamp(p.data1[1], 0.0f, 1.0f),
                            std::clamp(p.data1[2], 0.0f, 1.0f),
                            std::clamp(p.data1[3], 0.0f, 1.0f));
            }
            ImPlotSpec img_spec;
            img_spec.Flags = std::max(0, p.i0);
            ImPlot::PlotImage(
                label, (ImTextureID)(intptr_t)tex, ImPlotPoint(x_min, y_min),
                ImPlotPoint(x_max, y_max), ImVec2(uv0_x, uv0_y),
                ImVec2(uv1_x, uv1_y), tint, img_spec);
          }
        }
        break;
      }
      case kPrimPieChart: {
        if (p.data0 == nullptr || p.len0 == 0) {
          break;
        }
        label_storage.clear();
        label_ptrs.clear();
        label_storage.reserve(p.len0);
        label_ptrs.reserve(p.len0);
        std::string fmt = "%.1f";
        std::vector<std::string> parsed_labels;
        if (p.text != nullptr && p.text[0] != '\0') {
          std::string text = p.text;
          const std::size_t sep = text.find('\x1e');
          if (sep == std::string::npos) {
            fmt = text;
          } else {
            fmt = text.substr(0, sep);
            const std::string label_blob = text.substr(sep + 1U);
            parsed_labels = split_labels(label_blob.c_str(), '\x1f');
          }
        }
        for (std::uint32_t i = 0; i < p.len0; ++i) {
          if (i < parsed_labels.size() && !parsed_labels[i].empty()) {
            label_storage.push_back(parsed_labels[i]);
          } else {
            label_storage.push_back(std::to_string(i));
          }
        }
        for (std::uint32_t i = 0; i < p.len0; ++i) {
          label_ptrs.push_back(label_storage[static_cast<std::size_t>(i)].c_str());
        }
        ImPlot::PlotPieChart(label_ptrs.data(), p.data0, static_cast<int>(p.len0), p.f4,
                             p.f5, std::max(1e-6f, p.f6), fmt.c_str(), p.f7);
        break;
      }
      case kPrimText: {
        if (p.text != nullptr && p.text[0] != '\0') {
          ImPlot::PlotText(p.text, p.f4, p.f5, ImVec2(p.f6, p.f7));
        }
        break;
      }
      case kPrimAnnotation: {
        if (p.text != nullptr && p.text[0] != '\0') {
          ImPlot::Annotation(p.f4, p.f5, ImVec4(0.1f, 0.1f, 0.1f, 1.0f),
                             ImVec2(p.f6, p.f7), true, "%s", p.text);
        }
        break;
      }
      case kPrimDummy: {
        ImPlot::PlotDummy(label);
        break;
      }
      case kPrimDragLineX: {
        double x = static_cast<double>(p.f4);
        const bool changed = ImPlot::DragLineX(static_cast<int>(p.id), &x,
                                               color_for_slot(static_cast<int>(p.id)),
                                               std::max(1.0f, p.f6));
        p.f4 = static_cast<float>(x);
        p.i6 = changed ? 1 : 0;
        break;
      }
      case kPrimDragLineY: {
        double y = static_cast<double>(p.f5);
        const bool changed = ImPlot::DragLineY(static_cast<int>(p.id), &y,
                                               color_for_slot(static_cast<int>(p.id)),
                                               std::max(1.0f, p.f6));
        p.f5 = static_cast<float>(y);
        p.i6 = changed ? 1 : 0;
        break;
      }
      case kPrimDragPoint: {
        double x = static_cast<double>(p.f4);
        double y = static_cast<double>(p.f5);
        const bool changed = ImPlot::DragPoint(
            static_cast<int>(p.id), &x, &y,
            color_for_slot(static_cast<int>(p.id)), std::max(2.0f, p.f6));
        p.f4 = static_cast<float>(x);
        p.f5 = static_cast<float>(y);
        p.i6 = changed ? 1 : 0;
        break;
      }
      case kPrimDragRect: {
        double x1 = static_cast<double>(p.f4);
        double y1 = static_cast<double>(p.f5);
        double x2 = static_cast<double>(p.f6);
        double y2 = static_cast<double>(p.f7);
        const bool changed = ImPlot::DragRect(
            static_cast<int>(p.id), &x1, &y1, &x2, &y2,
            color_for_slot(static_cast<int>(p.id)));
        p.f4 = static_cast<float>(x1);
        p.f5 = static_cast<float>(y1);
        p.f6 = static_cast<float>(x2);
        p.f7 = static_cast<float>(y2);
        p.i6 = changed ? 1 : 0;
        break;
      }
      case kPrimTagX: {
        const double x = static_cast<double>(p.f4);
        const ImVec4 col = color_for_slot(static_cast<int>(p.id));
        if (p.text == nullptr || p.text[0] == '\0') {
          ImPlot::TagX(x, col, p.i1 != 0);
        } else {
          ImPlot::TagX(x, col, p.text, x);
        }
        break;
      }
      case kPrimTagY: {
        const double y = static_cast<double>(p.f5);
        const ImVec4 col = color_for_slot(static_cast<int>(p.id));
        if (p.text == nullptr || p.text[0] == '\0') {
          ImPlot::TagY(y, col, p.i1 != 0);
        } else {
          ImPlot::TagY(y, col, p.text, y);
        }
        break;
      }
      case kPrimColormapSlider: {
        std::vector<std::string> parts = split_labels(p.text, '\x1d');
        std::string slider_label =
            (parts.empty() || parts[0].empty()) ? "Colormap" : parts[0];
        slider_label += "###cms_" + std::to_string(p.id);
        const char *fmt =
            (parts.size() > 1U && !parts[1].empty()) ? parts[1].c_str() : "";
        float t = std::clamp(p.f4, 0.0f, 1.0f);
        const bool changed = ImPlot::ColormapSlider(slider_label.c_str(), &t, nullptr, fmt);
        p.f4 = t;
        p.i6 = changed ? 1 : 0;
        break;
      }
      case kPrimColormapButton: {
        std::string button_label = (p.text != nullptr && p.text[0] != '\0')
                                       ? std::string(p.text)
                                       : "Colormap";
        button_label += "###cmb_" + std::to_string(p.id);
        const ImVec2 size(std::max(0.0f, p.f4), std::max(0.0f, p.f5));
        const bool changed = ImPlot::ColormapButton(button_label.c_str(), size);
        p.i6 = changed ? 1 : 0;
        break;
      }
      case kPrimColormapSelector: {
        std::string selector_label = (p.text != nullptr && p.text[0] != '\0')
                                         ? std::string(p.text)
                                         : "Colormap";
        selector_label += "###cmsel_" + std::to_string(p.id);
        const bool changed = ImPlot::ShowColormapSelector(selector_label.c_str());
        p.f4 = static_cast<float>(ImPlot::GetStyle().Colormap);
        p.i6 = changed ? 1 : 0;
        break;
      }
      case kPrimDragDropPlot: {
        struct DragDropPayloadData {
          std::int32_t source_kind;
          std::int32_t source_axis;
          std::int32_t source_subplot;
        };
        constexpr const char *kPayloadPlot = "NBIMPLOT_DND_PLOT";
        constexpr const char *kPayloadAxis = "NBIMPLOT_DND_AXIS";
        constexpr const char *kPayloadLegend = "NBIMPLOT_DND_LEGEND";
        p.i6 = 0;
        p.f4 = 0.0f;
        p.f5 = 0.0f;
        p.f6 = 0.0f;
        p.f7 = 0.0f;
        if (p.i0 != 0 && ImPlot::BeginDragDropSourcePlot()) {
          const DragDropPayloadData payload = {kPrimDragDropPlot, -1, p.i7};
          ImGui::SetDragDropPayload(kPayloadPlot, &payload, sizeof(payload), ImGuiCond_Always);
          ImGui::TextUnformatted("Plot");
          ImPlot::EndDragDropSource();
        }
        if (p.i1 != 0 && ImPlot::BeginDragDropTargetPlot()) {
          const ImGuiPayload *accepted = ImGui::AcceptDragDropPayload(kPayloadPlot);
          if (accepted == nullptr) {
            accepted = ImGui::AcceptDragDropPayload(kPayloadAxis);
          }
          if (accepted == nullptr) {
            accepted = ImGui::AcceptDragDropPayload(kPayloadLegend);
          }
          if (accepted != nullptr) {
            p.i6 = 1;
            if (accepted->DataSize == static_cast<int>(sizeof(DragDropPayloadData))) {
              const auto *payload =
                  static_cast<const DragDropPayloadData *>(accepted->Data);
              p.f6 = static_cast<float>(payload->source_kind);
              p.f7 = static_cast<float>(payload->source_axis);
            }
            const ImPlotPoint mouse = ImPlot::GetPlotMousePos(
                static_cast<ImAxis>(x_axis), static_cast<ImAxis>(y_axis));
            p.f4 = static_cast<float>(mouse.x);
            p.f5 = static_cast<float>(mouse.y);
          }
          ImPlot::EndDragDropTarget();
        }
        break;
      }
      case kPrimDragDropAxis: {
        struct DragDropPayloadData {
          std::int32_t source_kind;
          std::int32_t source_axis;
          std::int32_t source_subplot;
        };
        constexpr const char *kPayloadPlot = "NBIMPLOT_DND_PLOT";
        constexpr const char *kPayloadAxis = "NBIMPLOT_DND_AXIS";
        constexpr const char *kPayloadLegend = "NBIMPLOT_DND_LEGEND";
        const std::int32_t axis_code = std::clamp(p.i2, 0, 5);
        p.i6 = 0;
        p.f4 = 0.0f;
        p.f5 = 0.0f;
        p.f6 = 0.0f;
        p.f7 = 0.0f;
        if (p.i0 != 0 &&
            ImPlot::BeginDragDropSourceAxis(static_cast<ImAxis>(axis_code))) {
          const DragDropPayloadData payload = {kPrimDragDropAxis, axis_code, p.i7};
          ImGui::SetDragDropPayload(kPayloadAxis, &payload, sizeof(payload), ImGuiCond_Always);
          ImGui::Text("Axis %d", axis_code);
          ImPlot::EndDragDropSource();
        }
        if (p.i1 != 0 &&
            ImPlot::BeginDragDropTargetAxis(static_cast<ImAxis>(axis_code))) {
          const ImGuiPayload *accepted = ImGui::AcceptDragDropPayload(kPayloadAxis);
          if (accepted == nullptr) {
            accepted = ImGui::AcceptDragDropPayload(kPayloadPlot);
          }
          if (accepted == nullptr) {
            accepted = ImGui::AcceptDragDropPayload(kPayloadLegend);
          }
          if (accepted != nullptr) {
            p.i6 = 1;
            p.f5 = static_cast<float>(axis_code);
            if (accepted->DataSize == static_cast<int>(sizeof(DragDropPayloadData))) {
              const auto *payload =
                  static_cast<const DragDropPayloadData *>(accepted->Data);
              p.f4 = static_cast<float>(payload->source_axis);
              p.f6 = static_cast<float>(payload->source_kind);
              p.f7 = static_cast<float>(payload->source_subplot);
            }
          }
          ImPlot::EndDragDropTarget();
        }
        break;
      }
      case kPrimDragDropLegend: {
        struct DragDropPayloadData {
          std::int32_t source_kind;
          std::int32_t source_axis;
          std::int32_t source_subplot;
        };
        constexpr const char *kPayloadPlot = "NBIMPLOT_DND_PLOT";
        constexpr const char *kPayloadAxis = "NBIMPLOT_DND_AXIS";
        constexpr const char *kPayloadLegend = "NBIMPLOT_DND_LEGEND";
        p.i6 = 0;
        p.f4 = 0.0f;
        p.f5 = 0.0f;
        p.f6 = 0.0f;
        p.f7 = 0.0f;
        if (p.i1 != 0 && ImPlot::BeginDragDropTargetLegend()) {
          const ImGuiPayload *accepted = ImGui::AcceptDragDropPayload(kPayloadLegend);
          if (accepted == nullptr) {
            accepted = ImGui::AcceptDragDropPayload(kPayloadAxis);
          }
          if (accepted == nullptr) {
            accepted = ImGui::AcceptDragDropPayload(kPayloadPlot);
          }
          if (accepted != nullptr) {
            p.i6 = 1;
            if (accepted->DataSize == static_cast<int>(sizeof(DragDropPayloadData))) {
              const auto *payload =
                  static_cast<const DragDropPayloadData *>(accepted->Data);
              p.f4 = static_cast<float>(payload->source_kind);
              p.f5 = static_cast<float>(payload->source_axis);
              p.f6 = static_cast<float>(payload->source_subplot);
            }
          }
          ImPlot::EndDragDropTarget();
        }
        if (p.i6 == 0 && p.i0 != 0 && ImPlot::BeginDragDropSourcePlot()) {
          const DragDropPayloadData payload = {kPrimDragDropLegend, -1, p.i7};
          ImGui::SetDragDropPayload(kPayloadLegend, &payload, sizeof(payload), ImGuiCond_Always);
          ImGui::TextUnformatted("Legend");
          ImPlot::EndDragDropSource();
        }
        break;
      }
      default:
        break;
      }
    }
  };

  bool ok = false;
  std::vector<float> tmp_x;
  std::vector<float> tmp_y;
  std::vector<const char *> label_ptrs;
  std::vector<std::string> label_storage;
  std::array<double, 6> axis_link_min = {0.0, 0.0, 0.0, -1.0, -1.0, -1.0};
  std::array<double, 6> axis_link_max = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  if (axis_view_min6 != nullptr && axis_view_max6 != nullptr) {
    for (std::int32_t axis = 0; axis < 6; ++axis) {
      axis_link_min[static_cast<std::size_t>(axis)] = axis_view_min6[axis];
      axis_link_max[static_cast<std::size_t>(axis)] = axis_view_max6[axis];
    }
  } else {
    if (x_min != nullptr && x_max != nullptr) {
      axis_link_min[0] = *x_min;
      axis_link_max[0] = *x_max;
      axis_link_min[1] = *x_min;
      axis_link_max[1] = *x_max;
      axis_link_min[2] = *x_min;
      axis_link_max[2] = *x_max;
    }
    if (y_min != nullptr && y_max != nullptr) {
      axis_link_min[3] = *y_min;
      axis_link_max[3] = *y_max;
      axis_link_min[4] = *y_min;
      axis_link_max[4] = *y_max;
      axis_link_min[5] = *y_min;
      axis_link_max[5] = *y_max;
    }
  }

  auto setup_axes = [&]() {
    for (std::int32_t axis = 0; axis < 6; ++axis) {
      bool enabled = axis == 0 || axis == 3;
      if (axis_enabled6 != nullptr && axis != 0 && axis != 3) {
        enabled = axis_enabled6[axis] != 0;
      }
      if (!enabled) {
        continue;
      }

      ImPlotAxisFlags axis_flags = ImPlotAxisFlags_None;
      if (axis == 1 || axis == 2 || axis == 4 || axis == 5) {
        axis_flags = ImPlotAxisFlags_AuxDefault;
      }
      const char *axis_label = nullptr;
      if (axis_labels6 != nullptr) {
        axis_label = axis_labels6[axis];
        if (axis_label != nullptr && axis_label[0] == '\0') {
          axis_label = nullptr;
        }
      }
      ImPlot::SetupAxis(static_cast<ImAxis>(axis), axis_label, axis_flags);

      std::int32_t scale_code = 0;
      if (axis_scale6 != nullptr) {
        scale_code = std::clamp(axis_scale6[axis], 0, 2);
      }
      if (scale_code == 1) {
        ImPlot::SetupAxisScale(static_cast<ImAxis>(axis), ImPlotScale_Log10);
      } else if (scale_code == 2) {
        ImPlot::SetupAxisScale(static_cast<ImAxis>(axis), ImPlotScale_Time);
      }

      if (axis_formats6 != nullptr && axis_formats6[axis] != nullptr &&
          axis_formats6[axis][0] != '\0') {
        ImPlot::SetupAxisFormat(static_cast<ImAxis>(axis), axis_formats6[axis]);
      }

      if (axis_ticks_values6 != nullptr && axis_ticks_counts6 != nullptr) {
        const double *ticks = axis_ticks_values6[axis];
        const std::int32_t tick_count = std::max(0, axis_ticks_counts6[axis]);
        if (ticks != nullptr && tick_count > 0) {
          std::vector<const char *> tick_label_ptrs;
          const char *const labels_blob =
              (axis_ticks_labels6 != nullptr) ? axis_ticks_labels6[axis] : nullptr;
          std::vector<std::string> parsed_labels = split_labels(labels_blob, '\x1f');
          const char *const *label_ptr = nullptr;
          if (!parsed_labels.empty()) {
            tick_label_ptrs.reserve(static_cast<std::size_t>(tick_count));
            for (std::int32_t i = 0; i < tick_count; ++i) {
              if (static_cast<std::size_t>(i) < parsed_labels.size()) {
                tick_label_ptrs.push_back(parsed_labels[static_cast<std::size_t>(i)].c_str());
              } else {
                tick_label_ptrs.push_back(nullptr);
              }
            }
            label_ptr = tick_label_ptrs.data();
          }
          const bool keep_default =
              axis_ticks_keep_default6 != nullptr &&
              axis_ticks_keep_default6[axis] != 0;
          ImPlot::SetupAxisTicks(static_cast<ImAxis>(axis), ticks, tick_count,
                                 label_ptr, keep_default);
        }
      }

      if (axis_limits_constraints_enabled6 != nullptr &&
          axis_limits_constraints_min6 != nullptr &&
          axis_limits_constraints_max6 != nullptr &&
          axis_limits_constraints_enabled6[axis] != 0) {
        ImPlot::SetupAxisLimitsConstraints(
            static_cast<ImAxis>(axis), axis_limits_constraints_min6[axis],
            axis_limits_constraints_max6[axis]);
      }

      if (axis_zoom_constraints_enabled6 != nullptr &&
          axis_zoom_constraints_min6 != nullptr &&
          axis_zoom_constraints_max6 != nullptr &&
          axis_zoom_constraints_enabled6[axis] != 0) {
        ImPlot::SetupAxisZoomConstraints(static_cast<ImAxis>(axis),
                                         axis_zoom_constraints_min6[axis],
                                         axis_zoom_constraints_max6[axis]);
      }

      if (axis_links6 != nullptr) {
        const std::int32_t target = axis_links6[axis];
        if (target >= 0 && target < 6 && target != axis &&
            ((axis <= 2) == (target <= 2))) {
          ImPlot::SetupAxisLinks(static_cast<ImAxis>(axis),
                                 &axis_link_min[static_cast<std::size_t>(target)],
                                 &axis_link_max[static_cast<std::size_t>(target)]);
        }
      }
    }
    if (force_view && x_min != nullptr && x_max != nullptr && y_min != nullptr &&
        y_max != nullptr) {
      ImPlot::SetupAxisLimits(ImAxis_X1, *x_min, *x_max, ImGuiCond_Always);
      ImPlot::SetupAxisLimits(ImAxis_Y1, *y_min, *y_max, ImGuiCond_Always);
    }
  };

  auto render_hover_tooltip = [&](std::int32_t subplot_index) {
    if (!ImPlot::IsPlotHovered()) {
      return;
    }

    enum class HighlightType { Point, Rect, VLine, HLine };
    struct Highlight {
      HighlightType type = HighlightType::Point;
      std::int32_t x_axis = 0;
      std::int32_t y_axis = 3;
      double x0 = 0.0;
      double y0 = 0.0;
      double x1 = 0.0;
      double y1 = 0.0;
      ImU32 color = IM_COL32(29, 78, 216, 255);
      float radius = 4.5f;
    };

    std::array<ImPlotPoint, 9> mouse_cache = {};
    std::array<bool, 9> mouse_cache_valid = {false, false, false,
                                             false, false, false,
                                             false, false, false};
    auto mouse_for_axes = [&](std::int32_t x_axis, std::int32_t y_axis) -> ImPlotPoint {
      const std::int32_t xa = std::clamp(x_axis, 0, 2);
      const std::int32_t ya = std::clamp(y_axis, 3, 5);
      const std::size_t cache_idx =
          static_cast<std::size_t>(xa * 3 + (ya - 3));
      if (!mouse_cache_valid[cache_idx]) {
        mouse_cache[cache_idx] = ImPlot::GetPlotMousePos(
            static_cast<ImAxis>(xa), static_cast<ImAxis>(ya));
        mouse_cache_valid[cache_idx] = true;
      }
      return mouse_cache[cache_idx];
    };

    const ImVec2 mouse_px = ImGui::GetIO().MousePos;
    Highlight best_highlight;
    bool has_highlight = false;
    double best_highlight_d2 = std::numeric_limits<double>::infinity();

    auto consider_point_highlight =
        [&](double x, double y, std::int32_t x_axis, std::int32_t y_axis,
            ImU32 color, float radius = 4.5f) {
          const ImVec2 px = ImPlot::PlotToPixels(
              x, y, static_cast<ImAxis>(x_axis), static_cast<ImAxis>(y_axis));
          if (!is_finite_vec2(px)) {
            return;
          }
          const double d2 = distance2(px, mouse_px);
          if (d2 < best_highlight_d2) {
            best_highlight_d2 = d2;
            has_highlight = true;
            best_highlight.type = HighlightType::Point;
            best_highlight.x_axis = x_axis;
            best_highlight.y_axis = y_axis;
            best_highlight.x0 = x;
            best_highlight.y0 = y;
            best_highlight.color = color;
            best_highlight.radius = radius;
          }
        };

    auto consider_rect_highlight =
        [&](double x0, double y0, double x1, double y1, std::int32_t x_axis,
            std::int32_t y_axis, ImU32 color) {
          const ImVec2 p0 = ImPlot::PlotToPixels(
              x0, y0, static_cast<ImAxis>(x_axis), static_cast<ImAxis>(y_axis));
          const ImVec2 p1 = ImPlot::PlotToPixels(
              x1, y1, static_cast<ImAxis>(x_axis), static_cast<ImAxis>(y_axis));
          if (!is_finite_vec2(p0) || !is_finite_vec2(p1)) {
            return;
          }
          const ImVec2 bmin(std::min(p0.x, p1.x), std::min(p0.y, p1.y));
          const ImVec2 bmax(std::max(p0.x, p1.x), std::max(p0.y, p1.y));
          const float cx = 0.5f * (bmin.x + bmax.x);
          const float cy = 0.5f * (bmin.y + bmax.y);
          const double d2 = distance2(ImVec2(cx, cy), mouse_px);
          if (d2 < best_highlight_d2) {
            best_highlight_d2 = d2;
            has_highlight = true;
            best_highlight.type = HighlightType::Rect;
            best_highlight.x_axis = x_axis;
            best_highlight.y_axis = y_axis;
            best_highlight.x0 = x0;
            best_highlight.y0 = y0;
            best_highlight.x1 = x1;
            best_highlight.y1 = y1;
            best_highlight.color = color;
          }
        };

    auto consider_line_highlight =
        [&](bool vertical, double value, std::int32_t x_axis, std::int32_t y_axis,
            ImU32 color) {
          const ImPlotRect lim =
              ImPlot::GetPlotLimits(static_cast<ImAxis>(x_axis),
                                    static_cast<ImAxis>(y_axis));
          ImVec2 p0;
          ImVec2 p1;
          if (vertical) {
            p0 = ImPlot::PlotToPixels(value, lim.Y.Min, static_cast<ImAxis>(x_axis),
                                      static_cast<ImAxis>(y_axis));
            p1 = ImPlot::PlotToPixels(value, lim.Y.Max, static_cast<ImAxis>(x_axis),
                                      static_cast<ImAxis>(y_axis));
          } else {
            p0 = ImPlot::PlotToPixels(lim.X.Min, value, static_cast<ImAxis>(x_axis),
                                      static_cast<ImAxis>(y_axis));
            p1 = ImPlot::PlotToPixels(lim.X.Max, value, static_cast<ImAxis>(x_axis),
                                      static_cast<ImAxis>(y_axis));
          }
          if (!is_finite_vec2(p0) || !is_finite_vec2(p1)) {
            return;
          }
          const double d2 = point_segment_distance2(mouse_px, p0, p1);
          if (d2 < best_highlight_d2) {
            best_highlight_d2 = d2;
            has_highlight = true;
            best_highlight.type =
                vertical ? HighlightType::VLine : HighlightType::HLine;
            best_highlight.x_axis = x_axis;
            best_highlight.y_axis = y_axis;
            best_highlight.x0 = value;
            best_highlight.color = color;
          }
        };

    std::vector<std::string> lines;
    lines.reserve(24);
    const ImPlotPoint mouse_main = mouse_for_axes(0, 3);
    lines.push_back(format_text("x=%.6g y=%.6g", mouse_main.x, mouse_main.y));

    if (series_views != nullptr && draw_points != nullptr && segments != nullptr) {
      for (std::uint32_t svi = 0; svi < series_count; ++svi) {
        const SeriesView &series = series_views[svi];
        if (series.subplot_index != subplot_index) {
          continue;
        }
        const auto range_it = slot_segment_ranges.find(series.slot);
        if (range_it == slot_segment_ranges.end()) {
          continue;
        }
        const std::int32_t x_axis = std::clamp(series.x_axis, 0, 2);
        const std::int32_t y_axis = std::clamp(series.y_axis, 3, 5);
        const ImPlotPoint mouse = mouse_for_axes(x_axis, y_axis);
        if (!std::isfinite(mouse.x) || !std::isfinite(mouse.y)) {
          continue;
        }

        double best_dist = std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        double best_y = 0.0;
        bool found = false;
        const auto [seg_begin, seg_end] = range_it->second;
        for (std::uint32_t si = seg_begin; si < seg_end; ++si) {
          const DrawSegmentView &seg = segments[si];
          if (seg.slot != series.slot) {
            continue;
          }
          if (seg.count < 1 || seg.start >= point_count ||
              seg.count > point_count - seg.start) {
            continue;
          }
          const float *base =
              draw_points + static_cast<std::size_t>(seg.start) * 4U;
          for (std::uint32_t i = 0; i < seg.count; ++i) {
            const float *pt = base + static_cast<std::size_t>(i) * 4U;
            const double x = static_cast<double>(pt[1]);
            const double y = static_cast<double>(pt[2]);
            if (!std::isfinite(x) || !std::isfinite(y)) {
              continue;
            }
            const double dist = std::fabs(x - mouse.x);
            if (dist < best_dist) {
              best_dist = dist;
              best_x = x;
              best_y = y;
              found = true;
            }
          }
        }
        if (!found) {
          continue;
        }
        const std::string label =
            (series.label != nullptr && series.label[0] != '\0')
                ? std::string(series.label)
                : ("series_" + std::to_string(series.slot));
        lines.push_back(
            format_text("line %s: x=%.6g y=%.6g", label.c_str(), best_x, best_y));
        consider_point_highlight(
            best_x, best_y, x_axis, y_axis,
            ImGui::ColorConvertFloat4ToU32(color_for_slot(series.slot)), 4.5f);
      }
    }

    for (std::uint32_t pi = 0; pi < primitive_count; ++pi) {
      const PrimitiveView &p = primitives[pi];
      if (p.i7 != subplot_index) {
        continue;
      }
      const bool has_x = p.i0 != 0;
      const std::int32_t x_axis = std::clamp(p.i4, 0, 2);
      const std::int32_t y_axis = std::clamp(p.i5, 3, 5);
      const ImPlotPoint mouse = mouse_for_axes(x_axis, y_axis);
      if (!std::isfinite(mouse.x) || !std::isfinite(mouse.y)) {
        continue;
      }
      const ImU32 prim_color =
          ImGui::ColorConvertFloat4ToU32(color_for_slot(static_cast<std::int32_t>(p.id)));

      std::string prefix = primitive_kind_name(p.kind);
      if (p.kind != kPrimPieChart && p.kind != kPrimBarGroups &&
          p.text != nullptr && p.text[0] != '\0') {
        prefix += " ";
        prefix += p.text;
      } else {
        prefix += "#";
        prefix += std::to_string(p.id);
      }

      switch (p.kind) {
      case kPrimScatter:
      case kPrimStairs:
      case kPrimStems:
      case kPrimDigital:
      case kPrimBars: {
        const std::uint32_t n = has_x ? std::min(p.len0, p.len1) : p.len0;
        if (n == 0U) {
          break;
        }
        double best_dist = std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        double best_y = 0.0;
        bool found = false;
        for (std::uint32_t i = 0; i < n; ++i) {
          const double x = has_x ? static_cast<double>(p.data0[i])
                                 : static_cast<double>(i);
          const double y = has_x ? static_cast<double>(p.data1[i])
                                 : static_cast<double>(p.data0[i]);
          if (!std::isfinite(x) || !std::isfinite(y)) {
            continue;
          }
          const double dist = std::fabs(x - mouse.x);
          if (dist < best_dist) {
            best_dist = dist;
            best_x = x;
            best_y = y;
            found = true;
          }
        }
        if (found) {
          lines.push_back(
              format_text("%s: x=%.6g y=%.6g", prefix.c_str(), best_x, best_y));
          consider_point_highlight(best_x, best_y, x_axis, y_axis, prim_color, 4.5f);
        }
        break;
      }
      case kPrimBubbles: {
        const std::uint32_t n =
            has_x ? std::min(p.len0, std::min(p.len1, p.len2))
                  : std::min(p.len0, p.len1);
        if (n == 0U) {
          break;
        }
        double best_dist = std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        double best_y = 0.0;
        double best_s = 0.0;
        bool found = false;
        for (std::uint32_t i = 0; i < n; ++i) {
          const double x = has_x ? static_cast<double>(p.data0[i])
                                 : static_cast<double>(i);
          const double y = has_x ? static_cast<double>(p.data1[i])
                                 : static_cast<double>(p.data0[i]);
          const double s = has_x ? static_cast<double>(p.data2[i])
                                 : static_cast<double>(p.data1[i]);
          if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(s)) {
            continue;
          }
          const double dist = std::fabs(x - mouse.x);
          if (dist < best_dist) {
            best_dist = dist;
            best_x = x;
            best_y = y;
            best_s = s;
            found = true;
          }
        }
        if (found) {
          lines.push_back(format_text("%s: x=%.6g y=%.6g size=%.6g",
                                      prefix.c_str(), best_x, best_y, best_s));
          consider_point_highlight(best_x, best_y, x_axis, y_axis, prim_color, 5.5f);
        }
        break;
      }
      case kPrimBarsH: {
        const std::uint32_t n = std::min(p.len0, p.len1);
        if (n == 0U) {
          break;
        }
        double best_dist = std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        double best_y = 0.0;
        bool found = false;
        for (std::uint32_t i = 0; i < n; ++i) {
          const double x = static_cast<double>(p.data0[i]);
          const double y = static_cast<double>(p.data1[i]);
          if (!std::isfinite(x) || !std::isfinite(y)) {
            continue;
          }
          const double dist = std::fabs(y - mouse.y);
          if (dist < best_dist) {
            best_dist = dist;
            best_x = x;
            best_y = y;
            found = true;
          }
        }
        if (found) {
          lines.push_back(
              format_text("%s: x=%.6g y=%.6g", prefix.c_str(), best_x, best_y));
          consider_point_highlight(best_x, best_y, x_axis, y_axis, prim_color, 4.5f);
        }
        break;
      }
      case kPrimShaded: {
        const std::uint32_t n =
            has_x ? std::min(p.len0, std::min(p.len1, p.len2))
                  : std::min(p.len0, p.len1);
        if (n == 0U) {
          break;
        }
        double best_dist = std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        double best_y1 = 0.0;
        double best_y2 = 0.0;
        bool found = false;
        for (std::uint32_t i = 0; i < n; ++i) {
          const double x = has_x ? static_cast<double>(p.data0[i])
                                 : static_cast<double>(i);
          const double y1 = has_x ? static_cast<double>(p.data1[i])
                                  : static_cast<double>(p.data0[i]);
          const double y2 = has_x ? static_cast<double>(p.data2[i])
                                  : static_cast<double>(p.data1[i]);
          if (!std::isfinite(x) || !std::isfinite(y1) || !std::isfinite(y2)) {
            continue;
          }
          const double dist = std::fabs(x - mouse.x);
          if (dist < best_dist) {
            best_dist = dist;
            best_x = x;
            best_y1 = y1;
            best_y2 = y2;
            found = true;
          }
        }
        if (found) {
          lines.push_back(format_text("%s: x=%.6g y1=%.6g y2=%.6g",
                                      prefix.c_str(), best_x, best_y1, best_y2));
          consider_point_highlight(best_x, 0.5 * (best_y1 + best_y2), x_axis, y_axis,
                                   prim_color, 4.5f);
        }
        break;
      }
      case kPrimErrorBars: {
        const std::uint32_t n =
            has_x ? std::min(p.len0, std::min(p.len1, p.len2))
                  : std::min(p.len0, p.len1);
        if (n == 0U) {
          break;
        }
        double best_dist = std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        double best_y = 0.0;
        double best_e = 0.0;
        bool found = false;
        for (std::uint32_t i = 0; i < n; ++i) {
          const double x = has_x ? static_cast<double>(p.data0[i])
                                 : static_cast<double>(i);
          const double y = has_x ? static_cast<double>(p.data1[i])
                                 : static_cast<double>(p.data0[i]);
          const double e = has_x ? static_cast<double>(p.data2[i])
                                 : static_cast<double>(p.data1[i]);
          if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(e)) {
            continue;
          }
          const double dist = std::fabs(x - mouse.x);
          if (dist < best_dist) {
            best_dist = dist;
            best_x = x;
            best_y = y;
            best_e = std::fabs(e);
            found = true;
          }
        }
        if (found) {
          lines.push_back(format_text("%s: x=%.6g y=%.6g ± %.6g",
                                      prefix.c_str(), best_x, best_y, best_e));
          consider_point_highlight(best_x, best_y, x_axis, y_axis, prim_color, 4.5f);
        }
        break;
      }
      case kPrimErrorBarsH: {
        const std::uint32_t n = std::min(p.len0, std::min(p.len1, p.len2));
        if (n == 0U) {
          break;
        }
        double best_dist = std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        double best_y = 0.0;
        double best_e = 0.0;
        bool found = false;
        for (std::uint32_t i = 0; i < n; ++i) {
          const double x = static_cast<double>(p.data0[i]);
          const double e = std::fabs(static_cast<double>(p.data1[i]));
          const double y = static_cast<double>(p.data2[i]);
          if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(e)) {
            continue;
          }
          const double dist = std::fabs(y - mouse.y);
          if (dist < best_dist) {
            best_dist = dist;
            best_x = x;
            best_y = y;
            best_e = e;
            found = true;
          }
        }
        if (found) {
          lines.push_back(format_text("%s: y=%.6g x=%.6g ± %.6g",
                                      prefix.c_str(), best_y, best_x, best_e));
          consider_point_highlight(best_x, best_y, x_axis, y_axis, prim_color, 4.5f);
        }
        break;
      }
      case kPrimInfLines: {
        if (p.data0 == nullptr || p.len0 == 0U) {
          break;
        }
        const bool horizontal = p.i1 == 1;
        double best_dist = std::numeric_limits<double>::infinity();
        double best_v = 0.0;
        bool found = false;
        for (std::uint32_t i = 0; i < p.len0; ++i) {
          const double v = static_cast<double>(p.data0[i]);
          if (!std::isfinite(v)) {
            continue;
          }
          const double dist = horizontal ? std::fabs(v - mouse.y)
                                         : std::fabs(v - mouse.x);
          if (dist < best_dist) {
            best_dist = dist;
            best_v = v;
            found = true;
          }
        }
        if (found) {
          lines.push_back(format_text("%s: %c=%.6g", prefix.c_str(),
                                      horizontal ? 'y' : 'x', best_v));
          consider_line_highlight(!horizontal, best_v, x_axis, y_axis, prim_color);
        }
        break;
      }
      case kPrimHistogram: {
        if (p.data0 == nullptr || p.data1 == nullptr || p.len0 <= 1U ||
            p.len1 == 0U) {
          break;
        }
        int bin = find_edge_bin(p.data0, p.len0, mouse.x);
        if (bin < 0) {
          double best_dist = std::numeric_limits<double>::infinity();
          for (std::uint32_t i = 0; i + 1U < p.len0 && i < p.len1; ++i) {
            const double center =
                0.5 * (static_cast<double>(p.data0[i]) +
                       static_cast<double>(p.data0[i + 1U]));
            const double dist = std::fabs(center - mouse.x);
            if (dist < best_dist) {
              best_dist = dist;
              bin = static_cast<int>(i);
            }
          }
        }
        if (bin >= 0 && static_cast<std::uint32_t>(bin + 1) < p.len0 &&
            static_cast<std::uint32_t>(bin) < p.len1) {
          const double x0 = static_cast<double>(p.data0[bin]);
          const double x1 = static_cast<double>(p.data0[bin + 1]);
          const double count = static_cast<double>(p.data1[bin]);
          lines.push_back(format_text("%s: [%.6g, %.6g] count=%.6g",
                                      prefix.c_str(), x0, x1, count));
          consider_rect_highlight(x0, std::min(0.0, count), x1, std::max(0.0, count),
                                  x_axis, y_axis, prim_color);
        }
        break;
      }
      case kPrimHistogram2D: {
        if (p.data2 == nullptr || p.i1 <= 0 || p.i2 <= 0) {
          break;
        }
        const int rows = p.i1;
        const int cols = p.i2;
        const double x0 = (p.data0 != nullptr && p.len0 > 0U)
                              ? static_cast<double>(p.data0[0])
                              : 0.0;
        const double x1 = (p.data0 != nullptr && p.len0 > 0U)
                              ? static_cast<double>(p.data0[p.len0 - 1U])
                              : static_cast<double>(cols);
        const double y0 = (p.data1 != nullptr && p.len1 > 0U)
                              ? static_cast<double>(p.data1[0])
                              : 0.0;
        const double y1 = (p.data1 != nullptr && p.len1 > 0U)
                              ? static_cast<double>(p.data1[p.len1 - 1U])
                              : static_cast<double>(rows);
        if (mouse.x < std::min(x0, x1) || mouse.x > std::max(x0, x1) ||
            mouse.y < std::min(y0, y1) || mouse.y > std::max(y0, y1)) {
          break;
        }
        const double ux = (mouse.x - x0) / std::max(1e-12, std::fabs(x1 - x0));
        const double uy = (mouse.y - y0) / std::max(1e-12, std::fabs(y1 - y0));
        const int col = std::clamp(static_cast<int>(std::floor(ux * cols)), 0, cols - 1);
        const int row = std::clamp(static_cast<int>(std::floor(uy * rows)), 0, rows - 1);
        const std::size_t idx = static_cast<std::size_t>(row * cols + col);
        const std::size_t cap = static_cast<std::size_t>(rows * cols);
        if (idx < cap) {
          const double z = static_cast<double>(p.data2[idx]);
          const double cx0 = x0 + (x1 - x0) * (static_cast<double>(col) / cols);
          const double cx1 = x0 + (x1 - x0) * (static_cast<double>(col + 1) / cols);
          const double cy0 = y0 + (y1 - y0) * (static_cast<double>(row) / rows);
          const double cy1 = y0 + (y1 - y0) * (static_cast<double>(row + 1) / rows);
          lines.push_back(format_text("%s: row=%d col=%d value=%.6g",
                                      prefix.c_str(), row, col, z));
          consider_rect_highlight(cx0, cy0, cx1, cy1, x_axis, y_axis, prim_color);
        }
        break;
      }
      case kPrimHeatmap: {
        if (p.data0 == nullptr || p.i1 <= 0 || p.i2 <= 0) {
          break;
        }
        const int rows = p.i1;
        const int cols = p.i2;
        if (mouse.x < 0.0 || mouse.x > static_cast<double>(cols) ||
            mouse.y < 0.0 || mouse.y > static_cast<double>(rows)) {
          break;
        }
        const int col = std::clamp(static_cast<int>(std::floor(mouse.x)), 0, cols - 1);
        const int row = std::clamp(static_cast<int>(std::floor(mouse.y)), 0, rows - 1);
        const std::size_t idx = static_cast<std::size_t>(row * cols + col);
        const std::size_t cap = static_cast<std::size_t>(rows * cols);
        if (idx < cap) {
          const double z = static_cast<double>(p.data0[idx]);
          lines.push_back(format_text("%s: row=%d col=%d value=%.6g",
                                      prefix.c_str(), row, col, z));
          consider_rect_highlight(static_cast<double>(col), static_cast<double>(row),
                                  static_cast<double>(col + 1),
                                  static_cast<double>(row + 1), x_axis, y_axis,
                                  prim_color);
        }
        break;
      }
      case kPrimImage: {
        if (p.data0 == nullptr || p.i1 <= 0 || p.i2 <= 0) {
          break;
        }
        const int rows = p.i1;
        const int cols = p.i2;
        int channels = p.i6;
        if (channels != 1 && channels != 3 && channels != 4) {
          channels = 1;
        }
        const double x0 = std::isfinite(static_cast<double>(p.f0))
                              ? static_cast<double>(p.f0)
                              : 0.0;
        const double x1 = std::isfinite(static_cast<double>(p.f1))
                              ? static_cast<double>(p.f1)
                              : static_cast<double>(cols);
        const double y0 = std::isfinite(static_cast<double>(p.f2))
                              ? static_cast<double>(p.f2)
                              : 0.0;
        const double y1 = std::isfinite(static_cast<double>(p.f3))
                              ? static_cast<double>(p.f3)
                              : static_cast<double>(rows);
        if (mouse.x < std::min(x0, x1) || mouse.x > std::max(x0, x1) ||
            mouse.y < std::min(y0, y1) || mouse.y > std::max(y0, y1)) {
          break;
        }
        const double ux = (mouse.x - x0) / std::max(1e-12, std::fabs(x1 - x0));
        const double uy = (mouse.y - y0) / std::max(1e-12, std::fabs(y1 - y0));
        const int col =
            std::clamp(static_cast<int>(std::floor(ux * cols)), 0, cols - 1);
        const int row =
            std::clamp(static_cast<int>(std::floor(uy * rows)), 0, rows - 1);
        const std::size_t pixel_idx = static_cast<std::size_t>(row * cols + col);
        const std::size_t base_idx = pixel_idx * static_cast<std::size_t>(channels);
        if (base_idx + static_cast<std::size_t>(channels - 1) >= p.len0) {
          break;
        }
        if (channels == 1) {
          const double v = static_cast<double>(p.data0[base_idx]);
          lines.push_back(format_text("%s: row=%d col=%d value=%.6g",
                                      prefix.c_str(), row, col, v));
        } else if (channels == 3) {
          lines.push_back(format_text(
              "%s: row=%d col=%d rgb=(%.4g, %.4g, %.4g)", prefix.c_str(), row,
              col, static_cast<double>(p.data0[base_idx + 0U]),
              static_cast<double>(p.data0[base_idx + 1U]),
              static_cast<double>(p.data0[base_idx + 2U])));
        } else {
          lines.push_back(format_text(
              "%s: row=%d col=%d rgba=(%.4g, %.4g, %.4g, %.4g)",
              prefix.c_str(), row, col,
              static_cast<double>(p.data0[base_idx + 0U]),
              static_cast<double>(p.data0[base_idx + 1U]),
              static_cast<double>(p.data0[base_idx + 2U]),
              static_cast<double>(p.data0[base_idx + 3U])));
        }
        const double cx0 = x0 + (x1 - x0) * (static_cast<double>(col) / cols);
        const double cx1 = x0 + (x1 - x0) * (static_cast<double>(col + 1) / cols);
        const double cy0 = y0 + (y1 - y0) * (static_cast<double>(row) / rows);
        const double cy1 = y0 + (y1 - y0) * (static_cast<double>(row + 1) / rows);
        consider_rect_highlight(cx0, cy0, cx1, cy1, x_axis, y_axis, prim_color);
        break;
      }
      case kPrimPieChart: {
        if (p.data0 == nullptr || p.len0 == 0U || !std::isfinite(p.f6)) {
          break;
        }
        const double cx = static_cast<double>(p.f4);
        const double cy = static_cast<double>(p.f5);
        const double radius = std::max(1e-12, std::fabs(static_cast<double>(p.f6)));
        const double dx = mouse.x - cx;
        const double dy = mouse.y - cy;
        const double rr = std::sqrt(dx * dx + dy * dy);
        if (rr > radius) {
          break;
        }

        double sum = 0.0;
        for (std::uint32_t i = 0; i < p.len0; ++i) {
          const double v = static_cast<double>(p.data0[i]);
          if (std::isfinite(v) && v > 0.0) {
            sum += v;
          }
        }
        if (sum <= 0.0) {
          break;
        }

        std::vector<std::string> labels;
        if (p.text != nullptr && p.text[0] != '\0') {
          std::string text = p.text;
          const std::size_t sep = text.find('\x1e');
          if (sep != std::string::npos) {
            labels = split_labels(text.substr(sep + 1U).c_str(), '\x1f');
          }
        }

        constexpr double two_pi = 2.0 * 3.14159265358979323846;
        const double a0 = static_cast<double>(p.f7) * two_pi / 360.0;
        double angle = std::atan2(dy, dx);
        if (angle < a0) {
          angle += two_pi * std::ceil((a0 - angle) / two_pi);
        }
        const double rel = std::fmod(angle - a0, two_pi);

        double acc = 0.0;
        int slice = -1;
        double slice_span = 0.0;
        double slice_start = 0.0;
        for (std::uint32_t i = 0; i < p.len0; ++i) {
          const double v = std::max(0.0, static_cast<double>(p.data0[i]));
          const double span = two_pi * (v / sum);
          if (rel <= acc + span || i + 1U == p.len0) {
            slice = static_cast<int>(i);
            slice_start = acc;
            slice_span = span;
            break;
          }
          acc += span;
        }
        if (slice >= 0) {
          const double v = std::max(0.0, static_cast<double>(p.data0[slice]));
          const double pct = 100.0 * v / sum;
          const std::string label =
              (static_cast<std::size_t>(slice) < labels.size() &&
               !labels[static_cast<std::size_t>(slice)].empty())
                  ? labels[static_cast<std::size_t>(slice)]
                  : std::to_string(slice);
          lines.push_back(format_text("%s: %s=%.6g (%.2f%%)",
                                      prefix.c_str(), label.c_str(), v, pct));
          const double mid = a0 + slice_start + 0.5 * slice_span;
          consider_point_highlight(cx + std::cos(mid) * radius * 0.65,
                                   cy + std::sin(mid) * radius * 0.65, x_axis,
                                   y_axis, prim_color, 6.0f);
        }
        break;
      }
      case kPrimText:
      case kPrimAnnotation: {
        if (p.text == nullptr || p.text[0] == '\0') {
          break;
        }
        const ImVec2 point_px =
            ImPlot::PlotToPixels(static_cast<double>(p.f4),
                                 static_cast<double>(p.f5),
                                 static_cast<ImAxis>(x_axis),
                                 static_cast<ImAxis>(y_axis));
        if (!is_finite_vec2(point_px)) {
          break;
        }
        if (distance2(point_px, mouse_px) <= 100.0) {
          lines.push_back(format_text("%s: %s (x=%.6g y=%.6g)",
                                      primitive_kind_name(p.kind), p.text,
                                      static_cast<double>(p.f4),
                                      static_cast<double>(p.f5)));
          consider_point_highlight(static_cast<double>(p.f4), static_cast<double>(p.f5),
                                   x_axis, y_axis, prim_color, 4.0f);
        }
        break;
      }
      case kPrimDummy: {
        if (p.text != nullptr && p.text[0] != '\0' &&
            ImPlot::IsLegendEntryHovered(p.text)) {
          lines.push_back(format_text("dummy: %s", p.text));
        }
        break;
      }
      case kPrimDragLineX: {
        lines.push_back(
            format_text("%s: x=%.6g", prefix.c_str(), static_cast<double>(p.f4)));
        consider_line_highlight(true, static_cast<double>(p.f4), x_axis, y_axis,
                                prim_color);
        break;
      }
      case kPrimDragLineY: {
        lines.push_back(
            format_text("%s: y=%.6g", prefix.c_str(), static_cast<double>(p.f5)));
        consider_line_highlight(false, static_cast<double>(p.f5), x_axis, y_axis,
                                prim_color);
        break;
      }
      case kPrimDragPoint: {
        lines.push_back(format_text("%s: x=%.6g y=%.6g", prefix.c_str(),
                                    static_cast<double>(p.f4),
                                    static_cast<double>(p.f5)));
        consider_point_highlight(static_cast<double>(p.f4), static_cast<double>(p.f5),
                                 x_axis, y_axis, prim_color, 5.0f);
        break;
      }
      case kPrimDragRect: {
        lines.push_back(format_text("%s: x1=%.6g y1=%.6g x2=%.6g y2=%.6g",
                                    prefix.c_str(), static_cast<double>(p.f4),
                                    static_cast<double>(p.f5),
                                    static_cast<double>(p.f6),
                                    static_cast<double>(p.f7)));
        consider_rect_highlight(static_cast<double>(p.f4), static_cast<double>(p.f5),
                                static_cast<double>(p.f6), static_cast<double>(p.f7),
                                x_axis, y_axis, prim_color);
        break;
      }
      case kPrimBarGroups: {
        const int item_count = std::max(0, p.i1);
        const int group_count = std::max(0, p.i2);
        if (p.data0 == nullptr || item_count <= 0 || group_count <= 0) {
          break;
        }
        const double group_size = (p.f1 > 0.0f) ? static_cast<double>(p.f1) : 0.67;
        const double shift = static_cast<double>(p.f2);
        const int group = std::clamp(static_cast<int>(std::llround(mouse.x - shift)), 0,
                                     group_count - 1);
        const double group_center = static_cast<double>(group) + shift;
        const double item_w = group_size / std::max(1, item_count);
        const double group_left = group_center - 0.5 * group_size;
        const int item = std::clamp(
            static_cast<int>(std::floor((mouse.x - group_left) /
                                        std::max(1e-12, item_w))),
            0, item_count - 1);
        const std::size_t idx =
            static_cast<std::size_t>(item * group_count + group);
        if (idx >= static_cast<std::size_t>(item_count * group_count)) {
          break;
        }
        const std::vector<std::string> labels = split_labels(p.text, '\x1f');
        const std::string item_name =
            (static_cast<std::size_t>(item) < labels.size() &&
             !labels[static_cast<std::size_t>(item)].empty())
                ? labels[static_cast<std::size_t>(item)]
                : ("item" + std::to_string(item));
        const double v = static_cast<double>(p.data0[idx]);
        lines.push_back(format_text("%s: group=%d %s=%.6g", prefix.c_str(), group,
                                    item_name.c_str(), v));
        const double xc = group_left + (static_cast<double>(item) + 0.5) * item_w;
        const double half_w = std::max(1e-6, 0.45 * item_w);
        consider_rect_highlight(xc - half_w, std::min(0.0, v), xc + half_w,
                                std::max(0.0, v), x_axis, y_axis, prim_color);
        break;
      }
      default:
        break;
      }
    }

    if (has_highlight) {
      ImDrawList *dl = ImPlot::GetPlotDrawList();
      if (dl != nullptr) {
        switch (best_highlight.type) {
        case HighlightType::Point: {
          const ImVec2 px = ImPlot::PlotToPixels(
              best_highlight.x0, best_highlight.y0,
              static_cast<ImAxis>(best_highlight.x_axis),
              static_cast<ImAxis>(best_highlight.y_axis));
          if (is_finite_vec2(px)) {
            dl->AddCircleFilled(px, best_highlight.radius + 7.0f,
                                with_alpha(best_highlight.color, 36), 24);
            dl->AddCircleFilled(px, best_highlight.radius + 3.5f,
                                with_alpha(best_highlight.color, 80), 24);
            dl->AddCircleFilled(px, best_highlight.radius,
                                with_alpha(best_highlight.color, 230), 20);
            dl->AddCircle(px, best_highlight.radius + 1.2f,
                          IM_COL32(255, 255, 255, 230), 20, 1.4f);
          }
          break;
        }
        case HighlightType::Rect: {
          const ImVec2 p0 = ImPlot::PlotToPixels(
              best_highlight.x0, best_highlight.y0,
              static_cast<ImAxis>(best_highlight.x_axis),
              static_cast<ImAxis>(best_highlight.y_axis));
          const ImVec2 p1 = ImPlot::PlotToPixels(
              best_highlight.x1, best_highlight.y1,
              static_cast<ImAxis>(best_highlight.x_axis),
              static_cast<ImAxis>(best_highlight.y_axis));
          if (is_finite_vec2(p0) && is_finite_vec2(p1)) {
            const ImVec2 bmin(std::min(p0.x, p1.x), std::min(p0.y, p1.y));
            const ImVec2 bmax(std::max(p0.x, p1.x), std::max(p0.y, p1.y));
            dl->AddRectFilled(bmin, bmax, with_alpha(best_highlight.color, 34), 2.0f);
            dl->AddRect(bmin, bmax, with_alpha(best_highlight.color, 220), 2.0f, 0,
                        2.2f);
            dl->AddRect(bmin, bmax, IM_COL32(255, 255, 255, 160), 2.0f, 0, 1.0f);
          }
          break;
        }
        case HighlightType::VLine:
        case HighlightType::HLine: {
          const ImPlotRect lim = ImPlot::GetPlotLimits(
              static_cast<ImAxis>(best_highlight.x_axis),
              static_cast<ImAxis>(best_highlight.y_axis));
          ImVec2 p0;
          ImVec2 p1;
          if (best_highlight.type == HighlightType::VLine) {
            p0 = ImPlot::PlotToPixels(best_highlight.x0, lim.Y.Min,
                                      static_cast<ImAxis>(best_highlight.x_axis),
                                      static_cast<ImAxis>(best_highlight.y_axis));
            p1 = ImPlot::PlotToPixels(best_highlight.x0, lim.Y.Max,
                                      static_cast<ImAxis>(best_highlight.x_axis),
                                      static_cast<ImAxis>(best_highlight.y_axis));
          } else {
            p0 = ImPlot::PlotToPixels(lim.X.Min, best_highlight.x0,
                                      static_cast<ImAxis>(best_highlight.x_axis),
                                      static_cast<ImAxis>(best_highlight.y_axis));
            p1 = ImPlot::PlotToPixels(lim.X.Max, best_highlight.x0,
                                      static_cast<ImAxis>(best_highlight.x_axis),
                                      static_cast<ImAxis>(best_highlight.y_axis));
          }
          if (is_finite_vec2(p0) && is_finite_vec2(p1)) {
            dl->AddLine(p0, p1, with_alpha(best_highlight.color, 42), 6.0f);
            dl->AddLine(p0, p1, with_alpha(best_highlight.color, 220), 2.6f);
            dl->AddLine(p0, p1, IM_COL32(255, 255, 255, 140), 1.0f);
          }
          break;
        }
        }
      }
    }

    if (lines.empty()) {
      return;
    }
    ImGui::BeginTooltip();
    const std::size_t max_lines = 20U;
    const std::size_t count = std::min(lines.size(), max_lines);
    for (std::size_t i = 0; i < count; ++i) {
      ImGui::TextUnformatted(lines[i].c_str());
    }
    if (lines.size() > max_lines) {
      ImGui::Text("... +%d more", static_cast<int>(lines.size() - max_lines));
    }
    ImGui::EndTooltip();
  };

  auto capture_axis_views = [&]() {
    if (axis_view_min6 == nullptr || axis_view_max6 == nullptr) {
      return;
    }
    for (std::int32_t axis = 0; axis < 6; ++axis) {
      bool enabled = axis == 0 || axis == 3;
      if (axis_enabled6 != nullptr && axis != 0 && axis != 3) {
        enabled = axis_enabled6[axis] != 0;
      }
      if (!enabled) {
        continue;
      }
      if (axis <= 2) {
        const ImPlotRect lim =
            ImPlot::GetPlotLimits(static_cast<ImAxis>(axis), ImAxis_Y1);
        axis_view_min6[axis] = lim.X.Min;
        axis_view_max6[axis] = lim.X.Max;
      } else {
        const ImPlotRect lim =
            ImPlot::GetPlotLimits(ImAxis_X1, static_cast<ImAxis>(axis));
        axis_view_min6[axis] = lim.Y.Min;
        axis_view_max6[axis] = lim.Y.Max;
      }
    }
  };

  const std::int32_t rows = std::max(1, subplot_rows);
  const std::int32_t cols = std::max(1, subplot_cols);
  const bool use_aligned_group =
      aligned_group_enabled != 0 && aligned_group_id != nullptr &&
      aligned_group_id[0] != '\0';
  bool aligned_group_open = false;
  if (use_aligned_group) {
    aligned_group_open =
        ImPlot::BeginAlignedPlots(aligned_group_id, aligned_group_vertical != 0);
  }
  if (!use_aligned_group || aligned_group_open) {
    if (rows > 1 || cols > 1) {
      if (ImPlot::BeginSubplots(plot_title, rows, cols, ImVec2(-1.0f, -1.0f),
                                implot_subplot_flags)) {
        for (std::int32_t r = 0; r < rows; ++r) {
          for (std::int32_t c = 0; c < cols; ++c) {
            const std::int32_t subplot_index = r * cols + c;
            std::string subplot_id = "##nbp_subplot_" + std::to_string(subplot_index);
            if (ImPlot::BeginPlot(subplot_id.c_str(), ImVec2(-1.0f, -1.0f),
                                  implot_flags)) {
              setup_axes();
              draw_series_for_subplot(subplot_index, tmp_x, tmp_y);
              draw_primitives_for_subplot(subplot_index, tmp_x, tmp_y, label_ptrs,
                                          label_storage);
              render_hover_tooltip(subplot_index);
              if (selection_out6 != nullptr && ImPlot::IsPlotSelected()) {
                const ImPlotRect sel = ImPlot::GetPlotSelection();
                selection_out6[0] = 1.0f;
                selection_out6[1] = static_cast<float>(subplot_index);
                selection_out6[2] = static_cast<float>(sel.X.Min);
                selection_out6[3] = static_cast<float>(sel.X.Max);
                selection_out6[4] = static_cast<float>(sel.Y.Min);
                selection_out6[5] = static_cast<float>(sel.Y.Max);
              }
              if (subplot_index == 0) {
                const ImPlotRect limits = ImPlot::GetPlotLimits();
                if (x_min != nullptr) {
                  *x_min = static_cast<float>(limits.X.Min);
                }
                if (x_max != nullptr) {
                  *x_max = static_cast<float>(limits.X.Max);
                }
                if (y_min != nullptr) {
                  *y_min = static_cast<float>(limits.Y.Min);
                }
                if (y_max != nullptr) {
                  *y_max = static_cast<float>(limits.Y.Max);
                }
                capture_axis_views();
              }
              ImPlot::EndPlot();
              ok = true;
            }
          }
        }
        ImPlot::EndSubplots();
      }
    } else {
      if (ImPlot::BeginPlot(plot_title, ImVec2(-1.0f, -1.0f), implot_flags)) {
        setup_axes();
        draw_series_for_subplot(0, tmp_x, tmp_y);
        draw_primitives_for_subplot(0, tmp_x, tmp_y, label_ptrs, label_storage);
        render_hover_tooltip(0);
        if (selection_out6 != nullptr && ImPlot::IsPlotSelected()) {
          const ImPlotRect sel = ImPlot::GetPlotSelection();
          selection_out6[0] = 1.0f;
          selection_out6[1] = 0.0f;
          selection_out6[2] = static_cast<float>(sel.X.Min);
          selection_out6[3] = static_cast<float>(sel.X.Max);
          selection_out6[4] = static_cast<float>(sel.Y.Min);
          selection_out6[5] = static_cast<float>(sel.Y.Max);
        }
        const ImPlotRect limits = ImPlot::GetPlotLimits();
        if (x_min != nullptr) {
          *x_min = static_cast<float>(limits.X.Min);
        }
        if (x_max != nullptr) {
          *x_max = static_cast<float>(limits.X.Max);
        }
        if (y_min != nullptr) {
          *y_min = static_cast<float>(limits.Y.Min);
        }
        if (y_max != nullptr) {
          *y_max = static_cast<float>(limits.Y.Max);
        }
        capture_axis_views();
        ImPlot::EndPlot();
        ok = true;
      }
    }
  }
  if (aligned_group_open) {
    ImPlot::EndAlignedPlots();
  }

  if (pushed_colormap) {
    ImPlot::PopColormap();
  }

  for (auto it = state->image_textures.begin();
       it != state->image_textures.end();) {
    if (state->image_textures_touched.find(it->first) ==
        state->image_textures_touched.end()) {
      destroy_image_texture(it->second);
      it = state->image_textures.erase(it);
      continue;
    }
    ++it;
  }
  state->image_textures_touched.clear();

  ImGui::End();
  ImGui::Render();

  const auto fb_w = static_cast<GLsizei>(std::max(
      1.0f, std::round(static_cast<float>(state->width) * state->dpr)));
  const auto fb_h = static_cast<GLsizei>(std::max(
      1.0f, std::round(static_cast<float>(state->height) * state->dpr)));
  glViewport(0, 0, fb_w, fb_h);
  glDisable(GL_SCISSOR_TEST);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  return ok;
#else
  (void)draw_points;
  (void)point_count;
  (void)segments;
  (void)segment_count;
  (void)series_views;
  (void)series_count;
  (void)x_min;
  (void)x_max;
  (void)y_min;
  (void)y_max;
  (void)title_id;
  (void)force_view;
  (void)plot_flags;
  (void)axis_enabled6;
  (void)axis_scale6;
  (void)axis_labels6;
  (void)axis_formats6;
  (void)axis_limits_constraints_enabled6;
  (void)axis_limits_constraints_min6;
  (void)axis_limits_constraints_max6;
  (void)axis_zoom_constraints_enabled6;
  (void)axis_zoom_constraints_min6;
  (void)axis_zoom_constraints_max6;
  (void)axis_links6;
  (void)axis_view_min6;
  (void)axis_view_max6;
  (void)axis_ticks_values6;
  (void)axis_ticks_counts6;
  (void)axis_ticks_labels6;
  (void)axis_ticks_keep_default6;
  (void)subplot_rows;
  (void)subplot_cols;
  (void)subplot_flags;
  (void)aligned_group_enabled;
  (void)aligned_group_id;
  (void)aligned_group_vertical;
  (void)colormap_name;
  (void)primitives;
  (void)selection_out6;
  (void)primitive_count;
  return false;
#endif
}

} // namespace nbimplot
