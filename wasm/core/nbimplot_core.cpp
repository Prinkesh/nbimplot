#include "nbimplot_implot_layer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr float kRawSwitchFactor = 3.0f;
constexpr std::uint32_t kLodBlockSize = 256;

template <typename T> T clamp_value(T value, T min_value, T max_value) {
  return std::max(min_value, std::min(max_value, value));
}

struct LodBlockSummary {
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  std::int32_t min_index = -1;
  std::int32_t max_index = -1;
  bool has_finite = false;
};

struct Series {
  std::uint32_t id = 0;
  std::uint32_t slot = 0;
  std::string name;
  std::int32_t subplot_index = 0;
  std::int32_t x_axis = 0;
  std::int32_t y_axis = 3;
  bool has_custom_color = false;
  float color_r = 0.0f;
  float color_g = 0.0f;
  float color_b = 0.0f;
  float color_a = 0.0f;
  float line_weight = 1.0f;
  std::int32_t marker = -2;
  float marker_size = 4.0f;
  std::vector<float> raw;
  bool visible = true;
  std::uint64_t version = 0;

  std::int32_t lod_start = -1;
  std::int32_t lod_end = -1;
  std::uint32_t lod_width = 0;
  std::uint64_t lod_version = 0;
  std::vector<float> lod_xy;

  std::uint64_t block_cache_version = 0;
  std::vector<LodBlockSummary> block_cache;
};

enum PrimitiveKind : std::int32_t {
  kPrimScatter = 1,
  kPrimBubbles = 2,
  kPrimStairs = 3,
  kPrimStems = 4,
  kPrimDigital = 5,
  kPrimBars = 6,
  kPrimBarGroups = 7,
  kPrimBarsH = 8,
  kPrimShaded = 9,
  kPrimErrorBars = 10,
  kPrimErrorBarsH = 11,
  kPrimInfLines = 12,
  kPrimHistogram = 13,
  kPrimHistogram2D = 14,
  kPrimHeatmap = 15,
  kPrimImage = 16,
  kPrimPieChart = 17,
  kPrimText = 18,
  kPrimAnnotation = 19,
  kPrimDummy = 20,
  kPrimDragLineX = 21,
  kPrimDragLineY = 22,
  kPrimDragPoint = 23,
  kPrimDragRect = 24,
  kPrimTagX = 25,
  kPrimTagY = 26,
  kPrimColormapSlider = 27,
  kPrimColormapButton = 28,
  kPrimColormapSelector = 29,
  kPrimDragDropPlot = 30,
  kPrimDragDropAxis = 31,
  kPrimDragDropLegend = 32,
};

struct Primitive {
  std::uint32_t id = 0;
  std::int32_t kind = 0;
  bool visible = true;
  std::vector<float> data0;
  std::vector<float> data1;
  std::vector<float> data2;
  std::array<std::int32_t, 8> ints = {0, 0, 0, 0, 0, 0, 0, 0};
  std::array<float, 8> floats = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::string text;
};

struct AxisTicksConfig {
  std::vector<double> values;
  std::vector<std::string> labels;
  std::string labels_blob;
  std::int32_t keep_default = 0;
};

struct PlotCore {
  std::unordered_map<std::uint32_t, Series> series_by_id;
  std::vector<std::uint32_t> order;
  std::unordered_map<std::uint32_t, Primitive> primitives_by_id;
  std::vector<std::uint32_t> primitive_order;
  bool primitive_views_dirty = true;

  float x_min = 0.0f;
  float x_max = 1.0f;
  float y_min = -1.0f;
  float y_max = 1.0f;
  bool view_initialized = false;
  std::int32_t plot_flags = 0;
  std::array<std::int32_t, 6> axis_enabled = {1, 0, 0, 1, 0, 0};
  std::array<std::int32_t, 6> axis_scales = {0, 0, 0, 0, 0, 0};
  std::array<std::string, 6> axis_labels;
  std::array<std::string, 6> axis_formats;
  std::array<AxisTicksConfig, 6> axis_ticks;
  std::array<std::int32_t, 6> axis_limits_constraints_enabled = {0, 0, 0, 0, 0, 0};
  std::array<double, 6> axis_limits_constraints_min = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double, 6> axis_limits_constraints_max = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<std::int32_t, 6> axis_zoom_constraints_enabled = {0, 0, 0, 0, 0, 0};
  std::array<double, 6> axis_zoom_constraints_min = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double, 6> axis_zoom_constraints_max = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<std::int32_t, 6> axis_links = {-1, -1, -1, -1, -1, -1};
  std::array<double, 6> axis_view_min = {0.0, 0.0, 0.0, -1.0, -1.0, -1.0};
  std::array<double, 6> axis_view_max = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::int32_t subplot_rows = 1;
  std::int32_t subplot_cols = 1;
  std::int32_t subplot_flags = 0;
  std::string aligned_group_id;
  std::int32_t aligned_group_enabled = 0;
  std::int32_t aligned_group_vertical = 1;
  std::string colormap_name;

  std::int32_t canvas_width = 900;
  std::int32_t canvas_height = 450;
  float dpr = 1.0f;
  std::string canvas_selector;

  bool implot_enabled = false;
  bool implot_force_view = true;
  nbimplot::ImPlotLayer implot_layer;

  float mouse_x = 0.0f;
  float mouse_y = 0.0f;
  bool mouse_inside = false;
  std::array<bool, 5> mouse_down = {false, false, false, false, false};
  float wheel_x = 0.0f;
  float wheel_y = 0.0f;

  // Packed draw tuples:
  // [series_slot, x, y, pen_down]
  std::vector<float> draw_tuples;
  std::vector<nbimplot::DrawSegmentView> draw_segments;
  std::vector<nbimplot::SeriesView> series_views;
  std::vector<nbimplot::PrimitiveView> primitive_views;
  std::vector<float> interaction_tuples;

  float last_lod_ms = 0.0f;
  float last_segment_build_ms = 0.0f;
  float last_render_ms = 0.0f;
  float last_frame_ms = 0.0f;
  std::uint32_t last_pixel_width = 0;
};

std::unordered_map<std::uint32_t, std::unique_ptr<PlotCore>> g_instances;
std::uint32_t g_next_handle = 1;

PlotCore *get_plot(std::uint32_t handle) {
  const auto it = g_instances.find(handle);
  if (it == g_instances.end()) {
    return nullptr;
  }
  return it->second.get();
}

std::vector<std::string> split_blob(const char *text, char delim) {
  std::vector<std::string> out;
  if (text == nullptr || text[0] == '\0') {
    return out;
  }
  std::string cur;
  for (const char *p = text; *p != '\0'; ++p) {
    if (*p == delim) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(*p);
    }
  }
  out.push_back(cur);
  return out;
}

void reset_lod_cache(Series &series) {
  series.lod_start = -1;
  series.lod_end = -1;
  series.lod_width = 0;
  series.lod_version = 0;
  series.lod_xy.clear();
  series.block_cache_version = 0;
  series.block_cache.clear();
}

void append_draw_point(std::vector<float> &draw_tuples, std::uint32_t slot,
                       float x, float y, bool pen_down) {
  draw_tuples.push_back(static_cast<float>(slot));
  draw_tuples.push_back(x);
  draw_tuples.push_back(y);
  draw_tuples.push_back(pen_down ? 1.0f : 0.0f);
}

void ensure_lod_block_cache(Series &series) {
  if (series.block_cache_version == series.version) {
    return;
  }

  if (series.raw.empty()) {
    series.block_cache.clear();
    series.block_cache_version = series.version;
    return;
  }

  const std::size_t n = series.raw.size();
  const std::size_t block_count =
      (n + static_cast<std::size_t>(kLodBlockSize) - 1U) /
      static_cast<std::size_t>(kLodBlockSize);
  series.block_cache.clear();
  series.block_cache.resize(block_count);

  for (std::size_t block = 0; block < block_count; ++block) {
    const std::size_t begin = block * static_cast<std::size_t>(kLodBlockSize);
    const std::size_t end = std::min(
        begin + static_cast<std::size_t>(kLodBlockSize), series.raw.size());

    LodBlockSummary summary;
    for (std::size_t i = begin; i < end; ++i) {
      const float value = series.raw[i];
      if (!std::isfinite(value)) {
        continue;
      }
      if (!summary.has_finite || value < summary.min_value) {
        summary.min_value = value;
        summary.min_index = static_cast<std::int32_t>(i);
      }
      if (!summary.has_finite || value > summary.max_value) {
        summary.max_value = value;
        summary.max_index = static_cast<std::int32_t>(i);
      }
      summary.has_finite = true;
    }
    series.block_cache[block] = summary;
  }

  series.block_cache_version = series.version;
}

void build_min_max_lod(Series &series, std::int32_t start, std::int32_t end,
                       std::uint32_t pixel_width) {
  if (start < 0 || end < start || pixel_width == 0) {
    series.lod_xy.clear();
    series.lod_start = start;
    series.lod_end = end;
    series.lod_width = pixel_width;
    series.lod_version = series.version;
    return;
  }

  if (series.lod_version == series.version && series.lod_start == start &&
      series.lod_end == end && series.lod_width == pixel_width) {
    return;
  }

  ensure_lod_block_cache(series);

  const std::int32_t visible_count = end - start + 1;
  const std::int32_t block_size = static_cast<std::int32_t>(kLodBlockSize);
  const float *raw_data = series.raw.data();

  series.lod_xy.clear();
  series.lod_xy.reserve(static_cast<std::size_t>(pixel_width) * 4);

  for (std::uint32_t bucket = 0; bucket < pixel_width; ++bucket) {
    const std::int32_t bucket_start =
        start + static_cast<std::int32_t>((static_cast<std::int64_t>(bucket) *
                                           visible_count) /
                                          pixel_width);
    const std::int32_t bucket_end =
        start +
        static_cast<std::int32_t>((static_cast<std::int64_t>(bucket + 1U) *
                                   visible_count) /
                                  pixel_width);
    if (bucket_end <= bucket_start) {
      continue;
    }

    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    std::int32_t min_index = bucket_start;
    std::int32_t max_index = bucket_start;

    auto update_point = [&](std::int32_t idx, float value) {
      if (!std::isfinite(value)) {
        return;
      }
      if (value < min_value) {
        min_value = value;
        min_index = idx;
      }
      if (value > max_value) {
        max_value = value;
        max_index = idx;
      }
    };

    std::int32_t i = bucket_start;
    if (bucket_end - bucket_start >= block_size * 2 &&
        !series.block_cache.empty()) {
      const std::int32_t aligned_start =
          ((bucket_start + block_size - 1) / block_size) * block_size;
      const std::int32_t aligned_end = (bucket_end / block_size) * block_size;

      for (; i < std::min(bucket_end, aligned_start); ++i) {
        update_point(i, raw_data[static_cast<std::size_t>(i)]);
      }

      for (std::int32_t block_begin = aligned_start; block_begin < aligned_end;
           block_begin += block_size) {
        const std::size_t block_idx =
            static_cast<std::size_t>(block_begin / block_size);
        if (block_idx >= series.block_cache.size()) {
          break;
        }
        const LodBlockSummary &summary = series.block_cache[block_idx];
        if (!summary.has_finite) {
          continue;
        }
        update_point(summary.min_index, summary.min_value);
        update_point(summary.max_index, summary.max_value);
      }
      i = std::max(i, aligned_end);
    }

    for (; i < bucket_end; ++i) {
      update_point(i, raw_data[static_cast<std::size_t>(i)]);
    }

    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
      continue;
    }

    if (min_index <= max_index) {
      series.lod_xy.push_back(static_cast<float>(min_index));
      series.lod_xy.push_back(min_value);
      if (max_index != min_index) {
        series.lod_xy.push_back(static_cast<float>(max_index));
        series.lod_xy.push_back(max_value);
      }
    } else {
      series.lod_xy.push_back(static_cast<float>(max_index));
      series.lod_xy.push_back(max_value);
      if (max_index != min_index) {
        series.lod_xy.push_back(static_cast<float>(min_index));
        series.lod_xy.push_back(min_value);
      }
    }
  }

  series.lod_start = start;
  series.lod_end = end;
  series.lod_width = pixel_width;
  series.lod_version = series.version;
}

void autoscale(PlotCore &plot) {
  float x_min = std::numeric_limits<float>::infinity();
  float x_max = -std::numeric_limits<float>::infinity();
  float y_min = std::numeric_limits<float>::infinity();
  float y_max = -std::numeric_limits<float>::infinity();
  bool has_any = false;

  for (const std::uint32_t series_id : plot.order) {
    const auto it = plot.series_by_id.find(series_id);
    if (it == plot.series_by_id.end()) {
      continue;
    }
    const Series &series = it->second;
    if (!series.visible || series.raw.empty()) {
      continue;
    }
    has_any = true;
    x_min = std::min(x_min, 0.0f);
    x_max = std::max(x_max, static_cast<float>(series.raw.size() - 1));
    for (float value : series.raw) {
      if (!std::isfinite(value)) {
        continue;
      }
      y_min = std::min(y_min, value);
      y_max = std::max(y_max, value);
    }
  }

  auto update_point = [&](float x, float y) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
      return;
    }
    has_any = true;
    x_min = std::min(x_min, x);
    x_max = std::max(x_max, x);
    y_min = std::min(y_min, y);
    y_max = std::max(y_max, y);
  };

  for (const std::uint32_t prim_id : plot.primitive_order) {
    const auto it = plot.primitives_by_id.find(prim_id);
    if (it == plot.primitives_by_id.end()) {
      continue;
    }
    const Primitive &p = it->second;
    if (!p.visible) {
      continue;
    }
    const bool has_x = p.ints[0] != 0;

    switch (p.kind) {
    case kPrimScatter:
    case kPrimStairs:
    case kPrimStems:
    case kPrimDigital:
    case kPrimBars: {
      const auto &ys = has_x ? p.data1 : p.data0;
      const auto &xs = has_x ? p.data0 : p.data1;
      const std::size_t n = ys.size();
      for (std::size_t i = 0; i < n; ++i) {
        const float x = has_x ? xs[i] : static_cast<float>(i);
        update_point(x, ys[i]);
        if (p.kind == kPrimBars) {
          update_point(x, 0.0f);
        }
      }
      break;
    }
    case kPrimBubbles: {
      const auto &ys = has_x ? p.data1 : p.data0;
      const auto &sz = has_x ? p.data2 : p.data1;
      const auto &xs = has_x ? p.data0 : p.data2;
      const std::size_t n = std::min(ys.size(), sz.size());
      for (std::size_t i = 0; i < n; ++i) {
        const float x = has_x ? xs[i] : static_cast<float>(i);
        const float y = ys[i];
        const float s = std::fabs(sz[i]);
        update_point(x - s, y - s);
        update_point(x + s, y + s);
      }
      break;
    }
    case kPrimBarsH: {
      const auto &xs = p.data0;
      const auto &ys = p.data1;
      const std::size_t n = std::min(xs.size(), ys.size());
      for (std::size_t i = 0; i < n; ++i) {
        update_point(xs[i], ys[i]);
        update_point(0.0f, ys[i]);
      }
      break;
    }
    case kPrimBarGroups: {
      const int item_count = std::max(1, p.ints[1]);
      const int group_count = std::max(0, p.ints[2]);
      const float group_size = std::fabs(p.floats[1]);
      const float shift = p.floats[2];
      for (int g = 0; g < group_count; ++g) {
        const float gc = static_cast<float>(g) + shift;
        update_point(gc - 0.5f * group_size, 0.0f);
        update_point(gc + 0.5f * group_size, 0.0f);
        for (int i = 0; i < item_count; ++i) {
          const std::size_t idx = static_cast<std::size_t>(i * group_count + g);
          if (idx >= p.data0.size()) {
            continue;
          }
          update_point(gc, p.data0[idx]);
        }
      }
      break;
    }
    case kPrimShaded: {
      const auto &x = has_x ? p.data0 : p.data2;
      const auto &y1 = has_x ? p.data1 : p.data0;
      const auto &y2 = has_x ? p.data2 : p.data1;
      const std::size_t n = std::min(y1.size(), y2.size());
      for (std::size_t i = 0; i < n; ++i) {
        const float xx = has_x ? x[i] : static_cast<float>(i);
        update_point(xx, y1[i]);
        update_point(xx, y2[i]);
      }
      break;
    }
    case kPrimErrorBars: {
      const bool asym = p.ints[1] != 0;
      if (asym) {
        if (has_x) {
          const auto &x = p.data0;
          const auto &y = p.data1;
          const auto &ep = p.data2;
          const std::size_t n = std::min(x.size(), std::min(y.size(), ep.size() / 2U));
          for (std::size_t i = 0; i < n; ++i) {
            const float en = std::fabs(ep[i * 2U + 0U]);
            const float epv = std::fabs(ep[i * 2U + 1U]);
            update_point(x[i], y[i] - en);
            update_point(x[i], y[i] + epv);
          }
        } else {
          const auto &y = p.data0;
          const auto &ep = p.data1;
          const std::size_t n = std::min(y.size(), ep.size() / 2U);
          for (std::size_t i = 0; i < n; ++i) {
            const float en = std::fabs(ep[i * 2U + 0U]);
            const float epv = std::fabs(ep[i * 2U + 1U]);
            update_point(static_cast<float>(i), y[i] - en);
            update_point(static_cast<float>(i), y[i] + epv);
          }
        }
      } else {
        const auto &x = has_x ? p.data0 : p.data2;
        const auto &y = has_x ? p.data1 : p.data0;
        const auto &e = has_x ? p.data2 : p.data1;
        const std::size_t n = std::min(y.size(), e.size());
        for (std::size_t i = 0; i < n; ++i) {
          const float xx = has_x ? x[i] : static_cast<float>(i);
          const float ee = std::fabs(e[i]);
          update_point(xx, y[i] - ee);
          update_point(xx, y[i] + ee);
        }
      }
      break;
    }
    case kPrimErrorBarsH: {
      const bool asym = p.ints[1] != 0;
      const auto &x = p.data0;
      const auto &y = p.data2;
      if (asym) {
        const auto &ep = p.data1;
        const std::size_t n = std::min(x.size(), std::min(y.size(), ep.size() / 2U));
        for (std::size_t i = 0; i < n; ++i) {
          const float en = std::fabs(ep[i * 2U + 0U]);
          const float epv = std::fabs(ep[i * 2U + 1U]);
          update_point(x[i] - en, y[i]);
          update_point(x[i] + epv, y[i]);
        }
      } else {
        const auto &e = p.data1;
        const std::size_t n = std::min(x.size(), std::min(e.size(), y.size()));
        for (std::size_t i = 0; i < n; ++i) {
          const float ee = std::fabs(e[i]);
          update_point(x[i] - ee, y[i]);
          update_point(x[i] + ee, y[i]);
        }
      }
      break;
    }
    case kPrimInfLines: {
      const int axis = p.ints[1];
      for (float v : p.data0) {
        if (!std::isfinite(v)) {
          continue;
        }
        if (axis == 1) {
          update_point(0.0f, v);
        } else {
          update_point(v, 0.0f);
        }
      }
      break;
    }
    case kPrimHistogram: {
      for (float edge : p.data0) {
        if (!std::isfinite(edge)) {
          continue;
        }
        has_any = true;
        x_min = std::min(x_min, edge);
        x_max = std::max(x_max, edge);
      }
      for (float c : p.data1) {
        if (!std::isfinite(c)) {
          continue;
        }
        has_any = true;
        y_min = std::min(y_min, 0.0f);
        y_max = std::max(y_max, c);
      }
      break;
    }
    case kPrimHistogram2D: {
      if (p.data0.size() > 1 && p.data1.size() > 1) {
        update_point(p.data0.front(), p.data1.front());
        update_point(p.data0.back(), p.data1.back());
      }
      break;
    }
    case kPrimHeatmap: {
      const int rows = std::max(0, p.ints[1]);
      const int cols = std::max(0, p.ints[2]);
      if (rows > 0 && cols > 0) {
        update_point(0.0f, 0.0f);
        update_point(static_cast<float>(cols), static_cast<float>(rows));
      }
      break;
    }
    case kPrimImage: {
      const int rows = std::max(0, p.ints[1]);
      const int cols = std::max(0, p.ints[2]);
      if (rows > 0 && cols > 0) {
        const float x0 = std::isfinite(p.floats[0]) ? p.floats[0] : 0.0f;
        const float x1 = std::isfinite(p.floats[1]) ? p.floats[1] : static_cast<float>(cols);
        const float y0 = std::isfinite(p.floats[2]) ? p.floats[2] : 0.0f;
        const float y1 = std::isfinite(p.floats[3]) ? p.floats[3] : static_cast<float>(rows);
        update_point(x0, y0);
        update_point(x1, y1);
      }
      break;
    }
    case kPrimPieChart: {
      const float cx = p.floats[4];
      const float cy = p.floats[5];
      const float r = std::fabs(p.floats[6]);
      update_point(cx - r, cy - r);
      update_point(cx + r, cy + r);
      break;
    }
    case kPrimText:
    case kPrimAnnotation: {
      update_point(p.floats[4], p.floats[5]);
      break;
    }
    case kPrimDragLineX: {
      update_point(p.floats[4], 0.0f);
      break;
    }
    case kPrimDragLineY: {
      update_point(0.0f, p.floats[5]);
      break;
    }
    case kPrimDragPoint: {
      update_point(p.floats[4], p.floats[5]);
      break;
    }
    case kPrimDragRect: {
      update_point(p.floats[4], p.floats[5]);
      update_point(p.floats[6], p.floats[7]);
      break;
    }
    default:
      break;
    }
  }

  if (!has_any || !std::isfinite(y_min) || !std::isfinite(y_max)) {
    y_min = -1.0f;
    y_max = 1.0f;
  } else if (y_min == y_max) {
    const float pad = std::fabs(y_min) * 0.05f + 1.0f;
    y_min -= pad;
    y_max += pad;
  }

  if (!has_any || !std::isfinite(x_min) || !std::isfinite(x_max)) {
    x_min = 0.0f;
    x_max = 1.0f;
  } else if (x_min == x_max) {
    const float pad = std::fabs(x_min) * 0.05f + 1.0f;
    x_min -= pad;
    x_max += pad;
  }

  plot.x_min = x_min;
  plot.x_max = x_max;
  plot.y_min = y_min;
  plot.y_max = y_max;
  plot.view_initialized = true;
}

} // namespace

extern "C" {

std::uint32_t nbp_create() {
  const std::uint32_t handle = g_next_handle++;
  g_instances.emplace(handle, std::make_unique<PlotCore>());
  return handle;
}

void nbp_destroy(std::uint32_t handle) {
  const auto it = g_instances.find(handle);
  if (it == g_instances.end()) {
    return;
  }
  PlotCore *plot = it->second.get();
  if (plot->implot_enabled) {
    plot->implot_layer.shutdown();
    plot->implot_enabled = false;
  }
  g_instances.erase(it);
}

std::int32_t nbp_set_canvas(std::uint32_t handle, std::int32_t width,
                            std::int32_t height, float dpr) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  plot->canvas_width = std::max(1, width);
  plot->canvas_height = std::max(1, height);
  plot->dpr = std::max(1.0f, dpr);
  if (plot->implot_enabled) {
    plot->implot_layer.set_canvas(plot->canvas_width, plot->canvas_height, plot->dpr);
  }
  return 0;
}

std::int32_t nbp_set_canvas_selector(std::uint32_t handle, const char *selector) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || selector == nullptr || selector[0] == '\0') {
    return -1;
  }
  plot->canvas_selector = selector;
  return 0;
}

std::int32_t nbp_line_set_data(std::uint32_t handle, std::uint32_t series_id,
                               const float *data, std::uint32_t len,
                               std::int32_t is_new_series) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || data == nullptr) {
    return -1;
  }

  auto it = plot->series_by_id.find(series_id);
  if (it == plot->series_by_id.end()) {
    if (!is_new_series) {
      // If the caller marks this as update but series does not exist, create it.
      // This keeps the transport resilient across frontend retries.
    }
    Series series;
    series.id = series_id;
    series.slot = static_cast<std::uint32_t>(plot->order.size());
    series.name = "series_" + std::to_string(series.slot);
    series.visible = true;
    series.raw.assign(data, data + len);
    series.version = 1;
    reset_lod_cache(series);
    plot->order.push_back(series_id);
    plot->series_by_id.emplace(series_id, std::move(series));
    return 0;
  }

  Series &series = it->second;
  series.raw.assign(data, data + len);
  series.version += 1;
  reset_lod_cache(series);
  return 0;
}

std::int32_t nbp_line_append_data(std::uint32_t handle, std::uint32_t series_id,
                                  const float *data, std::uint32_t len,
                                  std::uint32_t max_points) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || data == nullptr) {
    return -1;
  }
  auto it = plot->series_by_id.find(series_id);
  if (it == plot->series_by_id.end()) {
    return -1;
  }
  if (len == 0U) {
    return 0;
  }
  Series &series = it->second;
  const std::size_t old_size = series.raw.size();
  series.raw.resize(old_size + static_cast<std::size_t>(len));
  std::copy(data, data + len, series.raw.begin() + static_cast<std::ptrdiff_t>(old_size));
  if (max_points > 0U && series.raw.size() > static_cast<std::size_t>(max_points)) {
    const std::size_t trim = series.raw.size() - static_cast<std::size_t>(max_points);
    series.raw.erase(series.raw.begin(), series.raw.begin() + static_cast<std::ptrdiff_t>(trim));
  }
  series.version += 1;
  reset_lod_cache(series);
  return 0;
}

std::int32_t nbp_primitive_set_data(
    std::uint32_t handle, std::uint32_t primitive_id, std::int32_t kind,
    const float *data0, std::uint32_t len0, const float *data1, std::uint32_t len1,
    const float *data2, std::uint32_t len2, std::int32_t i0, std::int32_t i1,
    std::int32_t i2, std::int32_t i3, std::int32_t i4, std::int32_t i5,
    std::int32_t i6, std::int32_t i7, float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7, const char *text) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || primitive_id == 0) {
    return -1;
  }
  if (kind < kPrimScatter || kind > kPrimDragDropLegend) {
    return -1;
  }

  auto it = plot->primitives_by_id.find(primitive_id);
  if (it == plot->primitives_by_id.end()) {
    Primitive prim;
    prim.id = primitive_id;
    prim.kind = kind;
    plot->primitive_order.push_back(primitive_id);
    it = plot->primitives_by_id.emplace(primitive_id, std::move(prim)).first;
  }

  Primitive &prim = it->second;
  prim.kind = kind;
  if (data0 != nullptr && len0 > 0) {
    prim.data0.assign(data0, data0 + len0);
  } else {
    prim.data0.clear();
  }
  if (data1 != nullptr && len1 > 0) {
    prim.data1.assign(data1, data1 + len1);
  } else {
    prim.data1.clear();
  }
  if (data2 != nullptr && len2 > 0) {
    prim.data2.assign(data2, data2 + len2);
  } else {
    prim.data2.clear();
  }
  prim.ints = {i0, i1, i2, i3, i4, i5, i6, i7};
  prim.floats = {f0, f1, f2, f3, f4, f5, f6, f7};
  prim.text = text != nullptr ? text : "";
  plot->primitive_views_dirty = true;
  return 0;
}

std::int32_t nbp_primitive_remove(std::uint32_t handle, std::uint32_t primitive_id) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  const auto it = plot->primitives_by_id.find(primitive_id);
  if (it == plot->primitives_by_id.end()) {
    return 0;
  }
  plot->primitives_by_id.erase(it);
  plot->primitive_order.erase(
      std::remove(plot->primitive_order.begin(), plot->primitive_order.end(),
                  primitive_id),
      plot->primitive_order.end());
  plot->primitive_views_dirty = true;
  return 0;
}

std::int32_t nbp_primitive_set_visible(std::uint32_t handle, std::uint32_t primitive_id,
                                       std::int32_t visible) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  auto it = plot->primitives_by_id.find(primitive_id);
  if (it == plot->primitives_by_id.end()) {
    return -1;
  }
  it->second.visible = visible != 0;
  plot->primitive_views_dirty = true;
  return 0;
}

std::int32_t nbp_set_series_visible(std::uint32_t handle, std::uint32_t series_id,
                                    std::int32_t visible) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  auto it = plot->series_by_id.find(series_id);
  if (it == plot->series_by_id.end()) {
    return -1;
  }
  it->second.visible = visible != 0;
  return 0;
}

std::int32_t nbp_line_set_name(std::uint32_t handle, std::uint32_t series_id,
                               const char *name) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  auto it = plot->series_by_id.find(series_id);
  if (it == plot->series_by_id.end()) {
    return -1;
  }
  if (name == nullptr || name[0] == '\0') {
    it->second.name = "series_" + std::to_string(it->second.slot);
  } else {
    it->second.name = name;
  }
  return 0;
}

std::int32_t nbp_set_series_subplot(std::uint32_t handle, std::uint32_t series_id,
                                    std::int32_t subplot_index) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  auto it = plot->series_by_id.find(series_id);
  if (it == plot->series_by_id.end()) {
    return -1;
  }
  it->second.subplot_index = std::max(0, subplot_index);
  return 0;
}

std::int32_t nbp_set_series_axes(std::uint32_t handle, std::uint32_t series_id,
                                 std::int32_t x_axis, std::int32_t y_axis) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  auto it = plot->series_by_id.find(series_id);
  if (it == plot->series_by_id.end()) {
    return -1;
  }
  if (x_axis < 0 || x_axis > 2 || y_axis < 3 || y_axis > 5) {
    return -1;
  }
  it->second.x_axis = x_axis;
  it->second.y_axis = y_axis;
  return 0;
}

std::int32_t nbp_set_series_style(std::uint32_t handle, std::uint32_t series_id,
                                  std::int32_t has_color, float color_r, float color_g,
                                  float color_b, float color_a, float line_weight,
                                  std::int32_t marker, float marker_size) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  auto it = plot->series_by_id.find(series_id);
  if (it == plot->series_by_id.end()) {
    return -1;
  }
  Series &series = it->second;
  if (!std::isfinite(line_weight) || line_weight <= 0.0f) {
    return -1;
  }
  if (!std::isfinite(marker_size) || marker_size <= 0.0f) {
    return -1;
  }
  series.line_weight = line_weight;
  series.marker = clamp_value<std::int32_t>(marker, -2, 9);
  series.marker_size = marker_size;
  if (has_color == 0) {
    series.has_custom_color = false;
    return 0;
  }
  if (!std::isfinite(color_r) || !std::isfinite(color_g) || !std::isfinite(color_b) ||
      !std::isfinite(color_a)) {
    return -1;
  }
  auto clamp01 = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };
  series.has_custom_color = true;
  series.color_r = clamp01(color_r);
  series.color_g = clamp01(color_g);
  series.color_b = clamp01(color_b);
  series.color_a = clamp01(color_a);
  return 0;
}

std::int32_t nbp_set_plot_options(std::uint32_t handle, std::int32_t plot_flags,
                                  std::int32_t axis_scale_x,
                                  std::int32_t axis_scale_y) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  plot->plot_flags = std::max(0, plot_flags);
  plot->axis_enabled[0] = 1;
  plot->axis_enabled[3] = 1;
  plot->axis_scales[0] = clamp_value<std::int32_t>(axis_scale_x, 0, 2);
  plot->axis_scales[3] = clamp_value<std::int32_t>(axis_scale_y, 0, 2);
  return 0;
}

std::int32_t nbp_set_axis_state(std::uint32_t handle, std::int32_t axis_index,
                                std::int32_t enabled, std::int32_t scale) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || axis_index < 0 || axis_index >= 6) {
    return -1;
  }
  if (axis_index == 0 || axis_index == 3) {
    plot->axis_enabled[static_cast<std::size_t>(axis_index)] = 1;
  } else {
    plot->axis_enabled[static_cast<std::size_t>(axis_index)] = enabled != 0 ? 1 : 0;
  }
  plot->axis_scales[static_cast<std::size_t>(axis_index)] =
      clamp_value<std::int32_t>(scale, 0, 2);
  return 0;
}

std::int32_t nbp_set_axis_label(std::uint32_t handle, std::int32_t axis_index,
                                const char *label) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || axis_index < 0 || axis_index >= 6) {
    return -1;
  }
  plot->axis_labels[static_cast<std::size_t>(axis_index)] =
      (label == nullptr) ? "" : std::string(label);
  return 0;
}

std::int32_t nbp_set_axis_format(std::uint32_t handle, std::int32_t axis_index,
                                 const char *fmt) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || axis_index < 0 || axis_index >= 6) {
    return -1;
  }
  plot->axis_formats[static_cast<std::size_t>(axis_index)] =
      (fmt == nullptr) ? "" : std::string(fmt);
  return 0;
}

std::int32_t nbp_set_axis_ticks(std::uint32_t handle, std::int32_t axis_index,
                                const float *values, std::uint32_t count,
                                const char *labels_blob,
                                std::int32_t keep_default) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || axis_index < 0 || axis_index >= 6) {
    return -1;
  }
  AxisTicksConfig &cfg = plot->axis_ticks[static_cast<std::size_t>(axis_index)];
  cfg.values.clear();
  cfg.labels.clear();
  cfg.labels_blob.clear();
  cfg.keep_default = keep_default != 0 ? 1 : 0;
  if (values != nullptr && count > 0U) {
    cfg.values.reserve(static_cast<std::size_t>(count));
    for (std::uint32_t i = 0; i < count; ++i) {
      cfg.values.push_back(static_cast<double>(values[i]));
    }
  }
  cfg.labels = split_blob(labels_blob, '\x1f');
  if (!cfg.labels.empty()) {
    for (std::size_t i = 0; i < cfg.labels.size(); ++i) {
      if (i > 0U) {
        cfg.labels_blob.push_back('\x1f');
      }
      cfg.labels_blob += cfg.labels[i];
    }
  }
  return 0;
}

std::int32_t nbp_clear_axis_ticks(std::uint32_t handle, std::int32_t axis_index) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || axis_index < 0 || axis_index >= 6) {
    return -1;
  }
  AxisTicksConfig &cfg = plot->axis_ticks[static_cast<std::size_t>(axis_index)];
  cfg.values.clear();
  cfg.labels.clear();
  cfg.labels_blob.clear();
  cfg.keep_default = 0;
  return 0;
}

std::int32_t nbp_set_axis_limits_constraints(std::uint32_t handle,
                                             std::int32_t axis_index,
                                             std::int32_t enabled, double v_min,
                                             double v_max) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || axis_index < 0 || axis_index >= 6) {
    return -1;
  }
  const std::size_t idx = static_cast<std::size_t>(axis_index);
  if (enabled == 0) {
    plot->axis_limits_constraints_enabled[idx] = 0;
    return 0;
  }
  if (!std::isfinite(v_min) || !std::isfinite(v_max) || v_max <= v_min) {
    return -1;
  }
  plot->axis_limits_constraints_enabled[idx] = 1;
  plot->axis_limits_constraints_min[idx] = v_min;
  plot->axis_limits_constraints_max[idx] = v_max;
  return 0;
}

std::int32_t nbp_set_axis_zoom_constraints(std::uint32_t handle,
                                           std::int32_t axis_index,
                                           std::int32_t enabled, double z_min,
                                           double z_max) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || axis_index < 0 || axis_index >= 6) {
    return -1;
  }
  const std::size_t idx = static_cast<std::size_t>(axis_index);
  if (enabled == 0) {
    plot->axis_zoom_constraints_enabled[idx] = 0;
    return 0;
  }
  if (!std::isfinite(z_min) || !std::isfinite(z_max) || z_max <= z_min) {
    return -1;
  }
  plot->axis_zoom_constraints_enabled[idx] = 1;
  plot->axis_zoom_constraints_min[idx] = z_min;
  plot->axis_zoom_constraints_max[idx] = z_max;
  return 0;
}

std::int32_t nbp_set_axis_link(std::uint32_t handle, std::int32_t axis_index,
                               std::int32_t target_axis) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || axis_index < 0 || axis_index >= 6) {
    return -1;
  }
  if (target_axis < 0 || target_axis >= 6 || target_axis == axis_index) {
    plot->axis_links[static_cast<std::size_t>(axis_index)] = -1;
    return 0;
  }
  const bool axis_is_x = axis_index <= 2;
  const bool target_is_x = target_axis <= 2;
  if (axis_is_x != target_is_x) {
    return -1;
  }
  plot->axis_links[static_cast<std::size_t>(axis_index)] = target_axis;
  return 0;
}

std::int32_t nbp_set_subplots(std::uint32_t handle, std::int32_t rows,
                              std::int32_t cols, std::int32_t subplot_flags) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  plot->subplot_rows = std::max(1, rows);
  plot->subplot_cols = std::max(1, cols);
  plot->subplot_flags = std::max(0, subplot_flags);
  return 0;
}

std::int32_t nbp_set_aligned_group(std::uint32_t handle, const char *group_id,
                                   std::int32_t enabled,
                                   std::int32_t vertical) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  plot->aligned_group_id = (group_id == nullptr) ? "" : std::string(group_id);
  plot->aligned_group_enabled =
      (enabled != 0 && !plot->aligned_group_id.empty()) ? 1 : 0;
  plot->aligned_group_vertical = vertical != 0 ? 1 : 0;
  return 0;
}

std::int32_t nbp_set_colormap(std::uint32_t handle, const char *name) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  if (name == nullptr || name[0] == '\0') {
    plot->colormap_name.clear();
  } else {
    plot->colormap_name = name;
  }
  return 0;
}

std::int32_t nbp_set_view(std::uint32_t handle, float x_min, float x_max,
                          float y_min, float y_max) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  if (!std::isfinite(x_min) || !std::isfinite(x_max) || !std::isfinite(y_min) ||
      !std::isfinite(y_max)) {
    return -1;
  }

  if (x_max <= x_min) {
    x_max = x_min + 1.0f;
  }
  if (y_max <= y_min) {
    y_max = y_min + 1.0f;
  }

  plot->x_min = x_min;
  plot->x_max = x_max;
  plot->y_min = y_min;
  plot->y_max = y_max;
  plot->axis_view_min[0] = static_cast<double>(x_min);
  plot->axis_view_max[0] = static_cast<double>(x_max);
  plot->axis_view_min[3] = static_cast<double>(y_min);
  plot->axis_view_max[3] = static_cast<double>(y_max);
  plot->view_initialized = true;
  plot->implot_force_view = true;
  return 0;
}

std::int32_t nbp_get_view(std::uint32_t handle, float *out_view4) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || out_view4 == nullptr) {
    return -1;
  }
  out_view4[0] = plot->x_min;
  out_view4[1] = plot->x_max;
  out_view4[2] = plot->y_min;
  out_view4[3] = plot->y_max;
  return 0;
}

std::int32_t nbp_autoscale(std::uint32_t handle) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  autoscale(*plot);
  plot->axis_view_min[0] = static_cast<double>(plot->x_min);
  plot->axis_view_max[0] = static_cast<double>(plot->x_max);
  plot->axis_view_min[3] = static_cast<double>(plot->y_min);
  plot->axis_view_max[3] = static_cast<double>(plot->y_max);
  plot->implot_force_view = true;
  return 0;
}

std::uint32_t nbp_build_draw_data(std::uint32_t handle, std::uint32_t pixel_width) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return 0;
  }

  using Clock = std::chrono::steady_clock;
  auto duration_to_ms = [](Clock::duration duration) -> float {
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
               duration)
        .count();
  };

  float lod_ms = 0.0f;
  float segment_build_ms = 0.0f;

  const std::uint32_t lod_width = std::max<std::uint32_t>(1, pixel_width);
  plot->last_pixel_width = lod_width;
  plot->draw_tuples.clear();
  plot->draw_segments.clear();
  plot->series_views.clear();
  plot->draw_segments.reserve(plot->order.size() * 2U);
  plot->series_views.reserve(plot->order.size());

  if (!plot->view_initialized) {
    autoscale(*plot);
  }

  for (const std::uint32_t series_id : plot->order) {
    auto it = plot->series_by_id.find(series_id);
    if (it == plot->series_by_id.end()) {
      continue;
    }
    Series &series = it->second;
    if (!series.visible || series.raw.empty()) {
      continue;
    }

    nbimplot::SeriesView series_view;
    series_view.slot = static_cast<std::int32_t>(series.slot);
    series_view.subplot_index = std::max(0, series.subplot_index);
    series_view.x_axis = clamp_value<std::int32_t>(series.x_axis, 0, 2);
    series_view.y_axis = clamp_value<std::int32_t>(series.y_axis, 3, 5);
    series_view.has_custom_color = series.has_custom_color ? 1 : 0;
    series_view.color_r = series.color_r;
    series_view.color_g = series.color_g;
    series_view.color_b = series.color_b;
    series_view.color_a = series.color_a;
    series_view.line_weight = series.line_weight;
    series_view.marker = series.marker;
    series_view.marker_size = series.marker_size;
    series_view.label =
        (series.name.empty() ? "series" : series.name.c_str());
    plot->series_views.push_back(series_view);

    const std::int32_t max_idx = static_cast<std::int32_t>(series.raw.size() - 1);
    const std::int32_t start = clamp_value(
        static_cast<std::int32_t>(std::floor(plot->x_min)), 0, max_idx);
    const std::int32_t end =
        clamp_value(static_cast<std::int32_t>(std::ceil(plot->x_max)), 0, max_idx);
    if (end < start) {
      continue;
    }

    const std::int32_t visible_points = end - start + 1;
    if (visible_points <= static_cast<std::int32_t>(kRawSwitchFactor * lod_width)) {
      const auto segment_start = Clock::now();
      bool pen_down = false;
      std::int32_t seg_start = -1;
      std::uint32_t seg_count = 0;
      auto flush_segment = [&]() {
        if (seg_start >= 0 && seg_count >= 2) {
          nbimplot::DrawSegmentView seg;
          seg.slot = static_cast<std::int32_t>(series.slot);
          seg.start = static_cast<std::uint32_t>(seg_start);
          seg.count = seg_count;
          plot->draw_segments.push_back(seg);
        }
        seg_start = -1;
        seg_count = 0;
      };
      for (std::int32_t i = start; i <= end; ++i) {
        const float y = series.raw[static_cast<std::size_t>(i)];
        if (!std::isfinite(y)) {
          pen_down = false;
          flush_segment();
          continue;
        }
        const std::uint32_t tuple_idx =
            static_cast<std::uint32_t>(plot->draw_tuples.size() / 4U);
        append_draw_point(plot->draw_tuples, series.slot, static_cast<float>(i), y,
                          pen_down);
        if (!pen_down) {
          flush_segment();
          seg_start = static_cast<std::int32_t>(tuple_idx);
          seg_count = 1;
        } else if (seg_start < 0) {
          seg_start = static_cast<std::int32_t>(tuple_idx);
          seg_count = 1;
        } else {
          seg_count += 1;
        }
        pen_down = true;
      }
      flush_segment();
      segment_build_ms += duration_to_ms(Clock::now() - segment_start);
      continue;
    }

    const auto lod_start = Clock::now();
    build_min_max_lod(series, start, end, lod_width);
    lod_ms += duration_to_ms(Clock::now() - lod_start);

    const auto segment_start = Clock::now();
    bool pen_down = false;
    std::int32_t seg_start = -1;
    std::uint32_t seg_count = 0;
    auto flush_segment = [&]() {
      if (seg_start >= 0 && seg_count >= 2) {
        nbimplot::DrawSegmentView seg;
        seg.slot = static_cast<std::int32_t>(series.slot);
        seg.start = static_cast<std::uint32_t>(seg_start);
        seg.count = seg_count;
        plot->draw_segments.push_back(seg);
      }
      seg_start = -1;
      seg_count = 0;
    };
    for (std::size_t i = 0; i + 1 < series.lod_xy.size(); i += 2) {
      const float x = series.lod_xy[i];
      const float y = series.lod_xy[i + 1];
      if (!std::isfinite(x) || !std::isfinite(y)) {
        pen_down = false;
        flush_segment();
        continue;
      }
      const std::uint32_t tuple_idx =
          static_cast<std::uint32_t>(plot->draw_tuples.size() / 4U);
      append_draw_point(plot->draw_tuples, series.slot, x, y, pen_down);
      if (!pen_down) {
        flush_segment();
        seg_start = static_cast<std::int32_t>(tuple_idx);
        seg_count = 1;
      } else if (seg_start < 0) {
        seg_start = static_cast<std::int32_t>(tuple_idx);
        seg_count = 1;
      } else {
        seg_count += 1;
      }
      pen_down = true;
    }
    flush_segment();
    segment_build_ms += duration_to_ms(Clock::now() - segment_start);
  }

  plot->last_lod_ms = lod_ms;
  plot->last_segment_build_ms = segment_build_ms;

  return static_cast<std::uint32_t>(plot->draw_tuples.size() / 4);
}

const float *nbp_get_draw_ptr(std::uint32_t handle) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || plot->draw_tuples.empty()) {
    return nullptr;
  }
  return plot->draw_tuples.data();
}

std::uint32_t nbp_get_draw_len(std::uint32_t handle) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return 0;
  }
  return static_cast<std::uint32_t>(plot->draw_tuples.size() / 4);
}

const float *nbp_get_interaction_ptr(std::uint32_t handle) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || plot->interaction_tuples.empty()) {
    return nullptr;
  }
  return plot->interaction_tuples.data();
}

std::uint32_t nbp_get_interaction_len(std::uint32_t handle) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return 0;
  }
  return static_cast<std::uint32_t>(plot->interaction_tuples.size() / 8U);
}

std::int32_t nbp_set_mouse_pos(std::uint32_t handle, float x, float y,
                               std::int32_t inside) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || !std::isfinite(x) || !std::isfinite(y)) {
    return -1;
  }
  plot->mouse_x = x;
  plot->mouse_y = y;
  plot->mouse_inside = inside != 0;
  return 0;
}

std::int32_t nbp_set_mouse_button(std::uint32_t handle, std::int32_t button,
                                  std::int32_t down) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || button < 0 ||
      button >= static_cast<std::int32_t>(plot->mouse_down.size())) {
    return -1;
  }
  plot->mouse_down[static_cast<std::size_t>(button)] = down != 0;
  return 0;
}

std::int32_t nbp_add_mouse_wheel(std::uint32_t handle, float wheel_x,
                                 float wheel_y) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || !std::isfinite(wheel_x) || !std::isfinite(wheel_y)) {
    return -1;
  }
  plot->wheel_x += wheel_x;
  plot->wheel_y += wheel_y;
  return 0;
}

std::int32_t nbp_render(std::uint32_t handle, const char *title_id) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || !plot->implot_enabled) {
    return -1;
  }

  using Clock = std::chrono::steady_clock;
  auto duration_to_ms = [](Clock::duration duration) -> float {
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
               duration)
        .count();
  };
  const auto frame_start = Clock::now();

  const std::uint32_t pixel_width =
      static_cast<std::uint32_t>(std::max(1.0f, plot->canvas_width * plot->dpr));
  nbp_build_draw_data(handle, pixel_width);

  plot->implot_layer.set_canvas(plot->canvas_width, plot->canvas_height, plot->dpr);
  plot->implot_layer.set_mouse_pos(plot->mouse_x, plot->mouse_y, plot->mouse_inside);
  for (std::size_t i = 0; i < plot->mouse_down.size(); ++i) {
    plot->implot_layer.set_mouse_button(static_cast<std::int32_t>(i),
                                        plot->mouse_down[i]);
  }
  if (plot->wheel_x != 0.0f || plot->wheel_y != 0.0f) {
    plot->implot_layer.add_mouse_wheel(plot->wheel_x, plot->wheel_y);
    plot->wheel_x = 0.0f;
    plot->wheel_y = 0.0f;
  }

  if (plot->primitive_views_dirty) {
    plot->primitive_views.clear();
    plot->primitive_views.reserve(plot->primitive_order.size());
    for (const std::uint32_t prim_id : plot->primitive_order) {
      const auto it = plot->primitives_by_id.find(prim_id);
      if (it == plot->primitives_by_id.end()) {
        continue;
      }
      const Primitive &p = it->second;
      if (!p.visible) {
        continue;
      }
      nbimplot::PrimitiveView view;
      view.kind = p.kind;
      view.id = p.id;
      view.data0 = p.data0.empty() ? nullptr : p.data0.data();
      view.len0 = static_cast<std::uint32_t>(p.data0.size());
      view.data1 = p.data1.empty() ? nullptr : p.data1.data();
      view.len1 = static_cast<std::uint32_t>(p.data1.size());
      view.data2 = p.data2.empty() ? nullptr : p.data2.data();
      view.len2 = static_cast<std::uint32_t>(p.data2.size());
      view.i0 = p.ints[0];
      view.i1 = p.ints[1];
      view.i2 = p.ints[2];
      view.i3 = p.ints[3];
      view.i4 = p.ints[4];
      view.i5 = p.ints[5];
      view.i6 = p.ints[6];
      view.i7 = p.ints[7];
      view.f0 = p.floats[0];
      view.f1 = p.floats[1];
      view.f2 = p.floats[2];
      view.f3 = p.floats[3];
      view.f4 = p.floats[4];
      view.f5 = p.floats[5];
      view.f6 = p.floats[6];
      view.f7 = p.floats[7];
      view.text = p.text.c_str();
      plot->primitive_views.push_back(view);
    }
    plot->primitive_views_dirty = false;
  }

  float selection_state[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::array<const char *, 6> axis_label_ptrs = {nullptr, nullptr, nullptr,
                                                  nullptr, nullptr, nullptr};
  std::array<const char *, 6> axis_format_ptrs = {nullptr, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr};
  std::array<const double *, 6> axis_tick_value_ptrs = {nullptr, nullptr, nullptr,
                                                         nullptr, nullptr, nullptr};
  std::array<std::int32_t, 6> axis_tick_counts = {0, 0, 0, 0, 0, 0};
  std::array<const char *, 6> axis_tick_label_ptrs = {nullptr, nullptr, nullptr,
                                                       nullptr, nullptr, nullptr};
  std::array<std::int32_t, 6> axis_tick_keep_default = {0, 0, 0, 0, 0, 0};
  for (std::size_t axis = 0; axis < 6U; ++axis) {
    if (!plot->axis_labels[axis].empty()) {
      axis_label_ptrs[axis] = plot->axis_labels[axis].c_str();
    }
    if (!plot->axis_formats[axis].empty()) {
      axis_format_ptrs[axis] = plot->axis_formats[axis].c_str();
    }
    const AxisTicksConfig &ticks = plot->axis_ticks[axis];
    if (!ticks.values.empty()) {
      axis_tick_value_ptrs[axis] = ticks.values.data();
      axis_tick_counts[axis] = static_cast<std::int32_t>(ticks.values.size());
      axis_tick_keep_default[axis] = ticks.keep_default;
      if (!ticks.labels_blob.empty()) {
        axis_tick_label_ptrs[axis] = ticks.labels_blob.c_str();
      }
    }
  }
  const auto render_start = Clock::now();
  const bool ok = plot->implot_layer.render(
      plot->draw_tuples.empty() ? nullptr : plot->draw_tuples.data(),
      static_cast<std::uint32_t>(plot->draw_tuples.size() / 4),
      plot->draw_segments.empty() ? nullptr : plot->draw_segments.data(),
      static_cast<std::uint32_t>(plot->draw_segments.size()),
      plot->series_views.empty() ? nullptr : plot->series_views.data(),
      static_cast<std::uint32_t>(plot->series_views.size()), &plot->x_min,
      &plot->x_max, &plot->y_min, &plot->y_max, title_id, plot->implot_force_view,
      plot->plot_flags, plot->axis_enabled.data(), plot->axis_scales.data(),
      axis_label_ptrs.data(), axis_format_ptrs.data(),
      plot->axis_limits_constraints_enabled.data(),
      plot->axis_limits_constraints_min.data(),
      plot->axis_limits_constraints_max.data(),
      plot->axis_zoom_constraints_enabled.data(),
      plot->axis_zoom_constraints_min.data(),
      plot->axis_zoom_constraints_max.data(), plot->axis_links.data(),
      plot->axis_view_min.data(), plot->axis_view_max.data(),
      axis_tick_value_ptrs.data(), axis_tick_counts.data(),
      axis_tick_label_ptrs.data(), axis_tick_keep_default.data(),
      plot->subplot_rows, plot->subplot_cols, plot->subplot_flags,
      plot->aligned_group_enabled,
      plot->aligned_group_id.empty() ? nullptr : plot->aligned_group_id.c_str(),
      plot->aligned_group_vertical,
      plot->colormap_name.empty() ? nullptr : plot->colormap_name.c_str(),
      plot->primitive_views.empty() ? nullptr : plot->primitive_views.data(),
      selection_state,
      static_cast<std::uint32_t>(plot->primitive_views.size()));
  plot->last_render_ms = duration_to_ms(Clock::now() - render_start);
  plot->last_frame_ms = duration_to_ms(Clock::now() - frame_start);
  plot->interaction_tuples.clear();
  auto append_interaction = [&](std::int32_t kind, std::uint32_t id,
                                std::int32_t subplot_index, std::int32_t active,
                                float v0, float v1, float v2, float v3) {
    plot->interaction_tuples.push_back(static_cast<float>(kind));
    plot->interaction_tuples.push_back(static_cast<float>(id));
    plot->interaction_tuples.push_back(static_cast<float>(subplot_index));
    plot->interaction_tuples.push_back(static_cast<float>(active));
    plot->interaction_tuples.push_back(v0);
    plot->interaction_tuples.push_back(v1);
    plot->interaction_tuples.push_back(v2);
    plot->interaction_tuples.push_back(v3);
  };

  for (nbimplot::PrimitiveView &view : plot->primitive_views) {
    const auto prim_it = plot->primitives_by_id.find(view.id);
    if (prim_it == plot->primitives_by_id.end()) {
      continue;
    }
    Primitive &prim = prim_it->second;
    if (view.kind == kPrimDragLineX) {
      prim.floats[4] = view.f4;
      prim.ints[6] = view.i6;
      if (view.i6 != 0) {
        append_interaction(kPrimDragLineX, view.id, view.i7, view.i6, view.f4, 0.0f,
                           0.0f, 0.0f);
      }
    } else if (view.kind == kPrimDragLineY) {
      prim.floats[5] = view.f5;
      prim.ints[6] = view.i6;
      if (view.i6 != 0) {
        append_interaction(kPrimDragLineY, view.id, view.i7, view.i6, 0.0f, view.f5,
                           0.0f, 0.0f);
      }
    } else if (view.kind == kPrimDragPoint) {
      prim.floats[4] = view.f4;
      prim.floats[5] = view.f5;
      prim.ints[6] = view.i6;
      if (view.i6 != 0) {
        append_interaction(kPrimDragPoint, view.id, view.i7, view.i6, view.f4, view.f5,
                           0.0f, 0.0f);
      }
    } else if (view.kind == kPrimDragRect) {
      prim.floats[4] = view.f4;
      prim.floats[5] = view.f5;
      prim.floats[6] = view.f6;
      prim.floats[7] = view.f7;
      prim.ints[6] = view.i6;
      if (view.i6 != 0) {
        append_interaction(kPrimDragRect, view.id, view.i7, view.i6, view.f4, view.f5,
                           view.f6, view.f7);
      }
    } else if (view.kind == kPrimColormapSlider) {
      prim.floats[4] = view.f4;
      prim.ints[6] = view.i6;
      if (view.i6 != 0) {
        append_interaction(kPrimColormapSlider, view.id, view.i7, view.i6, view.f4,
                           0.0f, 0.0f, 0.0f);
      }
    } else if (view.kind == kPrimColormapButton) {
      prim.ints[6] = view.i6;
      if (view.i6 != 0) {
        append_interaction(kPrimColormapButton, view.id, view.i7, view.i6, 0.0f,
                           0.0f, 0.0f, 0.0f);
      }
    } else if (view.kind == kPrimColormapSelector) {
      prim.floats[4] = view.f4;
      prim.ints[6] = view.i6;
      if (view.i6 != 0) {
        append_interaction(kPrimColormapSelector, view.id, view.i7, view.i6,
                           view.f4, 0.0f, 0.0f, 0.0f);
      }
    } else if (view.kind == kPrimDragDropPlot || view.kind == kPrimDragDropAxis ||
               view.kind == kPrimDragDropLegend) {
      prim.floats[4] = view.f4;
      prim.floats[5] = view.f5;
      prim.ints[6] = view.i6;
      if (view.i6 != 0) {
        append_interaction(view.kind, view.id, view.i7, view.i6, view.f4, view.f5,
                           view.f6, view.f7);
      }
    }
  }

  if (selection_state[0] > 0.5f) {
    append_interaction(100, 0, static_cast<std::int32_t>(selection_state[1]), 1,
                       selection_state[2], selection_state[3], selection_state[4],
                       selection_state[5]);
  }
  if (ok) {
    plot->view_initialized = true;
    plot->implot_force_view = false;
    plot->axis_view_min[0] = static_cast<double>(plot->x_min);
    plot->axis_view_max[0] = static_cast<double>(plot->x_max);
    plot->axis_view_min[3] = static_cast<double>(plot->y_min);
    plot->axis_view_max[3] = static_cast<double>(plot->y_max);
  }
  return ok ? 0 : -1;
}

std::int32_t nbp_get_perf_stats(std::uint32_t handle, float *out_stats8) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr || out_stats8 == nullptr) {
    return -1;
  }
  out_stats8[0] = plot->last_lod_ms;
  out_stats8[1] = plot->last_segment_build_ms;
  out_stats8[2] = plot->last_render_ms;
  out_stats8[3] = plot->last_frame_ms;
  out_stats8[4] = static_cast<float>(plot->draw_tuples.size() / 4U);
  out_stats8[5] = static_cast<float>(plot->draw_segments.size());
  out_stats8[6] = static_cast<float>(plot->primitive_views.size());
  out_stats8[7] = static_cast<float>(plot->last_pixel_width);
  return 0;
}

std::int32_t nbp_is_implot_compiled() {
#if NBIMPLOT_WITH_IMPLOT
  return 1;
#else
  return 0;
#endif
}

std::int32_t nbp_set_implot_enabled(std::uint32_t handle, std::int32_t enabled) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return -1;
  }
  if (enabled == 0) {
    if (plot->implot_enabled) {
      plot->implot_layer.shutdown();
    }
    plot->implot_enabled = false;
    return 0;
  }

  if (plot->implot_enabled) {
    return 0;
  }

  if (!plot->implot_layer.is_compiled()) {
    plot->implot_enabled = false;
    return 0;
  }

  if (plot->canvas_selector.empty()) {
    plot->implot_enabled = false;
    return -1;
  }

  const bool ok = plot->implot_layer.initialize(plot->canvas_selector.c_str());
  if (ok) {
    plot->implot_layer.set_canvas(plot->canvas_width, plot->canvas_height, plot->dpr);
    plot->implot_force_view = true;
  }
  plot->implot_enabled = ok;
  return ok ? 0 : -1;
}

std::int32_t nbp_is_implot_enabled(std::uint32_t handle) {
  PlotCore *plot = get_plot(handle);
  if (plot == nullptr) {
    return 0;
  }
  return plot->implot_enabled ? 1 : 0;
}

} // extern "C"
