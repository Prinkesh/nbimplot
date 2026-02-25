#pragma once

#include <cstdint>

namespace nbimplot {

struct DrawSegmentView {
  std::int32_t slot = 0;
  std::uint32_t start = 0;
  std::uint32_t count = 0;
};

struct SeriesView {
  std::int32_t slot = 0;
  std::int32_t subplot_index = 0;
  std::int32_t x_axis = 0;
  std::int32_t y_axis = 3;
  std::int32_t has_custom_color = 0;
  float color_r = 0.0f;
  float color_g = 0.0f;
  float color_b = 0.0f;
  float color_a = 0.0f;
  float line_weight = 1.0f;
  std::int32_t marker = -2;
  float marker_size = 4.0f;
  const char *label = nullptr;
};

struct PrimitiveView {
  std::int32_t kind = 0;
  std::uint32_t id = 0;
  const float *data0 = nullptr;
  std::uint32_t len0 = 0;
  const float *data1 = nullptr;
  std::uint32_t len1 = 0;
  const float *data2 = nullptr;
  std::uint32_t len2 = 0;
  std::int32_t i0 = 0;
  std::int32_t i1 = 0;
  std::int32_t i2 = 0;
  std::int32_t i3 = 0;
  std::int32_t i4 = 0;
  std::int32_t i5 = 0;
  std::int32_t i6 = 0;
  std::int32_t i7 = 0;
  float f0 = 0.0f;
  float f1 = 0.0f;
  float f2 = 0.0f;
  float f3 = 0.0f;
  float f4 = 0.0f;
  float f5 = 0.0f;
  float f6 = 0.0f;
  float f7 = 0.0f;
  const char *text = nullptr;
};

class ImPlotLayer {
public:
  bool is_compiled() const noexcept;
  bool initialize(const char *canvas_selector) noexcept;
  void shutdown() noexcept;
  void set_canvas(std::int32_t width, std::int32_t height, float dpr) noexcept;
  void set_mouse_pos(float x, float y, bool inside) noexcept;
  void set_mouse_button(std::int32_t button, bool down) noexcept;
  void add_mouse_wheel(float wheel_x, float wheel_y) noexcept;
  bool render(const float *draw_points, std::uint32_t point_count,
              const DrawSegmentView *segments, std::uint32_t segment_count,
              const SeriesView *series_views, std::uint32_t series_count,
              float *x_min, float *x_max, float *y_min, float *y_max,
              const char *title_id, bool force_view,
              std::int32_t plot_flags, const std::int32_t *axis_enabled6,
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
              std::int32_t subplot_flags, std::int32_t aligned_group_enabled,
              const char *aligned_group_id, std::int32_t aligned_group_vertical,
              const char *colormap_name, PrimitiveView *primitives,
              float *selection_out6,
              std::uint32_t primitive_count) noexcept;
};

} // namespace nbimplot
