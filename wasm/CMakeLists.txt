cmake_minimum_required(VERSION 3.20)
project(nbimplot_wasm LANGUAGES CXX)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Use emcmake to configure this project for WebAssembly.")
endif()

option(NBIMPLOT_WITH_IMPLOT "Build with ImPlot/ImGui sources" OFF)
set(NBIMPLOT_IMGUI_DIR "" CACHE PATH "Path to Dear ImGui repository root")
set(NBIMPLOT_IMPLOT_DIR "" CACHE PATH "Path to ImPlot repository root")

set(CMAKE_EXECUTABLE_SUFFIX ".js")

add_executable(
  nbimplot_wasm
  core/nbimplot_core.cpp
  core/nbimplot_implot_layer.cpp
)

target_compile_features(nbimplot_wasm PRIVATE cxx_std_20)
target_compile_options(nbimplot_wasm PRIVATE -O3 -fno-exceptions -fno-rtti)

if(NBIMPLOT_WITH_IMPLOT)
  if(NOT EXISTS "${NBIMPLOT_IMGUI_DIR}/imgui.h")
    message(FATAL_ERROR "NBIMPLOT_IMGUI_DIR must point to Dear ImGui sources.")
  endif()
  if(NOT EXISTS "${NBIMPLOT_IMPLOT_DIR}/implot.h")
    message(FATAL_ERROR "NBIMPLOT_IMPLOT_DIR must point to ImPlot sources.")
  endif()

  target_include_directories(
    nbimplot_wasm
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/core"
      "${NBIMPLOT_IMGUI_DIR}"
      "${NBIMPLOT_IMGUI_DIR}/backends"
      "${NBIMPLOT_IMPLOT_DIR}"
  )
  target_compile_definitions(
    nbimplot_wasm
    PRIVATE
      IMGUI_USER_CONFIG=\"nbimplot_imgui_config.h\"
      IMGUI_IMPL_OPENGL_ES3
      NBIMPLOT_WITH_IMPLOT=1
  )
  target_sources(
    nbimplot_wasm
    PRIVATE
      "${NBIMPLOT_IMGUI_DIR}/imgui.cpp"
      "${NBIMPLOT_IMGUI_DIR}/imgui_draw.cpp"
      "${NBIMPLOT_IMGUI_DIR}/imgui_tables.cpp"
      "${NBIMPLOT_IMGUI_DIR}/imgui_widgets.cpp"
      "${NBIMPLOT_IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
      "${NBIMPLOT_IMPLOT_DIR}/implot.cpp"
      "${NBIMPLOT_IMPLOT_DIR}/implot_items.cpp"
  )
else()
  target_compile_definitions(nbimplot_wasm PRIVATE NBIMPLOT_WITH_IMPLOT=0)
endif()

target_link_options(
  nbimplot_wasm
  PRIVATE
    -O3
    "-sWASM=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sASSERTIONS=0"
    "-sMIN_WEBGL_VERSION=2"
    "-sMAX_WEBGL_VERSION=2"
    "-sFULL_ES3=1"
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sENVIRONMENT=web"
    "-sFILESYSTEM=0"
    "-sEXPORT_NAME=createNbImPlotModule"
    "-sEXPORTED_FUNCTIONS=['_malloc','_free','_nbp_create','_nbp_destroy','_nbp_set_canvas','_nbp_set_canvas_selector','_nbp_line_set_data','_nbp_line_append_data','_nbp_line_set_name','_nbp_set_series_subplot','_nbp_set_series_axes','_nbp_set_series_style','_nbp_set_plot_options','_nbp_set_axis_state','_nbp_set_axis_label','_nbp_set_axis_format','_nbp_set_axis_ticks','_nbp_clear_axis_ticks','_nbp_set_axis_limits_constraints','_nbp_set_axis_zoom_constraints','_nbp_set_axis_link','_nbp_set_subplots','_nbp_set_aligned_group','_nbp_set_colormap','_nbp_primitive_set_data','_nbp_primitive_remove','_nbp_primitive_set_visible','_nbp_set_series_visible','_nbp_set_view','_nbp_get_view','_nbp_autoscale','_nbp_build_draw_data','_nbp_get_draw_ptr','_nbp_get_draw_len','_nbp_get_interaction_ptr','_nbp_get_interaction_len','_nbp_set_mouse_pos','_nbp_set_mouse_button','_nbp_add_mouse_wheel','_nbp_render','_nbp_get_perf_stats','_nbp_is_implot_compiled','_nbp_set_implot_enabled','_nbp_is_implot_enabled']"
)

set_target_properties(
  nbimplot_wasm
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../nbimplot/wasm"
    OUTPUT_NAME "nbimplot_wasm"
)
