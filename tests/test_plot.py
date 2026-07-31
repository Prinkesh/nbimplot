import numpy as np
import pytest
from unittest.mock import patch

import nbimplot as ip


def _capture_messages(plot: ip.Plot):
    sent = []

    def _send(content, buffers=None):
        sent.append((content, buffers))

    plot.send = _send  # type: ignore[assignment]
    return sent


class _Frame:
    def __init__(self, **columns):
        self._columns = columns
        self.columns = list(columns)

    def __getitem__(self, key):
        return self._columns[key]


def test_line_sends_binary_float32_payload():
    plot = ip.Plot(width=640, height=320, title="t")
    sent = _capture_messages(plot)

    handle = plot.line("mid", np.arange(5, dtype=np.float64))
    assert handle.series_id.startswith("s")

    content, buffers = sent[-1]
    assert content["type"] == "line"
    assert content["name"] == "mid"
    assert content["subplot_index"] == 0
    assert content["dtype"] == "float32"
    assert content["length"] == 5

    payload = np.frombuffer(buffers[0], dtype=np.float32)
    np.testing.assert_allclose(payload, np.arange(5, dtype=np.float32))

    sent.clear()
    plot.line("kw", y=np.arange(4, dtype=np.float32))
    content, buffers = sent[-1]
    assert content["name"] == "kw"
    np.testing.assert_allclose(np.frombuffer(buffers[0], dtype=np.float32), np.arange(4, dtype=np.float32))


def test_line_accepts_explicit_x_and_preserves_it_on_same_length_update():
    plot = ip.Plot(width=640, height=320, title="x")
    sent = _capture_messages(plot)
    x = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 1.5, 0.0], dtype=np.float32)

    handle = plot.line("custom-x", y, x=x)
    content, buffers = sent[-1]
    assert content["type"] == "line"
    assert content["has_x"] is True
    assert len(buffers) == 2
    np.testing.assert_allclose(np.frombuffer(buffers[0], dtype=np.float32), x)
    np.testing.assert_allclose(np.frombuffer(buffers[1], dtype=np.float32), y)

    sent.clear()
    y2 = (y * 2).astype(np.float32)
    handle.set_data(y2)
    content, buffers = sent[-1]
    assert content["type"] == "set_data"
    assert content["has_x"] is True
    assert content["reuse_allocation"] is True
    np.testing.assert_allclose(np.frombuffer(buffers[0], dtype=np.float32), x)
    np.testing.assert_allclose(np.frombuffer(buffers[1], dtype=np.float32), y2)


def test_line_explicit_x_validation_and_append_rules():
    plot = ip.Plot()
    _capture_messages(plot)
    y = np.arange(4, dtype=np.float32)

    with pytest.raises(ValueError, match="same length"):
        plot.line("bad", y, x=np.arange(3, dtype=np.float32))
    with pytest.raises(ValueError, match="sorted"):
        plot.line("bad", y, x=np.array([0, 2, 1, 3], dtype=np.float32))

    handle = plot.line("custom-x", y, x=np.arange(4, dtype=np.float32))
    with pytest.raises(ValueError, match="resizing"):
        handle.set_data(np.arange(5, dtype=np.float32))
    with pytest.raises(ValueError, match="x must be provided"):
        handle.append(np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError, match="non-decreasing"):
        handle.append(np.ones(2, dtype=np.float32), x=np.array([2, 3], dtype=np.float32))

    handle.append(np.array([4, 5], dtype=np.float32), x=np.array([4, 5], dtype=np.float32))


def test_set_data_reports_reuse_flag():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    handle = plot.line("mid", np.ones(4, dtype=np.float32))
    sent.clear()

    handle.set_data(np.zeros(4, dtype=np.float32))
    content, _ = sent[-1]
    assert content["type"] == "set_data"
    assert content["reuse_allocation"] is True

    handle.set_data(np.zeros(7, dtype=np.float32))
    content, _ = sent[-1]
    assert content["reuse_allocation"] is False


def test_line_style_payload_and_handle_updates():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    handle = plot.line(
        "mid",
        np.arange(5, dtype=np.float32),
        color="#3b82f680",
        line_weight=2.5,
        marker="circle",
        marker_size=6.0,
    )
    content, _ = sent[-1]
    assert content["type"] == "line"
    assert content["has_color"] is True
    assert content["line_weight"] == 2.5
    assert content["marker"] == 0
    assert content["marker_size"] == 6.0

    sent.clear()
    handle.set_style(color=(1.0, 0.0, 0.0, 0.5), line_weight=1.25, marker="diamond", marker_size=5.0)
    content, _ = sent[-1]
    assert content["type"] == "series_style"
    assert content["has_color"] is True
    assert content["line_weight"] == 1.25
    assert content["marker"] == 2
    assert content["marker_size"] == 5.0

    handle.set_style(color=None)
    content, _ = sent[-1]
    assert content["type"] == "series_style"
    assert content["has_color"] is False

    with pytest.raises(ValueError):
        handle.set_style(marker="unknown")


def test_line_requires_non_empty_numeric_1d():
    plot = ip.Plot()
    _capture_messages(plot)

    with pytest.raises(ValueError):
        plot.line("bad", np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError):
        plot.line("bad", [])

    with pytest.raises(TypeError):
        plot.line("bad", np.array(["a", "b"]))


def test_line_accepts_dataframe_columns_without_new_api():
    plot = ip.Plot(width=640, height=320, title="df")
    sent = _capture_messages(plot)
    frame = _Frame(
        t=np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float64),
        mid=np.array([1.0, 2.0, 1.5, 0.0], dtype=np.float64),
        next_mid=np.array([1.5, 2.5, 2.0, 0.5], dtype=np.float64),
    )

    handle = plot.line("mid", frame, x="t", y="mid")
    content, buffers = sent[-1]
    assert content["type"] == "line"
    assert content["has_x"] is True
    np.testing.assert_allclose(np.frombuffer(buffers[0], dtype=np.float32), frame["t"])
    np.testing.assert_allclose(np.frombuffer(buffers[1], dtype=np.float32), frame["mid"])

    sent.clear()
    handle.set_data(frame, x="t", y="next_mid")
    content, buffers = sent[-1]
    assert content["type"] == "set_data"
    assert content["reuse_allocation"] is True
    np.testing.assert_allclose(np.frombuffer(buffers[0], dtype=np.float32), frame["t"])
    np.testing.assert_allclose(np.frombuffer(buffers[1], dtype=np.float32), frame["next_mid"])


def test_dataframe_column_errors_are_explicit():
    plot = ip.Plot()
    _capture_messages(plot)
    frame = _Frame(x=np.arange(3, dtype=np.float32), y=np.arange(3, dtype=np.float32))

    with pytest.raises(ValueError, match="y column"):
        plot.line("bad", frame, x="x")
    with pytest.raises(KeyError, match="Unknown y column"):
        plot.line("bad", frame, x="x", y="missing")
    with pytest.raises(KeyError, match="Unknown x column"):
        plot.line("bad", frame, x="missing", y="y")


def test_plot_replays_series_after_frontend_ready():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.line("mid", np.arange(3, dtype=np.float32))
    sent.clear()

    plot._handle_frontend_message(plot, {"type": "frontend_ready"}, None)
    replayed = [content for content, _ in sent if content.get("type") == "line"]
    assert replayed
    content = replayed[0]
    assert content["type"] == "line"
    assert content["name"] == "mid"
    assert any(content.get("type") in {"wasm_assets", "wasm_assets_missing"} for content, _ in sent)


def test_plot_requires_implot_mode():
    with pytest.raises(ValueError):
        ip.Plot(prefer_implot=False)
    with pytest.raises(ValueError):
        ip.Plot(strict_wasm=False)


def test_scatter_sends_primitive_message():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.scatter("pts", np.array([1, 2, 3], dtype=np.float32))
    content, buffers = sent[-1]
    assert content["type"] == "primitive_add"
    assert content["kind"] == "scatter"
    assert content["name"] == "pts"
    assert content["length"] == 3
    payload = np.frombuffer(buffers[0], dtype=np.float32)
    np.testing.assert_allclose(payload, np.array([1, 2, 3], dtype=np.float32))


def test_plot_replays_primitives_after_frontend_ready():
    plot = ip.Plot()
    sent = _capture_messages(plot)
    plot.histogram("h", np.array([1, 2, 2, 3], dtype=np.float32), bins=2)
    sent.clear()

    plot._handle_frontend_message(plot, {"type": "frontend_ready"}, None)
    replayed = [content for content, _ in sent if content.get("type") == "primitive_add"]
    assert replayed
    content = replayed[0]
    assert content["type"] == "primitive_add"
    assert content["kind"] == "histogram"


def test_subplots_grid_access():
    sub = ip.Subplots(2, 2)
    p = sub.subplot(1, 1)
    assert hasattr(p, "line")
    with pytest.raises(IndexError):
        sub.subplot(2, 2)


def test_additional_primitives_send_payloads():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.bubbles("b", np.array([1, 2], dtype=np.float32), np.array([0.2, 0.4], dtype=np.float32))
    content, buffers = sent[-1]
    assert content["kind"] == "bubbles"
    assert len(buffers) == 2

    plot.histogram2d(
        "h2d",
        np.array([0, 0.4, 0.9, 1.0], dtype=np.float32),
        np.array([0, 0.3, 0.8, 1.0], dtype=np.float32),
        x_bins=2,
        y_bins=2,
    )
    content, buffers = sent[-1]
    assert content["kind"] == "histogram2d"
    assert content["rows"] == 2
    assert content["cols"] == 2
    assert len(buffers) == 3

    plot.dummy("legend-only")
    content, buffers = sent[-1]
    assert content["kind"] == "dummy"
    assert len(buffers) == 0


def test_dataframe_columns_work_for_common_primitives():
    plot = ip.Plot()
    sent = _capture_messages(plot)
    frame = _Frame(
        x=np.array([0, 1, 2, 3], dtype=np.float32),
        y=np.array([1.0, 2.0, 1.5, 2.5], dtype=np.float32),
        y2=np.array([1.5, 2.5, 2.0, 3.0], dtype=np.float32),
        size=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
        err=np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float32),
    )

    plot.scatter("scatter", frame, x="x", y="y", size=3.0)
    content, buffers = sent[-1]
    assert content["kind"] == "scatter"
    np.testing.assert_allclose(np.frombuffer(buffers[0], dtype=np.float32), frame["x"])
    np.testing.assert_allclose(np.frombuffer(buffers[1], dtype=np.float32), frame["y"])

    plot.bubbles("bubbles", frame, sizes="size", x="x", y="y")
    content, buffers = sent[-1]
    assert content["kind"] == "bubbles"
    assert len(buffers) == 3

    plot.shaded("band", frame, y1="y", y2="y2", x="x")
    content, buffers = sent[-1]
    assert content["kind"] == "shaded"
    assert len(buffers) == 3

    plot.error_bars("err", frame, y="y", err="err", x="x")
    content, buffers = sent[-1]
    assert content["kind"] == "error_bars"
    assert len(buffers) == 3

    plot.bars("bars", frame, x="x", y="y")
    content, buffers = sent[-1]
    assert content["kind"] == "bars"
    assert len(buffers) == 2

    plot.bars_h("bars_h", frame, x="y", y="x")
    content, buffers = sent[-1]
    assert content["kind"] == "bars_h"
    assert len(buffers) == 2

    plot.histogram("hist", frame, y="y", bins=2)
    content, buffers = sent[-1]
    assert content["kind"] == "histogram"
    assert len(buffers) == 2

    plot.histogram2d("hist2d", frame, x="x", y="y", x_bins=2, y_bins=2)
    content, buffers = sent[-1]
    assert content["kind"] == "histogram2d"
    assert len(buffers) == 3


def test_heatmap_and_histogram2d_format_options():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    with pytest.raises(ValueError):
        plot.heatmap("bad", np.ones((2, 2), dtype=np.float32), scale_min=float("nan"))

    plot.heatmap(
        "hm",
        np.arange(12, dtype=np.float32).reshape(3, 4),
        label_fmt="%.3e",
        scale_min=-1.0,
        scale_max=2.0,
        show_colorbar=True,
        colorbar_label="Intensity",
        colorbar_format="%.2f",
        colorbar_flags=3,
    )
    content, _ = sent[-1]
    assert content["kind"] == "heatmap"
    assert content["label_fmt"] == "%.3e"
    assert content["scale_min"] == -1.0
    assert content["scale_max"] == 2.0
    assert content["show_colorbar"] is True
    assert content["colorbar_label"] == "Intensity"
    assert content["colorbar_format"] == "%.2f"
    assert content["colorbar_flags"] == 3

    plot.histogram2d(
        "h2d",
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
        x_bins=2,
        y_bins=2,
        label_fmt="%.1f",
        show_colorbar=True,
        colorbar_label="Counts",
        colorbar_format="%g",
        colorbar_flags=1,
    )
    content, _ = sent[-1]
    assert content["kind"] == "histogram2d"
    assert content["label_fmt"] == "%.1f"
    assert content["show_colorbar"] is True
    assert content["colorbar_label"] == "Counts"
    assert content["colorbar_format"] == "%g"
    assert content["colorbar_flags"] == 1

    plot.heatmap("hm_no_text", np.ones((2, 2), dtype=np.float32), label_fmt="")
    content, _ = sent[-1]
    assert content["kind"] == "heatmap"
    assert content["label_fmt"] == ""

    plot.histogram2d(
        "h2d_no_text",
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
        x_bins=2,
        y_bins=2,
        label_fmt="",
    )
    content, _ = sent[-1]
    assert content["kind"] == "histogram2d"
    assert content["label_fmt"] == ""

    with pytest.raises(ValueError):
        plot.heatmap("hm_bad_cbar", np.ones((2, 2), dtype=np.float32), colorbar_flags=-1)
    with pytest.raises(ValueError):
        plot.histogram2d(
            "h2d_bad_cbar",
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            colorbar_flags=-1,
        )


def test_image_payload_with_bounds_uv_and_channels():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.image(
        "img",
        np.arange(12, dtype=np.float32).reshape(3, 4),
        bounds=((1.0, 2.0), (5.0, 8.0)),
        uv0=(0.1, 0.2),
        uv1=(0.9, 0.8),
    )
    content, buffers = sent[-1]
    assert content["kind"] == "image"
    assert content["rows"] == 3
    assert content["cols"] == 4
    assert content["channels"] == 1
    assert content["bounds_x_min"] == 1.0
    assert content["bounds_y_min"] == 2.0
    assert content["bounds_x_max"] == 5.0
    assert content["bounds_y_max"] == 8.0
    assert content["uv0_x"] == 0.1
    assert content["uv0_y"] == 0.2
    assert content["uv1_x"] == 0.9
    assert content["uv1_y"] == 0.8
    payload = np.frombuffer(buffers[0], dtype=np.float32)
    assert payload.size == 12

    plot.image("rgb", np.zeros((2, 3, 3), dtype=np.uint8))
    content, buffers = sent[-1]
    assert content["kind"] == "image"
    assert content["rows"] == 2
    assert content["cols"] == 3
    assert content["channels"] == 3
    payload = np.frombuffer(buffers[0], dtype=np.float32)
    assert payload.size == 18

    with pytest.raises(ValueError):
        plot.image("bad", np.zeros((2, 3, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        plot.image("bad_bounds", np.zeros((2, 3), dtype=np.float32), bounds=((0, 0), (0, 1)))


def test_set_colormap_message_and_replay():
    plot = ip.Plot(colormap="Viridis")
    sent = _capture_messages(plot)

    plot.set_colormap("Jet")
    content, _ = sent[-1]
    assert content["type"] == "colormap"
    assert content["name"] == "Jet"

    plot.set_colormap(None)
    content, _ = sent[-1]
    assert content["type"] == "colormap"
    assert content["name"] == ""

    plot.set_colormap("plasma")
    content, _ = sent[-1]
    assert content["type"] == "colormap"
    assert content["name"] == "Plasma"

    sent.clear()
    plot._handle_frontend_message(plot, {"type": "frontend_ready"}, None)
    colormaps = [content for content, _ in sent if content.get("type") == "colormap"]
    assert colormaps
    assert colormaps[-1]["name"] == "Plasma"


def test_bar_groups_and_pie_chart_validation():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    with pytest.raises(ValueError):
        plot.bar_groups(["a"], np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError):
        plot.pie_chart("pie", np.array([1, 2], dtype=np.float32), labels=["one"])

    plot.bar_groups(
        ["a", "b"],
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        group_size=0.8,
        shift=0.1,
    )
    content, _ = sent[-1]
    assert content["kind"] == "bar_groups"
    assert content["item_count"] == 2
    assert content["group_count"] == 3
    assert content["labels"] == ["a", "b"]

    plot.pie_chart(
        "pie",
        np.array([1, 2, 3], dtype=np.float32),
        labels=["x", "y", "z"],
        x=1.0,
        y=2.0,
        radius=3.0,
    )
    content, _ = sent[-1]
    assert content["kind"] == "pie_chart"
    assert content["labels"] == ["x", "y", "z"]
    assert content["x"] == 1.0
    assert content["y"] == 2.0
    assert content["radius"] == 3.0


def test_view_control_messages_and_line_helpers():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.set_view(0, 10, -2, 2)
    content, _ = sent[-1]
    assert content["type"] == "set_view"
    assert content["x_min"] == 0.0
    assert content["x_max"] == 10.0

    with pytest.raises(ValueError):
        plot.set_view(1, 1, -1, 1)

    plot.autoscale()
    content, _ = sent[-1]
    assert content["type"] == "autoscale"

    plot.vlines("vx", np.array([1, 2], dtype=np.float32))
    content, _ = sent[-1]
    assert content["kind"] == "inf_lines"
    assert content["axis"] == "x"

    plot.hlines("hy", np.array([3], dtype=np.float32))
    content, _ = sent[-1]
    assert content["kind"] == "inf_lines"
    assert content["axis"] == "y"


def test_perf_stats_callback_and_frontend_toggle_message():
    plot = ip.Plot()
    sent = _capture_messages(plot)
    received = []

    def _on_perf(_: ip.Plot, stats: dict[str, float]) -> None:
        received.append(stats)

    plot.on_perf_stats(_on_perf, interval_ms=250)
    content, _ = sent[-1]
    assert content["type"] == "set_perf_reporting"
    assert content["enabled"] is True
    assert content["interval_ms"] == 250

    sent.clear()
    plot._handle_frontend_message(
        plot,
        {
            "type": "perf_stats",
            "fps": 42,
            "lod_ms": 0.4,
            "segment_build_ms": 0.3,
            "render_ms": 1.2,
            "frame_ms": 2.0,
            "draw_points": 1234,
            "draw_segments": 12,
            "primitive_count": 3,
            "pixel_width": 900,
        },
        None,
    )
    assert received
    assert received[-1]["fps"] == 42.0
    assert received[-1]["render_ms"] == 1.2

    plot._handle_frontend_message(plot, {"type": "frontend_ready"}, None)
    toggles = [c for c, _ in sent if c.get("type") == "set_perf_reporting"]
    assert toggles
    assert toggles[-1]["enabled"] is True
    assert toggles[-1]["interval_ms"] == 250


def test_export_png_sends_frontend_action_message():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.export_png("signal.png")
    content, buffers = sent[-1]
    assert content == {"type": "export_png", "filename": "signal.png"}
    assert buffers is None

    plot.export_png("")
    content, _ = sent[-1]
    assert content["filename"] == "nbimplot.png"

    sub = ip.Subplots(1, 1)
    sent = _capture_messages(sub._plot)
    sub.export_png("subplots.png")
    content, _ = sent[-1]
    assert content == {"type": "export_png", "filename": "subplots.png"}


def test_subplots_use_native_config_and_subplot_routing():
    sub = ip.Subplots(1, 2, link_all_x=True, no_menus=True)
    root_plot = sub._plot
    sent = _capture_messages(root_plot)

    sub.subplot(0, 1).line("rhs", np.array([1, 2, 3], dtype=np.float32))
    content, _ = sent[-1]
    assert content["type"] == "line"
    assert content["subplot_index"] == 1

    sent.clear()
    root_plot._handle_frontend_message(root_plot, {"type": "frontend_ready"}, None)
    subplot_cfg = [content for content, _ in sent if content.get("type") == "subplots_config"]
    assert subplot_cfg
    assert subplot_cfg[-1]["rows"] == 1
    assert subplot_cfg[-1]["cols"] == 2
    assert subplot_cfg[-1]["flags"] != 0


def test_subplots_accept_implot_style_options():
    sub = ip.Subplots(
        1,
        1,
        title="My Subplot",
        share_items=True,
        no_legend=True,
        no_menus=True,
        no_resize=True,
        no_align=True,
        col_major=True,
    )
    assert sub.title == "My Subplot"
    assert sub.share_items is True
    assert hasattr(sub.subplot(0, 0), "line")


def test_plot_axis_scale_and_flags_messages():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.set_axis_scale(x="log", y="linear")
    content = [msg for msg, _ in sent if msg.get("type") == "plot_options"][-1]
    assert content["type"] == "plot_options"
    assert content["axis_scale_x"] == 1
    assert content["axis_scale_y"] == 0

    plot.set_plot_flags(no_legend=True, no_menus=True, crosshairs=True)
    content, _ = sent[-1]
    assert content["type"] == "plot_options"
    assert content["plot_flags"] != 0


def test_axis_state_and_time_helpers_send_messages():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.set_axis_state("y2", enabled=True, scale="log")
    content, _ = sent[-1]
    assert content["type"] == "axis_state"
    assert content["axis"] == 4
    assert content["enabled"] is True
    assert content["scale"] == 1

    plot.set_time_axis("x1")
    axis_msg = sent[-2][0]
    plot_msg = sent[-1][0]
    assert axis_msg["type"] == "axis_state"
    assert axis_msg["axis"] == 0
    assert axis_msg["scale"] == 2
    assert plot_msg["type"] == "plot_options"
    assert plot_msg["axis_scale_x"] == 2


def test_drag_tools_and_interaction_callbacks():
    plot = ip.Plot()
    sent = _capture_messages(plot)
    tool_events = []
    selection_events = []

    plot.on_tool_change(lambda _p, payload: tool_events.append(payload))
    plot.on_selection_change(lambda _p, payload: selection_events.append(payload))

    plot.drag_line_x("lx", 2.5)
    content, _ = sent[-1]
    assert content["type"] == "primitive_add"
    assert content["kind"] == "drag_line_x"
    assert content["value"] == 2.5

    plot.drag_point("pt", 1.0, -1.0)
    content, _ = sent[-1]
    assert content["kind"] == "drag_point"
    assert content["x"] == 1.0
    assert content["y"] == -1.0

    plot._handle_frontend_message(
        plot,
        {
            "type": "interaction_update",
            "tools": [{"type": "drag_line_x", "tool_id": 7, "value": 3.0, "subplot_index": 0}],
            "selection": {"subplot_index": 0, "x_min": 1, "x_max": 2, "y_min": 3, "y_max": 4},
        },
        None,
    )
    assert tool_events
    assert tool_events[-1]["type"] == "drag_line_x"
    assert tool_events[-1]["value"] == 3.0
    assert selection_events
    assert selection_events[-1]["x_min"] == 1.0


def test_hover_click_callbacks_and_exact_selection_indices():
    plot = ip.Plot()
    _capture_messages(plot)
    x = np.array([0.0, 0.5, 1.5, 3.0, 4.0], dtype=np.float32)
    y = np.array([1.0, -1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    handle = plot.line("custom", y, x=x)
    plot.line("implicit", np.array([0.0, 2.0, 4.0, 1.0], dtype=np.float32))

    hover_events = []
    click_events = []
    selection_events = []
    plot.on_hover(lambda _p, payload: hover_events.append(payload))
    plot.on_click(lambda _p, payload: click_events.append(payload))
    plot.on_select(lambda _p, payload: selection_events.append(payload))

    selection = {
        "subplot_index": 0,
        "x_min": 0.25,
        "x_max": 3.2,
        "y_min": 0.0,
        "y_max": 2.1,
        "series": [
            {
                "series_id": handle.series_id,
                "series_name": "custom",
                "series_token": 1,
                "subplot_index": 0,
                "index_min": 1,
                "index_max": 3,
                "count": 3,
                "has_x": True,
            }
        ],
    }
    plot._handle_frontend_message(
        plot,
        {
            "type": "interaction_update",
            "tools": [],
            "selection": selection,
            "hover": {
                "series_id": handle.series_id,
                "series_name": "custom",
                "series_token": 1,
                "subplot_index": 0,
                "active": True,
                "x": 1.5,
                "y": 2.0,
                "index": 2,
                "distance_px": 3.5,
            },
            "click": {
                "series_id": handle.series_id,
                "series_name": "custom",
                "series_token": 1,
                "subplot_index": 0,
                "active": True,
                "x": 1.5,
                "y": 2.0,
                "button": 0,
                "index": 2,
            },
        },
        None,
    )

    assert hover_events[-1]["series_name"] == "custom"
    assert hover_events[-1]["index"] == 2
    assert click_events[-1]["button"] == 0
    assert selection_events[-1]["series"][0]["index_min"] == 1

    np.testing.assert_array_equal(
        plot.indices_for_selection(selection_events[-1], handle),
        np.array([2, 3], dtype=np.int64),
    )
    all_indices = plot.selection_indices(selection_events[-1])
    assert set(all_indices) == {handle.series_id, "s2"}
    np.testing.assert_array_equal(all_indices[handle.series_id], np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(all_indices["s2"], np.array([1, 3], dtype=np.int64))


def test_primitives_accept_axis_routing():
    import inspect

    plot = ip.Plot()
    sent = _capture_messages(plot)
    assert "x_axis" in inspect.signature(plot.bars_h).parameters

    plot.bars_h(
        "bh",
        np.array([1, 2], dtype=np.float32),
        x_axis="x2",
        y_axis="y2",
    )
    content, _ = sent[-1]
    assert content["kind"] == "bars_h"
    assert content["x_axis"] == 1
    assert content["y_axis"] == 4

    plot.hlines(
        "hy",
        np.array([3, 4], dtype=np.float32),
        x_axis="x3",
        y_axis="y3",
    )
    content, _ = sent[-1]
    assert content["kind"] == "inf_lines"
    assert content["axis"] == "y"
    assert content["x_axis"] == 2
    assert content["y_axis"] == 5

    plot.text("pt", 1.0, 2.0, x_axis="x1", y_axis="y2")
    content, _ = sent[-1]
    assert content["kind"] == "text"
    assert content["x_axis"] == 0
    assert content["y_axis"] == 4


def test_stream_line_append_and_hidden_next_item():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    handle = plot.stream_line("stream", capacity=3, initial=np.array([1, 2], dtype=np.float32))
    content, _ = sent[-1]
    assert content["type"] == "line"
    assert content["max_points"] == 3

    sent.clear()
    handle.append(np.array([3, 4], dtype=np.float32))
    content, buffers = sent[-1]
    assert content["type"] == "append_data"
    assert content["series_id"] == handle.series_id
    assert content["max_points"] == 3
    assert content["length"] == 3
    payload = np.frombuffer(buffers[0], dtype=np.float32)
    np.testing.assert_allclose(payload, np.array([3, 4], dtype=np.float32))

    sent.clear()
    plot.hide_next_item()
    plot.scatter("hidden_scatter", np.array([1, 2, 3], dtype=np.float32))
    content, _ = sent[-1]
    assert content["type"] == "primitive_add"
    assert content["kind"] == "scatter"
    assert content["hidden"] is True


def test_stream_line_explicit_x_controls_and_state_exports(tmp_path):
    plot = ip.Plot()
    sent = _capture_messages(plot)

    handle = plot.stream_line(
        "ticks",
        capacity=4,
        initial=np.array([1, 2], dtype=np.float32),
        initial_x=np.array([10, 11], dtype=np.float32),
        auto_render=True,
    )
    sent.clear()
    handle.append(np.array([3, 4, 5], dtype=np.float32), x=np.array([12, 13, 14], dtype=np.float32))
    content, buffers = sent[-2]
    assert content["type"] == "set_data"
    assert content["has_x"] is True
    np.testing.assert_allclose(np.frombuffer(buffers[0], dtype=np.float32), np.array([11, 12, 13, 14], dtype=np.float32))
    np.testing.assert_allclose(np.frombuffer(buffers[1], dtype=np.float32), np.array([2, 3, 4, 5], dtype=np.float32))
    assert sent[-1][0]["type"] == "render"

    handle.pause()
    sent.clear()
    handle.append(np.array([6], dtype=np.float32), x=np.array([15], dtype=np.float32))
    assert sent == []
    handle.resume()
    handle.set_window(2)
    assert plot.get_state(include_data=True)["series"][0]["stream_capacity"] == 2
    handle.clear(x0=100.0, y0=-1.0)
    assert plot.get_state(include_data=True)["series"][0]["y"] == [-1.0]

    path = tmp_path / "state.json"
    text = plot.export_json_state(path, include_data=True)
    assert path.read_text(encoding="utf-8") == text
    assert '"series"' in text


def test_selection_highlight_csv_theme_linked_and_clipboard_messages(tmp_path):
    plot = ip.Plot()
    sent = _capture_messages(plot)
    handle = plot.line("mid", np.array([0, 1, 2, 3], dtype=np.float32), x=np.array([0, 1, 2, 3], dtype=np.float32))
    selection = {"x_min": 0.5, "x_max": 2.5, "y_min": 0.5, "y_max": 2.5, "subplot_index": 0}

    indices = plot.indices_for_selection(selection, "mid")
    np.testing.assert_array_equal(indices, np.array([1, 2], dtype=np.int64))

    sent.clear()
    plot.highlight_selection(selection, handle)
    assert [msg[0]["type"] for msg in sent] == ["primitive_add", "primitive_add"]
    assert sent[0][0]["kind"] == "drag_rect"
    assert sent[1][0]["kind"] == "scatter"

    csv_path = tmp_path / "selection.csv"
    csv_text = plot.export_csv_selection(selection, "mid", filename=csv_path)
    assert "series_id,series_name,index,x,y" in csv_text
    assert ",mid,1,1,1" in csv_text
    assert csv_path.read_text(encoding="utf-8") == csv_text

    sent.clear()
    plot.set_theme("notebook")
    plot.set_colormap("Viridis")
    plot.set_linked_crosshair("group-a", axis="xy")
    plot.copy_png_to_clipboard()
    assert [msg[0]["type"] for msg in sent] == ["theme", "colormap", "linked_crosshair", "copy_png_to_clipboard"]
    assert sent[2][0]["group_id"] == "group-a"


def test_state_restore_and_dashboard_api():
    plot = ip.Plot(title="state")
    _capture_messages(plot)
    plot.set_view(0, 10, -1, 1)
    plot.set_theme("nbimplot")
    plot.set_linked_crosshair("state-group")
    plot.line("mid", np.array([1, 2, 3], dtype=np.float32), x=np.array([10, 11, 12], dtype=np.float32), marker="circle")
    state = plot.get_state(include_data=True)

    restored = ip.Plot()
    sent = _capture_messages(restored)
    restored.set_state(state)
    assert restored.get_state(include_data=True)["series"][0]["name"] == "mid"
    assert any(msg[0]["type"] == "theme" for msg in sent)
    assert any(msg[0]["type"] == "linked_crosshair" for msg in sent)

    dash = ip.Dashboard(2, 2, title="D", theme="nbimplot")
    sent = _capture_messages(dash._plot)
    dash.set_theme("notebook")
    dash.set_linked_crosshair("dash")
    dash.copy_png_to_clipboard()
    assert [msg[0]["type"] for msg in sent] == ["theme", "linked_crosshair", "copy_png_to_clipboard"]


def test_axis_config_and_aligned_group_messages_replay():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.set_axis_label("x2", "Top")
    plot.set_axis_format("y2", "%.2f")
    plot.set_axis_ticks("x1", np.array([1, 2, 3], dtype=np.float32), labels=["a", "b", "c"])
    plot.set_axis_limits_constraints("x1", 0.0, 10.0)
    plot.set_axis_zoom_constraints("y1", 0.1, 100.0)
    plot.set_axis_link("x2", "x1")
    plot.set_aligned_group("grp", enabled=True, vertical=False)

    sent.clear()
    plot._handle_frontend_message(plot, {"type": "frontend_ready"}, None)

    axis_labels = [c for c, _ in sent if c.get("type") == "axis_label"]
    assert any(msg["axis"] == 1 and msg["label"] == "Top" for msg in axis_labels)

    axis_formats = [c for c, _ in sent if c.get("type") == "axis_format"]
    assert any(msg["axis"] == 4 and msg["format"] == "%.2f" for msg in axis_formats)

    axis_ticks = [c for c, _ in sent if c.get("type") == "axis_ticks"]
    assert any(msg["axis"] == 0 and msg["labels"] == ["a", "b", "c"] for msg in axis_ticks)

    axis_limits = [c for c, _ in sent if c.get("type") == "axis_limits_constraints"]
    assert any(msg["axis"] == 0 and msg["enabled"] is True for msg in axis_limits)

    axis_zoom = [c for c, _ in sent if c.get("type") == "axis_zoom_constraints"]
    assert any(msg["axis"] == 3 and msg["enabled"] is True for msg in axis_zoom)

    axis_links = [c for c, _ in sent if c.get("type") == "axis_link"]
    assert any(msg["axis"] == 1 and msg["target_axis"] == 0 for msg in axis_links)

    aligned = [c for c, _ in sent if c.get("type") == "aligned_group"]
    assert aligned
    assert aligned[-1]["group_id"] == "grp"
    assert aligned[-1]["enabled"] is True
    assert aligned[-1]["vertical"] is False


def test_new_implot_widget_and_dragdrop_primitives_payloads():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.tag_x(2.5, label_fmt="", round_value=True)
    content, _ = sent[-1]
    assert content["kind"] == "tag_x"
    assert content["value"] == 2.5
    assert content["label_fmt"] == ""
    assert content["round_value"] is True

    plot.tag_y(-1.25, label_fmt="%g")
    content, _ = sent[-1]
    assert content["kind"] == "tag_y"
    assert content["value"] == -1.25

    plot.colormap_slider(label="CMap", t=0.4, fmt="%.3f")
    content, _ = sent[-1]
    assert content["kind"] == "colormap_slider"
    assert content["label"] == "CMap"
    assert content["value"] == 0.4
    assert content["label_fmt"] == "%.3f"

    plot.colormap_button(label="Swap", width=40, height=12)
    content, _ = sent[-1]
    assert content["kind"] == "colormap_button"
    assert content["label"] == "Swap"
    assert content["x"] == 40.0
    assert content["y"] == 12.0

    plot.colormap_selector(label="Select")
    content, _ = sent[-1]
    assert content["kind"] == "colormap_selector"
    assert content["label"] == "Select"

    plot.drag_drop_plot(source=True, target=False)
    content, _ = sent[-1]
    assert content["kind"] == "drag_drop_plot"
    assert content["has_x"] is True
    assert content["axis"] == "none"

    plot.drag_drop_axis("y2", source=False, target=True)
    content, _ = sent[-1]
    assert content["kind"] == "drag_drop_axis"
    assert content["has_x"] is False
    assert content["length"] == 4
    assert content["value"] == 1.0

    plot.drag_drop_legend(target=False)
    content, _ = sent[-1]
    assert content["kind"] == "drag_drop_legend"
    assert content["value"] == 0.0


def test_error_bars_asymmetric_payload_and_image_flags():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    y = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    err_neg = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    err_pos = np.array([0.6, 0.5, 0.7], dtype=np.float32)
    plot.error_bars("eb", y, err_neg=err_neg, err_pos=err_pos)
    content, buffers = sent[-1]
    assert content["kind"] == "error_bars"
    assert content["asymmetric"] is True
    assert len(buffers) == 2
    interleaved = np.frombuffer(buffers[1], dtype=np.float32)
    np.testing.assert_allclose(interleaved, np.array([0.2, 0.6, 0.3, 0.5, 0.4, 0.7], dtype=np.float32))

    plot.error_bars_h("ebh", y, err_neg=err_neg, err_pos=err_pos)
    content, buffers = sent[-1]
    assert content["kind"] == "error_bars_h"
    assert content["asymmetric"] is True
    assert len(buffers) == 3

    plot.image(
        "imgf",
        np.arange(9, dtype=np.float32).reshape(3, 3),
        tint=(0.2, 0.4, 0.6, 0.8),
        image_flags=3,
    )
    content, buffers = sent[-1]
    assert content["kind"] == "image"
    assert content["image_flags"] == 3
    assert len(buffers) == 2
    tint = np.frombuffer(buffers[1], dtype=np.float32)
    np.testing.assert_allclose(tint, np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32))


def test_batch_lines_datetime_and_single_rgba_color_payloads():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    dates = np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[s]")
    handles = plot.lines(
        {
            "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "b": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            "c": np.array([3.0, 4.0, 5.0], dtype=np.float32),
            "d": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        },
        x=dates,
        colors=(0.1, 0.2, 0.3, 0.4),
    )
    assert len(handles) == 4
    content, buffers = sent[0]
    assert content["type"] == "lines"
    assert len(content["items"]) == 4
    assert len(buffers) == 8
    assert all(item["has_color"] is True for item in content["items"])
    assert all(item["color_r"] == 0.1 for item in content["items"])
    assert all(item["has_x"] is True for item in content["items"])
    axis_messages = [msg for msg, _ in sent if msg.get("type") == "axis_state"]
    assert axis_messages
    assert axis_messages[-1]["axis"] == 0
    assert axis_messages[-1]["scale"] == 2

    plot2 = ip.Plot()
    sent2 = _capture_messages(plot2)
    x = np.arange(3, dtype=np.float32)
    handles2 = plot2.lines(
        {
            "mid": {"x": x, "y": np.array([1.0, 2.0, 3.0], dtype=np.float32), "color": "#1f6f66"},
            "vwap": {"x": x, "data": np.array([1.5, 2.5, 3.5], dtype=np.float32), "color": "#b74b2b"},
        },
        line_weight=1.5,
    )
    assert len(handles2) == 2
    content2, buffers2 = sent2[0]
    assert content2["type"] == "lines"
    assert len(content2["items"]) == 2
    assert all(item["has_x"] for item in content2["items"])
    assert all(item["has_color"] for item in content2["items"])
    assert len(buffers2) == 4


def test_categorical_axis_inputs_emit_axis_ticks():
    plot = ip.Plot()
    sent = _capture_messages(plot)

    plot.bars("by-sector", np.array([3.0, 5.0, 2.0], dtype=np.float32), x=["energy", "tech", "health"])
    primitive = next(msg for msg, _ in sent if msg.get("type") == "primitive_add")
    assert primitive["kind"] == "bars"
    assert primitive["has_x"] is True
    ticks = [(msg, buffers) for msg, buffers in sent if msg.get("type") == "axis_ticks"]
    assert ticks
    assert ticks[-1][0]["labels"] == ["energy", "tech", "health"]
    np.testing.assert_allclose(
        np.frombuffer(ticks[-1][1][0], dtype=np.float32),
        np.array([0, 1, 2], dtype=np.float32),
    )


def test_specialty_primitives_and_export_html_payloads(tmp_path):
    plot = ip.Plot(title="specialty")
    sent = _capture_messages(plot)

    x = np.arange(4, dtype=np.float32)
    open_ = np.array([10.0, 11.0, 10.5, 12.0], dtype=np.float32)
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + np.array([0.5, -0.25, 0.75, -0.5], dtype=np.float32)
    plot.candlestick("candles", x=x, open=open_, high=high, low=low, close=close)
    content, buffers = sent[-1]
    assert content["kind"] == "candlestick"
    assert content["length"] == 4
    assert len(buffers) == 3
    assert np.frombuffer(buffers[2], dtype=np.float32).size == 12

    plot.ohlc("ohlc", x=x, open=open_, high=high, low=low, close=close)
    content, buffers = sent[-1]
    assert content["kind"] == "ohlc"
    assert len(buffers) == 3

    plot.quiver("vectors", x, x, np.ones_like(x), -np.ones_like(x), scale=0.5, normalize=True)
    content, buffers = sent[-1]
    assert content["kind"] == "quiver"
    assert content["scale"] == 0.5
    assert content["normalize"] is True
    assert np.frombuffer(buffers[2], dtype=np.float32).size == 8

    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    plot.contour("contours", z, levels=[2.0, 5.0, 9.0], bounds=((0, -1), (3, 1)), line_weight=1.5)
    content, buffers = sent[-1]
    assert content["kind"] == "contour"
    assert content["rows"] == 3
    assert content["cols"] == 4
    assert content["levels"] == 3
    assert content["line_weight"] == 1.5
    assert np.frombuffer(buffers[1], dtype=np.float32).tolist() == [2.0, 5.0, 9.0]

    plot.waterfall("waterfall", z, x=x, y_offsets=np.array([0.0, 2.0, 4.0], dtype=np.float32), scale=0.25)
    content, buffers = sent[-1]
    assert content["kind"] == "waterfall"
    assert content["has_x"] is True
    assert content["scale"] == 0.25
    assert len(buffers) == 3

    plot.spectrogram("spectrogram", z, label_fmt="", show_colorbar=True, colorbar_format="%.2f")
    content, buffers = sent[-1]
    assert content["kind"] == "spectrogram"
    assert content["label_fmt"] == ""
    assert content["show_colorbar"] is True
    assert content["colorbar_format"] == "%.2f"
    assert len(buffers) == 1

    html_path = tmp_path / "plot.html"
    html_text = plot.export_html(html_path, runtime_url="/nbimplot/index.js")
    assert html_path.read_text(encoding="utf-8") == html_text
    assert "import { createPlot } from \"/nbimplot/index.js\"" in html_text
    assert "\"primitives\"" in html_text
    assert "spectrogram" in html_text


def test_show_returns_none_to_prevent_double_display():
    plot = ip.Plot()
    with patch("nbimplot._plot.display") as mocked_display:
        result = plot.show()
    assert result is None
    mocked_display.assert_called_once_with(plot)

    sub = ip.Subplots(1, 1)
    with patch("nbimplot._plot.display") as mocked_display:
        result = sub.show()
    assert result is None
    assert mocked_display.call_count == 1


def test_plot_has_notebook_cell_renderer_mimebundle():
    plot = ip.Plot(width=640, height=320, title="Cell Renderer")
    _capture_messages(plot)
    plot.line("mid", np.arange(5, dtype=np.float32))

    bundle = plot._repr_mimebundle_()
    assert bundle is not None
    data, metadata = bundle

    assert metadata == {}
    assert "application/vnd.jupyter.widget-view+json" in data
    assert data["application/vnd.jupyter.widget-view+json"]["version_major"] == 2
    assert data["text/plain"] == (
        "nbimplot.Plot(width=640, height=320, title='Cell Renderer', "
        "series=1, primitives=0, renderer='wasm-implot')"
    )

    data, _ = plot._repr_mimebundle_(exclude=["text/plain"])
    assert "text/plain" not in data


def test_subplot_wrappers_delegate_notebook_cell_renderer():
    sub = ip.Subplots(2, 2, title="Grid")
    bundle = sub._repr_mimebundle_()
    assert bundle is not None
    data, _ = bundle
    assert "application/vnd.jupyter.widget-view+json" in data
    assert data["text/plain"] == "nbimplot.Subplots(rows=2, cols=2, title='Grid', renderer='wasm-implot')"

    dash = ip.Dashboard(1, 2, title="Desk")
    bundle = dash._repr_mimebundle_()
    assert bundle is not None
    data, _ = bundle
    assert "application/vnd.jupyter.widget-view+json" in data
    assert data["text/plain"] == "nbimplot.Dashboard(rows=1, cols=2, title='Desk', renderer='wasm-implot')"
