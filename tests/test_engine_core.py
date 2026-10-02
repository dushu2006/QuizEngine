from __future__ import annotations

import pytest

from quizengine.config import ConfigError, EngineConfig
from quizengine.contracts import Frame
from quizengine.geometry import Geometry, box_center, box_iou
from quizengine.harness import build_runtime
from quizengine.fixtures import demo_sequence, write_fixture_frames
from quizengine.perception.pixels import marker_zone, rectangular_outlines
from quizengine.fixtures import build_scene
from quizengine.render import render_scene


def test_geometry_round_trips_and_rescales_without_stale_pixels():
    source = Geometry(1280, 800)
    box = (320, 200, 256, 120)
    relative = source.to_rel(box)
    target = Geometry(1920, 1200)
    mapped = target.to_abs(relative)
    assert mapped == (480, 300, 384, 180)
    assert target.rescale(source).map_box(box) == mapped
    assert box_iou(box, box) == 1.0
    assert box_center((0, 0, 10, 10)) == (5, 5)


def test_frame_contract_requires_hash_sequence_and_geometry():
    frame = Frame(seq=4, ts=1.5, size_px=(1280, 800), hash="sha1:test")
    assert frame.seq == 4 and frame.size_px == (1280, 800)
    with pytest.raises(Exception):
        Frame(seq=4, ts=1.5, width=1280, height=800, hash="sha1:test")


def test_attestation_requirement_cannot_be_disabled():
    with pytest.raises(ConfigError):
        EngineConfig.from_dict({"run": {"attest_required": False}})
    assert EngineConfig.default().run.attest_required is True


def test_thin_button_outlines_survive_pixel_area_floor():
    scene = build_scene("button_row", "Pick a button", ["Alpha", "Beta", "Gamma", "Delta"])
    pixels = render_scene(scene)
    background = tuple(int(channel) for channel in pixels[0, 0])
    outlines = rectangular_outlines(pixels, background)
    for option in scene.options:
        assert any(box_iou(tuple(option.hit_box), box) > 0.98 for box in outlines)


def test_marker_zone_does_not_cross_into_selected_left_sibling():
    # Two side-by-side row options. The second option's glyph is left of its label,
    # but its detector must not inspect beyond the first option's right edge.
    zone = marker_zone(
        (660, 240, 251, 121),
        "radio",
        text_box=(681, 289, 41, 21),
        left_limit=621,
    )
    assert zone[0] >= 623
    assert zone[2] > 0


@pytest.mark.integration
def test_exported_fixture_directory_replays_through_closed_loop(tmp_path):
    fixture_dir = tmp_path / "fixture-replay"
    write_fixture_frames(demo_sequence(), fixture_dir)
    runtime = build_runtime(
        target="sim",
        scenario="fixture-dir",
        fixture_dir=fixture_dir,
        runs_dir=tmp_path / "runs",
    )
    report = runtime.run()
    assert report.outcome.value == "completed"
    assert len(report.questions) == 6
    assert report.correct == 6 and report.accuracy == 1.0
    assert report.unverified_actions == 0
    assert report.illegal_transitions == 0
    assert report.stale_coordinate_violations == 0


@pytest.mark.integration
def test_full_variant_matrix_is_verified_closed_loop(tmp_path):
    runtime = build_runtime(target="sim", scenario="variant-matrix", runs_dir=tmp_path)
    report = runtime.run()
    assert report.outcome.value == "completed"
    assert len(report.questions) == 18
    assert report.correct == 18 and report.incorrect == 0 and report.accuracy == 1.0
    assert report.unverified_actions == 0
    assert report.illegal_transitions == 0
    assert report.stale_coordinate_violations == 0
    assert runtime.orchestrator.stats.overlays_dismissed >= 1
    assert runtime.orchestrator.stats.end_state_frames >= 2
