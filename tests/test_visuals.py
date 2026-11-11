import json
import re

import numpy as np
import pytest

from reachlens import load_arm
from reachlens.arms import ArmNotFound, resolve_xml
from reachlens.cli import main
from reachlens.render import fixture, peg_sequence
from reachlens.volume import sample_volume, write_html


def _have(arm):
    try:
        resolve_xml(arm)
        return True
    except ArmNotFound:
        return False


needs_ur3 = pytest.mark.skipif(not _have("ur3"), reason="UR3 MJCF not found (set REACHLENS_UR3_XML)")


def _gl_works():
    try:
        import mujoco
        from reachlens.render import _make_renderer
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
        _make_renderer(m, 32, 32).close()
        return True
    except Exception:
        return False


needs_gl = pytest.mark.skipif(not _gl_works(), reason="no headless OpenGL backend (EGL/OSMesa)")


def _cols():
    return dict(x=[0.0, 0.1, 0.2], y=[0.0, 0.0, 0.1], z=[0.2, 0.2, 0.3], r=[1, 1, 0], m=[0.01, 0.02, 0],
                g=[0.5, 0.4, 0])


def test_volume_html_is_self_contained_and_data_roundtrips(tmp_path):
    out = tmp_path / "v.html"
    info = write_html(_cols(), "ur3", str(out), n=3)
    html = out.read_text()
    assert info["reachable"] == 2 and info["points"] == 3
    assert "http://" not in html and "https://" not in html and "<script src" not in html   # no network, no CDN
    assert "2 of 3 sampled poses reachable" in html
    blob = re.search(r"const D=(\{.*?\});const cv", html, re.S).group(1)
    d = json.loads(blob)
    assert d["r"] == [1, 1, 0] and d["n"] == 3 and len(d["x"]) == 3


def test_volume_html_has_controls(tmp_path):
    out = tmp_path / "v.html"
    write_html(_cols(), "ur5e", str(out), n=3)
    html = out.read_text()
    for el in ('id="zlo"', 'id="zhi"', 'id="mode"', 'id="un"', "ur5e reachable workspace"):
        assert el in html


@needs_ur3
def test_sample_volume_columns_are_consistent():
    model, data, site, q0 = load_arm("ur3")
    cols = sample_volume(model, data, site, q0, zs=[0.2, 0.3], half=0.3, n=4, tool_length=0.06)
    assert {len(v) for v in cols.values()} == {32}
    assert sorted(set(cols["z"])) == [0.2, 0.3]
    assert all(m == 0 for m, r in zip(cols["m"], cols["r"]) if not r)
    assert any(cols["r"])


def test_fixture_has_a_square_hole():
    boxes = fixture((0.28, 0.14, 0.12), half=0.0105, wall=0.03, depth=0.03)
    assert len(boxes) == 5
    walls = boxes[:4]
    # no wall box covers the hole centre column (x, y) at the top face height
    for (cx, cy, cz), (sx, sy, sz), _ in walls:
        inside = abs(0.28 - cx) < sx and abs(0.14 - cy) < sy
        assert not inside
    assert all(c[2] < 0.12 for c, _, _ in boxes)       # everything is below the top face


@needs_ur3
def test_peg_sequence_legs_depth_and_continuity():
    model, data, site, q0 = load_arm("ur3")
    hole = np.array([0.28, 0.14, 0.12])
    qs, tips, legs = peg_sequence(model, data, site, q0, hole, tool_length=0.06)
    assert len(qs) == len(tips) == len(legs)
    order = [l for i, l in enumerate(legs) if i == 0 or legs[i - 1] != l]
    assert order == ["move above the hole", "descend to the top face", "spiral search", "re-centre", "insert"]
    assert tips[-1][2] == pytest.approx(hole[2] + 0.001 - 0.028, abs=1e-6)   # 1 mm above the face, then 28 mm down = 27 mm in
    spiral = [t for t, l in zip(tips, legs) if l == "spiral search"]
    assert max(np.hypot(t[0] - hole[0], t[1] - hole[1]) for t in spiral) == pytest.approx(0.006, abs=1e-6)
    steps = [np.max(np.abs(np.array(qs[i + 1]) - qs[i])) for i in range(len(qs) - 1)]
    assert max(steps) < 0.35                                                    # smooth: no joint jumps between frames


@needs_ur3
def test_peg_sequence_rejects_unreachable_hole():
    model, data, site, q0 = load_arm("ur3")
    with pytest.raises(ValueError, match="unreachable"):
        peg_sequence(model, data, site, q0, np.array([1.5, 0.0, 0.12]), tool_length=0.06)


@needs_ur3
def test_cli_render_unreachable_exits_1(tmp_path, capsys):
    out = tmp_path / "p.png"
    assert main(["render", "--arm", "ur3", "--pose", "1.5", "0", "0.2", "--out", str(out)]) == 1
    assert not out.exists()


@needs_ur3
def test_cli_volume_writes_html(tmp_path):
    out = tmp_path / "v.html"
    assert main(["volume", "--arm", "ur3", "--slices", "2", "-n", "4", "--out", str(out)]) == 0
    assert "reachable workspace" in out.read_text()


@needs_ur3
@needs_gl
def test_render_and_sweep_gif_write_files(tmp_path):
    from PIL import Image
    png, gif = tmp_path / "p.png", tmp_path / "s.gif"
    assert main(["render", "--arm", "ur3", "--tool-length", "0.06", "--pose", "0.28", "0.14", "0.15",
                 "--size", "160", "120", "--out", str(png)]) == 0
    img = np.asarray(Image.open(png))
    assert img.shape == (120, 160, 3) and img.std() > 5                          # not a blank frame
    assert main(["sweep-gif", "--arm", "ur3", "--size", "160", "120", "--out", str(gif)]) == 0
    assert Image.open(gif).n_frames > 20
