import csv
import math
import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from reachlens import ARMS, load_arm, map_slice, solve_pose, write_csv
from reachlens.arms import ArmNotFound, resolve_xml
from reachlens.cli import main
from reachlens.ik import limit_margin, manipulability, reach_bound, tool_down, wrap_q


def _have(arm):
    try:
        resolve_xml(arm)
        return True
    except ArmNotFound:
        return False


needs_ur3 = pytest.mark.skipif(not _have("ur3"), reason="UR3 MJCF not found (set REACHLENS_UR3_XML)")
needs_ur5e = pytest.mark.skipif(not _have("ur5e"), reason="UR5e MJCF not found (set REACHLENS_UR5E_XML)")


def test_tool_down_points_z_down_and_is_a_rotation():
    for yaw in (0.0, 0.7, -2.0):
        R = tool_down(yaw)
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
        assert np.allclose(R[:, 2], [0, 0, -1])


def test_unknown_xml_is_a_clear_error():
    with pytest.raises(ArmNotFound):
        load_arm("ur3", xml="/nonexistent/arm.xml")


def test_manipulability_of_identity_and_singular():
    assert manipulability(np.eye(6))[0] == pytest.approx(1.0)
    J = np.eye(6)
    J[5] = J[4]
    w, cond = manipulability(J)
    assert w == pytest.approx(0.0, abs=1e-12) and math.isinf(cond)


@needs_ur3
def test_wrap_and_limit_margin():
    m, d, s, q0 = load_arm("ur3")
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2 * np.pi - 0.1])
    assert wrap_q(m, q)[5] == pytest.approx(-0.1)
    assert limit_margin(m, np.zeros(6)) == pytest.approx(1.0)
    q = np.zeros(6)
    q[2] = m.jnt_range[2, 1]                           # elbow exactly on its limit
    assert limit_margin(m, q) == pytest.approx(0.0, abs=1e-9)


@needs_ur3
def test_reach_bound_is_never_violated():
    m, d, s, q0 = load_arm("ur3")
    anchor, bound = reach_bound(m, d, s)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        d.qpos[:6] = rng.uniform(-np.pi, np.pi, 6)
        mujoco.mj_kinematics(m, d)
        worst = max(worst, np.linalg.norm(d.site_xpos[s] - anchor))
    assert worst <= bound


@needs_ur3
def test_ur3_reaches_a_normal_pose_and_solution_is_correct():
    m, d, s, q0 = load_arm("ur3")
    r = solve_pose(m, d, s, (0.3, 0.1, 0.25), seeds=[q0])
    assert r.reachable and r.pos_err < 1e-3 and r.ori_err_deg < 0.5
    assert r.manipulability > 0 and 0.0 <= r.limit_margin <= 1.0
    d.qpos[:6] = r.q                                   # forward kinematics of the answer lands on the target
    mujoco.mj_kinematics(m, d)
    assert np.allclose(d.site_xpos[s], (0.3, 0.1, 0.25), atol=2e-3)
    assert np.allclose(d.site_xmat[s].reshape(3, 3)[:, 2], [0, 0, -1], atol=1e-2)


@needs_ur3
def test_tool_length_moves_the_tip_not_the_site():
    m, d, s, q0 = load_arm("ur3")
    r = solve_pose(m, d, s, (0.3, 0.1, 0.25), tool_length=0.06, seeds=[q0])
    d.qpos[:6] = r.q
    mujoco.mj_kinematics(m, d)
    assert r.reachable
    assert np.allclose(d.site_xpos[s], (0.3, 0.1, 0.31), atol=2e-3)       # site sits 60 mm above the tip


@needs_ur3
def test_far_pose_is_unreachable_without_running_ik():
    m, d, s, q0 = load_arm("ur3")
    r = solve_pose(m, d, s, (3.0, 0.0, 0.2), seeds=[q0])
    assert not r.reachable and r.iters == 0 and r.manipulability == 0.0


@needs_ur3
def test_pose_just_inside_workspace_limits_fails_gracefully():
    m, d, s, q0 = load_arm("ur3")
    r = solve_pose(m, d, s, (0.4, 0.0, -0.35), seeds=[q0])     # below the base: inside the radial bound, but no IK
    assert not r.reachable and math.isfinite(r.pos_err)


@needs_ur3
def test_map_slice_shape_summary_and_csv(tmp_path):
    from reachlens.mapping import summary
    m, d, s, q0 = load_arm("ur3")
    res = map_slice(m, d, s, 0.25, half=0.6, n=9, q_home=q0)
    assert res["grids"]["reachable"].shape == (9, 9) and len(res["rows"]) == 81
    sm = summary(res)
    assert 0.2 < sm["success_rate"] < 0.95                      # some inside the annulus, corners out of reach
    centre = res["grids"]["reachable"][4, 4]
    assert centre == 0.0                                        # directly above the base is inside the dead zone
    p = tmp_path / "r.csv"
    write_csv(p, res["rows"])
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 81 and set(rows[0]) >= {"x", "y", "z", "reachable", "manipulability", "limit_margin"}


@needs_ur3
def test_cli_check_exit_codes(capsys):
    assert main(["check", "--arm", "ur3", "--pose", "0.3", "0.1", "0.25"]) == 0
    assert "REACHABLE" in capsys.readouterr().out
    assert main(["check", "--arm", "ur3", "--pose", "2", "0", "0.2"]) == 1
    assert main(["check", "--arm", "ur3", "--xml", "/nope.xml", "--pose", "0", "0", "0"]) == 2


@needs_ur3
def test_cli_map_writes_png_and_csv(tmp_path):
    out = tmp_path / "m.png"
    assert main(["map", "--arm", "ur3", "--z", "0.25", "-n", "7", "--out", str(out), "--mark", "0.3", "0.1", "hole"]) == 0
    assert out.stat().st_size > 5000 and (tmp_path / "m.csv").exists()


@needs_ur5e
def test_ur5e_larger_reach_than_ur3():
    m3, d3, s3, _ = load_arm("ur3")
    m5, d5, s5, q5 = load_arm("ur5e")
    assert reach_bound(m5, d5, s5)[1] > reach_bound(m3, d3, s3)[1]
    assert solve_pose(m5, d5, s5, (0.6, 0.2, 0.3), seeds=[q5]).reachable
    assert not solve_pose(m3, d3, s3, (0.6, 0.2, 0.3)).reachable      # beyond the UR3's arm
