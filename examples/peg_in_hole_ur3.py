#!/usr/bin/env python3
"""Is the UR3 peg-in-hole task reachable? Checks the hole, the hover pose and the spiral-search area.

Uses the UR3 model and constants from UR3_ROS2_PICK_AND_PLACE (ur_force_control/peg_in_hole_env.py):
hole top face centre (0.28, 0.14, 0.12) in the arm base frame, 30 mm deep, 10 mm radius / 60 mm long peg,
hover 30 mm above the top face, spiral search up to 6 mm. In Gazebo the same hole is at z = 1.135 because the
URDF root frame includes the 1.015 m table.

    python3 examples/peg_in_hole_ur3.py [--repo ~/UR3_ROS2_PICK_AND_PLACE]
"""
import argparse
import os
import sys

import numpy as np

from reachlens import load_arm, solve_pose
from reachlens.ik import tool_down

HOLE = np.array([0.28, 0.14, 0.12])
PEG_LEN, HOLE_DEPTH, HOVER, R_SPIRAL = 0.06, 0.03, 0.03, 0.006
GAZEBO_HOVER_Q = [0.0966, -1.4885, 1.9917, -2.0739, -1.5708, -1.4742]   # from record_peg_in_hole.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/UR3_ROS2_PICK_AND_PLACE"))
    a = ap.parse_args()
    xml = os.path.join(a.repo, "ur_rl_training/models/ur3/ur3.xml")
    model, data, site, q0 = load_arm("ur3", xml=xml)
    R = tool_down()
    bad = 0

    def check(label, p):
        nonlocal bad
        r = solve_pose(model, data, site, p, R, tool_length=PEG_LEN, seeds=[q0, GAZEBO_HOVER_Q], n_random=10)
        bad += not r.reachable
        tag = "ok " if r.reachable else "FAIL"
        print(f"{tag} {label:<34} tip=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})  manip={r.manipulability:.4f}  "
              f"margin={r.limit_margin:.2f}  err={r.pos_err * 1e3:.2f} mm")
        return r

    check("hover (30 mm above the hole)", HOLE + [0, 0, HOVER])
    check("hole top face", HOLE)
    check("fully inserted (28 mm deep)", HOLE - [0, 0, HOLE_DEPTH - 0.002])
    for k in range(8):
        th = 2 * np.pi * k / 8
        off = R_SPIRAL * np.array([np.cos(th), np.sin(th), 0])
        check(f"spiral edge {np.degrees(th):3.0f} deg (r=6 mm)", HOLE + [0, 0, HOVER] + off)

    # does the recorded Gazebo hover joint vector actually put the peg tip over the hole?
    import mujoco
    data.qpos[:6] = GAZEBO_HOVER_Q
    mujoco.mj_kinematics(model, data)
    tip = data.site_xpos[site] + data.site_xmat[site].reshape(3, 3)[:, 2] * PEG_LEN
    want = HOLE + [0, 0, HOVER]
    print(f"FK of the Gazebo HOVER_Q -> peg tip {np.round(tip, 4).tolist()}, hover target {want.tolist()}, "
          f"distance {np.linalg.norm(tip - want) * 1e3:.1f} mm")
    print("RESULT:", "all reachable" if bad == 0 and np.linalg.norm(tip - want) < 5e-3 else f"{bad} problem(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
