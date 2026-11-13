# Changelog

## Unreleased
- Optional C++ core (`reachlens._fast`, built by `pip install --no-build-isolation .` when pybind11, a compiler and the MuJoCo headers are present). It runs the damped-least-squares refine loop through the MuJoCo C API on the same model and data. A 41x41 UR3 map went from 18.4 s to 1.5 s (12.6x, one process, same reachable mask). Without it, or with `REACHLENS_NO_CPP=1`, the pure Python solver runs as before.
- `tests/test_fast.py` compares the C++ and Python refine loops call by call (iterations equal, joint angles and error within 1e-6) and checks that a map is identical either way.

## 0.3.0
- `pip install .` (or `pip install git+https://github.com/darshmenon/reachlens`) works in a clean virtualenv: `reachlens check`, `map`, `volume` and `render` were run from one with no ROS or other workspace on `PYTHONPATH`.
- Model lookup: `$REACHLENS_MODELS` takes one or more folders and finds `ur3` and `ur5e` under common layouts. The error for a missing model now lists what was tried and the three ways to fix it.
- `map` is about 3x faster in one process: the orientation error no longer calls `np.cross`, which was about 60% of the run time. The output is bit-identical to 0.2.0.
- `map --jobs N` solves rows in parallel (4 jobs: about 9x faster than 0.2.0 on a 31x31 UR3 grid). The result does not depend on N and has the same reachable mask, but a few points can pick a different arm configuration than the single chain, so `--jobs 1` stays the default.
- `reachlens --version`.
- GitHub Actions: installs the package on Python 3.10, fetches the UR3 and UR5e models, renders with OSMesa and runs the whole test suite. `REACHLENS_REQUIRE_MODELS=1` makes a missing model fail the build instead of skipping tests.

## 0.2.0
- `reachlens volume`: self-contained interactive 3D HTML of the reachable workspace (rotate, zoom, z-range sliders, colour by manipulability, joint-limit margin or height).
- `reachlens render`: MuJoCo offscreen PNG of the arm at the IK solution for a pose, with the tool and an optional hole fixture.
- `reachlens sweep-gif`: GIF of a peg-in-hole approach (hover, descend, spiral, re-centre, insert), IK solved frame by frame.
- Pillow is now a dependency.

## 0.1.0
- `reachlens map`: sweep a horizontal grid of tool-down poses, damped-least-squares IK per point, heatmap PNG (IK success, Yoshikawa manipulability, joint-limit margin) plus a CSV.
- `reachlens check --pose X Y Z`: IK for one pose with the solution, manipulability, condition number and margin; exit 1 if unreachable.
- `--tool-length` so poses name the tool tip (a 60 mm peg) instead of the wrist flange; `--mark` to pin a point such as a hole on the map.
- UR3 and UR5e supported out of the box (models are found on disk, never bundled; `--xml` or `$REACHLENS_<ARM>_XML` to point elsewhere).
- Out-of-reach points are rejected with a provable chain-length bound before any IK runs.
- `examples/peg_in_hole_ur3.py`: checks that a UR3 peg-in-hole hole, hover pose and spiral-search area are reachable.
