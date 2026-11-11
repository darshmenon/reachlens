# Changelog

## Unreleased
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
