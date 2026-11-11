import argparse
import os
import sys

import numpy as np

from reachlens.arms import ARMS, ArmNotFound, load_arm
from reachlens.ik import solve_pose, tool_down
from reachlens.mapping import map_slice, summary, write_csv


def _add_common(p):
    p.add_argument("--arm", choices=sorted(ARMS), default="ur3")
    p.add_argument("--xml", help="MJCF path (overrides the lookup; needs a tool site)")
    p.add_argument("--site", help="tool site name (default attachment_site)")
    p.add_argument("--yaw", type=float, default=0.0, help="tool yaw about the vertical, rad")
    p.add_argument("--tool-length", type=float, default=0.0,
                   help="tool tip distance beyond the site along the tool z axis, m (e.g. 0.06 for a peg)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reachlens", description="Reachability and IK-success maps for arms (MuJoCo).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="map a horizontal slice of tool-down poses")
    _add_common(m)
    m.add_argument("--z", type=float, help="slice height in m (default per arm)")
    m.add_argument("--half", type=float, help="half-width of the xy window in m (default per arm)")
    m.add_argument("-n", type=int, default=41, help="grid points per side")
    m.add_argument("--out", default="reach.png", help="heatmap PNG; a CSV with the same name is written next to it")
    m.add_argument("--no-csv", action="store_true")
    m.add_argument("--mark", nargs=3, action="append", metavar=("X", "Y", "LABEL"), default=[],
                   help="mark a point on the map (repeatable), e.g. --mark 0.28 0.14 hole")
    c = sub.add_parser("check", help="IK for one tool-down pose; exit 1 if unreachable")
    _add_common(c)
    c.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "Z"), required=True)
    v = sub.add_parser("volume", help="interactive 3D HTML of the reachable workspace (z slices stacked)")
    _add_common(v)
    v.add_argument("--zmin", type=float, help="lowest slice in m (default per arm)")
    v.add_argument("--zmax", type=float, help="highest slice in m")
    v.add_argument("--slices", type=int, default=7)
    v.add_argument("--half", type=float)
    v.add_argument("-n", type=int, default=25, help="grid points per side per slice")
    v.add_argument("--out", default="reach3d.html")
    r = sub.add_parser("render", help="render the arm at the IK solution for one tool-down pose (PNG)")
    _add_common(r)
    r.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "Z"), required=True)
    r.add_argument("--out", default="pose.png")
    r.add_argument("--size", nargs=2, type=int, default=[800, 600], metavar=("W", "H"))
    r.add_argument("--hole", nargs=3, type=float, metavar=("X", "Y", "Z"), help="also draw a hole fixture (top face centre)")
    g = sub.add_parser("sweep-gif", help="GIF of the arm hovering, descending, spiralling and inserting at a hole")
    _add_common(g)
    g.add_argument("--hole", nargs=3, type=float, metavar=("X", "Y", "Z"), default=[0.28, 0.14, 0.12],
                   help="hole top face centre, arm base frame (default: the UR3 peg-in-hole example)")
    g.add_argument("--hover", type=float, default=0.03)
    g.add_argument("--depth", type=float, default=0.03)
    g.add_argument("--spiral-radius", type=float, default=0.006)
    g.add_argument("--out", default="sweep.gif")
    g.add_argument("--size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    g.add_argument("--fps", type=int, default=12)
    a = ap.parse_args(argv)
    try:
        model, data, site, q0 = load_arm(a.arm, a.xml, a.site)
    except (ArmNotFound, ValueError) as e:
        print(f"reachlens: {e}", file=sys.stderr)
        return 2

    if a.cmd == "check":
        r = solve_pose(model, data, site, a.pose, tool_down(a.yaw), tool_length=a.tool_length, seeds=[q0])
        print(f"{'REACHABLE' if r.reachable else 'UNREACHABLE'}  pos_err={r.pos_err * 1e3:.2f} mm  "
              f"ori_err={r.ori_err_deg:.2f} deg")
        if r.reachable:
            print(f"manipulability={r.manipulability:.4f}  cond={r.cond:.1f}  limit_margin={r.limit_margin:.2f}")
            print("q = [" + ", ".join(f"{v:.4f}" for v in r.q) + "]")
        return 0 if r.reachable else 1

    spec = ARMS.get(a.arm)
    if a.cmd == "volume":
        from reachlens.volume import sample_volume, write_html
        z0 = a.zmin if a.zmin is not None else (spec.default_z - 0.2 if spec else 0.1)
        z1 = a.zmax if a.zmax is not None else (spec.default_z + 0.3 if spec else 0.6)
        half = a.half if a.half is not None else (spec.default_range if spec else 0.8)
        cols = sample_volume(model, data, site, q0, np.linspace(z0, z1, a.slices), half, a.n,
                             tool_length=a.tool_length, yaw=a.yaw)
        info = write_html(cols, a.arm, a.out, a.n)
        print(f"{a.arm}: {info['reachable']}/{info['points']} poses reachable over z={z0:.2f}..{z1:.2f} m")
        print(f"wrote {a.out} ({info['bytes'] / 1e3:.0f} kB, self-contained)")
        return 0
    if a.cmd == "render":
        from reachlens.render import render_pose
        r = solve_pose(model, data, site, a.pose, tool_down(a.yaw), tool_length=a.tool_length, seeds=[q0])
        if not r.reachable:
            print(f"UNREACHABLE  pos_err={r.pos_err * 1e3:.2f} mm; nothing rendered")
            return 1
        render_pose(model, data, r.q, a.out, width=a.size[0], height=a.size[1], tool_length=a.tool_length,
                    site=site, target=a.pose, hole_top=a.hole)
        print(f"wrote {a.out}")
        return 0
    if a.cmd == "sweep-gif":
        from reachlens.render import sweep_gif
        try:
            _, n = sweep_gif(model, data, site, q0, a.hole, a.tool_length or 0.06, a.out, width=a.size[0],
                             height=a.size[1], fps=a.fps, hover=a.hover, depth=a.depth, spiral_r=a.spiral_radius,
                             yaw=a.yaw)
        except ValueError as e:
            print(f"reachlens: {e}", file=sys.stderr)
            return 1
        print(f"wrote {a.out} ({n} frames)")
        return 0

    from reachlens.plot import plot_slice
    z = a.z if a.z is not None else (spec.default_z if spec else 0.3)
    half = a.half if a.half is not None else (spec.default_range if spec else 0.8)
    res = map_slice(model, data, site, z, half=half, n=a.n, q_home=q0, yaw=a.yaw, tool_length=a.tool_length)
    s = summary(res)
    plot_slice(res, a.arm, a.out, s, marks=[(float(x), float(y), l) for x, y, l in a.mark])
    print(f"{a.arm}: {s['reachable']}/{s['points']} poses reachable ({s['success_rate']:.1%}) at z={z:.2f} m, "
          f"mean manipulability {s['mean_manipulability']:.4f}, min limit margin {s['min_limit_margin']:.2f}")
    print(f"wrote {a.out}")
    if not a.no_csv:
        csv_path = os.path.splitext(a.out)[0] + ".csv"
        write_csv(csv_path, res["rows"])
        print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
