"""Sweeping a grid of tool-down poses and collecting per-point results."""
import csv

import numpy as np

from reachlens.ik import solve_pose

FIELDS = ["x", "y", "z", "reachable", "pos_err_m", "ori_err_deg", "manipulability", "cond", "limit_margin", "iters"]


def _best_solution(model, data, site, pos, R, tool_length, seeds, ik_kw):
    """IK from each given seed separately and keep the solution with the best joint-limit margin (then manipulability),
    so the map does not depend on which arm configuration a neighbouring point happened to land in.
    Falls back to random restarts only when no seed converges."""
    best = None
    for sd in (s for s in seeds if s is not None):
        r = solve_pose(model, data, site, pos, R, tool_length=tool_length, seeds=[sd], n_random=0, **ik_kw)
        if r.reachable and (best is None or (r.limit_margin, r.manipulability) > (best.limit_margin, best.manipulability)):
            best = r
    if best is not None:
        return best
    return solve_pose(model, data, site, pos, R, tool_length=tool_length, seeds=[], **ik_kw)


def map_slice(model, data, site, z, half=0.6, n=41, center=(0.0, 0.0), q_home=None, yaw=0.0, tool_length=0.0, **ik_kw):
    """Map an n x n horizontal slice at height z. Each point is solved from its left neighbour's solution and from
    the home pose (best margin wins), then from random seeds if both fail. Returns dict(xs, ys, z, rows, grids)."""
    from reachlens.ik import tool_down
    R = tool_down(yaw)
    xs = np.linspace(center[0] - half, center[0] + half, n)
    ys = np.linspace(center[1] - half, center[1] + half, n)
    rows, g = [], {k: np.full((n, n), np.nan) for k in ("reachable", "manipulability", "limit_margin", "cond")}
    prev = None
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            r = _best_solution(model, data, site, (x, y, z), R, tool_length, [prev, q_home], ik_kw)
            if r.reachable:
                prev = r.q
            rows.append(dict(x=x, y=y, z=z, reachable=int(r.reachable), pos_err_m=r.pos_err, ori_err_deg=r.ori_err_deg,
                             manipulability=r.manipulability, cond=r.cond, limit_margin=r.limit_margin, iters=r.iters))
            g["reachable"][j, i] = float(r.reachable)
            if r.reachable:
                g["manipulability"][j, i] = r.manipulability
                g["limit_margin"][j, i] = r.limit_margin
                g["cond"][j, i] = r.cond
    return dict(xs=xs, ys=ys, z=z, rows=rows, grids=g)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in FIELDS})


def summary(res):
    rows = res["rows"]
    ok = [r for r in rows if r["reachable"]]
    return dict(points=len(rows), reachable=len(ok), success_rate=len(ok) / max(len(rows), 1),
                mean_manipulability=float(np.mean([r["manipulability"] for r in ok])) if ok else 0.0,
                min_limit_margin=float(min((r["limit_margin"] for r in ok), default=0.0)))
