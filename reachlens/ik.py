"""Damped-least-squares IK for a tool-down pose, with manipulability and joint-limit margin."""
from dataclasses import dataclass

import os

import mujoco
import numpy as np

try:                                   # optional C++ core, built by setup.py when pybind11 and a compiler are present
    if os.environ.get("REACHLENS_NO_CPP"):
        raise ImportError
    from reachlens import _fast
except ImportError:
    _fast = None


def have_cpp():
    return _fast is not None


def tool_down(yaw=0.0):
    """Rotation whose tool z axis points straight down (world -z), rotated by `yaw` about the vertical."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Rz @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])


@dataclass
class PointResult:
    reachable: bool
    pos_err: float            # m
    ori_err_deg: float
    manipulability: float     # Yoshikawa sqrt(det(J J^T)) of the 6x6 Jacobian at the solution
    cond: float               # sigma_max / sigma_min (inf at a singularity)
    limit_margin: float       # 0 = a joint sits on its limit, 1 = every joint mid-range
    iters: int
    q: np.ndarray


def _err(R, Rd, p, pd):
    # small-angle orientation error 0.5 * sum_i (R_i x Rd_i), written with M = R Rd^T instead of three np.cross calls
    # (np.cross on 3-vectors dominated the runtime of a map)
    M = R @ Rd.T
    e_o = 0.5 * np.array([M[1, 2] - M[2, 1], M[2, 0] - M[0, 2], M[0, 1] - M[1, 0]])
    return np.concatenate([pd - p, e_o])


def limit_margin(model, q):
    lo, hi = model.jnt_range[: len(q), 0], model.jnt_range[: len(q), 1]
    lim = model.jnt_limited[: len(q)].astype(bool)
    if not lim.any():
        return 1.0
    half = (hi - lo) / 2
    m = np.minimum(q - lo, hi - q) / np.where(half > 0, half, np.inf)
    return float(np.clip(m[lim].min(), 0.0, 1.0))


def wrap_q(model, q):
    """Equivalent joint angles closest to zero: shift each joint by multiples of 2*pi while it stays in its limits.
    (UR joints span +-2*pi, so IK can land on a solution that is 2*pi away from the natural one.)"""
    q = np.array(q, dtype=float)
    lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    lim = model.jnt_limited[:6].astype(bool)
    for i in range(6):
        w = (q[i] + np.pi) % (2 * np.pi) - np.pi
        if not lim[i] or lo[i] <= w <= hi[i]:
            q[i] = w
    return q


def jacobian(model, data, site):
    J = np.empty((6, model.nv))
    mujoco.mj_jacSite(model, data, J[:3], J[3:], site)      # translational rows, then rotational rows
    return J[:, :6]


def manipulability(J):
    s = np.linalg.svd(J, compute_uv=False)
    w = float(np.prod(s))                      # = sqrt(det(J J^T)) for a square or full-row-rank J
    cond = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")
    return w, cond


def reach_bound(model, data, site):
    """Provable upper bound on |site - joint1 anchor|: the sum of link offsets along the chain (triangle inequality).
    Returns (anchor xyz, bound). Targets farther than the bound cannot be reached, so IK is skipped for them."""
    mujoco.mj_kinematics(model, data)
    anchor = data.xanchor[0].copy()
    bound = float(np.linalg.norm(model.site_pos[site]))
    b = int(model.site_bodyid[site])
    first = int(model.jnt_bodyid[0])
    while b != first and b > 0:
        bound += float(np.linalg.norm(model.body_pos[b]))
        b = int(model.body_parentid[b])
    for j in range(6):
        bound += float(np.linalg.norm(model.jnt_pos[j]))
    return anchor, bound


def _refine(model, data, site, q, pd, Rd, iters, damping, step, patience=25):
    if _fast is not None:
        q, it, e = _fast.refine(model._address, data._address, site, q, pd, np.ascontiguousarray(Rd), iters, damping, step, patience)
        return q, it, e
    return _refine_py(model, data, site, q, pd, Rd, iters, damping, step, patience)


def _refine_py(model, data, site, q, pd, Rd, iters, damping, step, patience=25):
    lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    lim = model.jnt_limited[:6].astype(bool)
    best_e, since = np.inf, 0
    damp = damping ** 2 * np.eye(6)
    for it in range(1, iters + 1):
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        R = data.site_xmat[site].reshape(3, 3)
        e = _err(R, Rd, data.site_xpos[site], pd)
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            return q, it, e
        ne = np.linalg.norm(e[:3]) + 0.1 * np.linalg.norm(e[3:])
        if ne < best_e - 1e-5:
            best_e, since = ne, 0
        else:
            since += 1
            if since >= patience:      # stalled: stuck at a limit or a singularity, more iterations will not help
                break
        J = jacobian(model, data, site)
        dq = J.T @ np.linalg.solve(J @ J.T + damp, e)
        n = np.abs(dq).max()
        if n > step:
            dq *= step / n
        q = q + dq
        q = np.where(lim, np.clip(q, lo, hi), q)
    data.qpos[:6] = q
    mujoco.mj_kinematics(model, data)
    R = data.site_xmat[site].reshape(3, 3)
    return q, it, _err(R, Rd, data.site_xpos[site], pd)


def solve_pose(model, data, site, pos, R=None, tool_length=0.0, seeds=None, n_random=6, iters=150, damping=0.05, step=0.3,
               pos_tol=1e-3, ori_tol_deg=0.5, rng=None):
    """Try IK from each seed until one meets the tolerances. Returns a PointResult (best attempt if none do).

    `pos` is where the tool TIP goes; the tip sits `tool_length` m beyond the site along the tool z axis
    (e.g. 0.06 for a peg), so the site target is pos - tool_length * z_axis."""
    Rd = tool_down() if R is None else R
    pd = np.asarray(pos, dtype=float) - tool_length * Rd[:, 2]
    anchor, bound = reach_bound(model, data, site)
    if np.linalg.norm(pd - anchor) > bound + 1e-6:       # provably out of reach: no IK needed
        return PointResult(False, float(np.linalg.norm(pd - anchor) - bound), 180.0, 0.0, float("inf"), 0.0, 0,
                           np.asarray(seeds[0] if seeds else np.zeros(6), dtype=float)[:6])
    rng = np.random.default_rng(0) if rng is None else rng
    tries = [np.asarray(s, dtype=float)[:6] for s in (seeds or [])]
    tries += [rng.uniform(-np.pi, np.pi, 6) for _ in range(n_random)]
    best = None
    for q0 in tries:
        q, it, e = _refine(model, data, site, q0.copy(), pd, Rd, iters, damping, step)
        pe, oe = float(np.linalg.norm(e[:3])), float(np.degrees(np.linalg.norm(e[3:])))
        ok = pe <= pos_tol and oe <= ori_tol_deg
        if best is None or (ok, -pe - oe * 1e-2) > (best[0], -best[1] - best[2] * 1e-2):
            best = (ok, pe, oe, q, it)
        if ok:
            break
    ok, pe, oe, q, it = best
    if ok:
        q = wrap_q(model, q)
    data.qpos[:6] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    w, cond = manipulability(jacobian(model, data, site))
    return PointResult(ok, pe, oe, w if ok else 0.0, cond, limit_margin(model, q) if ok else 0.0, it, q)
