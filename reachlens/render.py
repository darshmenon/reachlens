"""Offscreen MuJoCo rendering of an arm at an IK solution, and a GIF of it following a pose sequence.

Headless backend: EGL is tried first, then OSMesa (set MUJOCO_GL yourself to force one)."""
import os
import sys

import mujoco
import numpy as np


def _make_renderer(model, width, height):
    """mujoco.Renderer on whichever GL backend works here; MUJOCO_GL is only set when the user has not."""
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    if "MUJOCO_GL" in os.environ:
        return mujoco.Renderer(model, height, width)
    last = None
    quiet = sys.unraisablehook
    sys.unraisablehook = lambda *_: None      # a Renderer that failed to start complains again in __del__
    try:
        for gl in ("egl", "osmesa"):
            os.environ["MUJOCO_GL"] = gl
            try:
                return mujoco.Renderer(model, height, width)
            except Exception as e:      # backend missing or broken: try the next one
                last = f"{gl}: {e}"
    finally:
        sys.unraisablehook = quiet
    del os.environ["MUJOCO_GL"]
    raise RuntimeError(f"no headless OpenGL backend worked (tried egl, osmesa): {last}")


def _add(scene, gtype, size, pos, rgba, mat=None):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).flatten() if mat is None else np.asarray(mat, float).flatten(),
                        np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _cylinder_between(scene, p0, p1, radius, rgba):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CYLINDER, radius, p0, p1)
    scene.ngeom += 1


def fixture(hole_top, half=0.0105, wall=0.03, depth=0.03):
    """Boxes (centre, half-size, rgba) for a square-hole block whose top face centre is `hole_top`, and a floor."""
    x, y, z = hole_top
    h = depth / 2
    cz = z - h
    out = []
    t = wall / 2
    rgba = (0.55, 0.57, 0.62, 1.0)
    out.append(((x, y - half - t, cz), (half + wall, t, h), rgba))
    out.append(((x, y + half + t, cz), (half + wall, t, h), rgba))
    out.append(((x - half - t, y, cz), (t, half, h), rgba))
    out.append(((x + half + t, y, cz), (t, half, h), rgba))
    out.append(((x, y, z - depth - 0.005), (half + wall, half + wall, 0.005), (0.4, 0.42, 0.46, 1.0)))
    return out


def render_frame(renderer, model, data, q, tool_length=0.0, site=None, target=None, trail=(), hole_top=None,
                 camera=None):
    """One RGB frame (H, W, 3 uint8) of the arm at joint vector q, with the tool, an optional target marker,
    a tip trail (list of xyz) and an optional hole fixture drawn on top of the model scene."""
    data.qpos[: len(q)] = q
    mujoco.mj_forward(model, data)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.75
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    c = dict(lookat=(0.17, 0.06, 0.14), distance=0.95, azimuth=128.0, elevation=-22.0)
    c.update(camera or {})
    cam.lookat[:] = c["lookat"]
    cam.distance, cam.azimuth, cam.elevation = c["distance"], c["azimuth"], c["elevation"]
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1           # arms with capsule-only models (e.g. the UR3 training model) keep them in group 3
    for gi in range(model.ngeom):  # recolour untextured grey collision capsules so the arm reads as an arm
        if model.geom_group[gi] == 3 and model.geom_matid[gi] < 0 and tuple(np.round(model.geom_rgba[gi][:3], 2)) == (0.5, 0.5, 0.5):
            model.geom_rgba[gi] = (0.30, 0.52, 0.80, 1.0)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    sc = renderer.scene
    _add(sc, mujoco.mjtGeom.mjGEOM_BOX, (2.0, 2.0, 0.002), (0.15, 0.1, -0.004), (0.96, 0.96, 0.98, 1))
    if hole_top is not None:
        for pos, size, rgba in fixture(hole_top):
            _add(sc, mujoco.mjtGeom.mjGEOM_BOX, size, pos, rgba)
    if site is not None and tool_length > 0:
        p = data.site_xpos[site].copy()
        zax = data.site_xmat[site].reshape(3, 3)[:, 2]
        _cylinder_between(sc, p, p + zax * tool_length, 0.01, (0.95, 0.62, 0.12, 1))
    for p in trail:
        _add(sc, mujoco.mjtGeom.mjGEOM_SPHERE, (0.0016,) * 3, p, (0.2, 0.55, 1.0, 0.9))
    if target is not None:
        _add(sc, mujoco.mjtGeom.mjGEOM_SPHERE, (0.006,) * 3, target, (0.9, 0.15, 0.15, 0.85))
    return renderer.render().copy()


def render_pose(model, data, q, out, width=800, height=600, **kw):
    from PIL import Image
    r = _make_renderer(model, width, height)
    try:
        img = render_frame(r, model, data, q, **kw)
    finally:
        r.close()
    Image.fromarray(img).save(out)
    return out


def peg_sequence(model, data, site, q_home, hole_top, tool_length, hover=0.03, depth=0.03, spiral_r=0.006,
                 turns=1.5, frames=(14, 8, 22, 6, 10), seeds=(), yaw=0.0):
    """Joint vectors + tip positions for: home -> hover -> down to the top face -> spiral out -> back to centre ->
    insert. Cartesian legs are solved by IK frame by frame (seeded from the previous frame), so the tip path is
    straight and the arm moves smoothly. Returns (list of q, list of tip xyz, list of leg names), or raises if a
    waypoint is unreachable."""
    from reachlens.ik import solve_pose, tool_down
    R = tool_down(yaw)
    hole = np.asarray(hole_top, float)

    def ik(p, seed_list):
        r = solve_pose(model, data, site, p, R, tool_length=tool_length, seeds=seed_list, n_random=10)
        if not r.reachable:
            raise ValueError(f"waypoint {np.round(p, 4).tolist()} is unreachable (pos err {r.pos_err * 1e3:.1f} mm)")
        return r.q

    hover_p = hole + [0, 0, hover]
    q_hover = ik(hover_p, [*seeds, q_home])
    qs, tips, legs = [], [], []

    def tip_of(q):
        data.qpos[: len(q)] = q
        mujoco.mj_kinematics(model, data)
        return data.site_xpos[site] + data.site_xmat[site].reshape(3, 3)[:, 2] * tool_length

    for k in range(frames[0]):
        s = (k + 1) / frames[0]
        s = s * s * (3 - 2 * s)
        q = q_home + s * (q_hover - q_home)
        qs.append(q), tips.append(tip_of(q)), legs.append("move above the hole")
    q_prev = q_hover

    def leg(points, name):
        nonlocal q_prev
        for p in points:
            q_prev = ik(p, [q_prev])
            qs.append(q_prev), tips.append(np.asarray(p, float)), legs.append(name)

    leg([hover_p - [0, 0, hover * (k + 1) / frames[1]] for k in range(frames[1])], "descend to the top face")
    top = hole + [0, 0, 0.001]
    n = frames[2]
    spiral = []
    for k in range(n):
        th = 2 * np.pi * turns * (k + 1) / n
        r = spiral_r * (k + 1) / n
        spiral.append(top + [r * np.cos(th), r * np.sin(th), 0])
    leg(spiral, "spiral search")
    back = [spiral[-1] + (top - spiral[-1]) * (k + 1) / frames[3] for k in range(frames[3])]
    leg(back, "re-centre")
    leg([top - [0, 0, (depth - 0.002) * (k + 1) / frames[4]] for k in range(frames[4])], "insert")
    return qs, tips, legs


def sweep_gif(model, data, site, q_home, hole_top, tool_length, out, width=640, height=480, fps=12, hover=0.03,
              depth=0.03, spiral_r=0.006, yaw=0.0, seeds=(), camera=None):
    from PIL import Image, ImageDraw
    qs, tips, legs = peg_sequence(model, data, site, q_home, hole_top, tool_length, hover=hover, depth=depth,
                                  spiral_r=spiral_r, yaw=yaw, seeds=seeds)
    r = _make_renderer(model, width, height)
    imgs = []
    try:
        n_move = sum(1 for l in legs if l == "move above the hole")
        hole = np.asarray(hole_top, float)
        for i, q in enumerate(qs):
            if camera is None:        # start wide to see the whole arm, then ease in on the hole for the contact phase
                u = min(1.0, i / max(n_move - 1, 1))
                u = u * u * (3 - 2 * u)
                cam_i = dict(lookat=tuple(np.array([0.17, 0.06, 0.14]) * (1 - u) + (hole + [0, 0, 0.02]) * u),
                             distance=0.95 * (1 - u) + 0.34 * u, azimuth=128.0 - 14.0 * u, elevation=-22.0 - 8.0 * u)
            else:
                cam_i = camera
            trail = [t for t, l in zip(tips[: i + 1], legs[: i + 1]) if l != "move above the hole"][-80:]
            img = render_frame(r, model, data, q, tool_length=tool_length, site=site, target=None, trail=trail,
                               hole_top=hole_top, camera=cam_i)
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, width, 24], fill=(30, 33, 40))
            d.text((10, 6), f"reachlens: {legs[i]}   tip z = {tips[i][2] * 1e3:.0f} mm", fill=(235, 238, 245))
            imgs.append(im)
    finally:
        r.close()
    imgs += [imgs[-1]] * 6
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return out, len(imgs)
