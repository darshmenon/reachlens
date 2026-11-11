"""Locating and loading arm models. Nothing is bundled: models are found on disk, or passed with --xml."""
import os
from dataclasses import dataclass

import mujoco

_HOME = os.path.expanduser("~")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    env_var: str
    candidates: tuple
    rel: tuple = ()                 # where the MJCF sits under a directory listed in $REACHLENS_MODELS
    site: str = "attachment_site"
    default_range: float = 0.6      # half-width (m) of the default xy map
    default_z: float = 0.3


ARMS = {
    "ur3": ArmSpec("ur3", "REACHLENS_UR3_XML",
                   (os.path.join(_HOME, "UR3_ROS2_PICK_AND_PLACE/ur_rl_training/models/ur3/ur3.xml"),),
                   rel=("ur3.xml", "ur3/ur3.xml", "models/ur3/ur3.xml", "ur_rl_training/models/ur3/ur3.xml",
                        "UR3_ROS2_PICK_AND_PLACE/ur_rl_training/models/ur3/ur3.xml"),
                   default_range=0.55, default_z=0.25),
    "ur5e": ArmSpec("ur5e", "REACHLENS_UR5E_XML",
                    (os.path.join(os.environ.get("MUJOCO_MENAGERIE_PATH", os.path.join(_HOME, "mujoco_menagerie")),
                                  "universal_robots_ur5e/ur5e.xml"),
                     os.path.join(_HOME, "mujoco_menagerie/universal_robots_ur5e/ur5e.xml")),
                    rel=("ur5e.xml", "ur5e/ur5e.xml", "universal_robots_ur5e/ur5e.xml",
                         "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"),
                    default_range=0.95, default_z=0.3),
}


class ArmNotFound(FileNotFoundError):
    pass


def _model_dirs():
    return [d for d in os.environ.get("REACHLENS_MODELS", "").split(os.pathsep) if d]


def resolve_xml(arm, xml=None):
    """--xml wins, then $REACHLENS_<ARM>_XML, then the directories in $REACHLENS_MODELS, then the known locations."""
    if xml:
        if not os.path.isfile(xml):
            raise ArmNotFound(f"model file not found: {xml}")
        return xml
    spec = ARMS[arm]
    env = os.environ.get(spec.env_var)
    tried = []
    for p in ([env] if env else []):
        tried.append(p)
        if os.path.isfile(p):
            return p
    for d in _model_dirs():
        for r in spec.rel:
            p = os.path.join(d, r)
            tried.append(p)
            if os.path.isfile(p):
                return p
    for p in spec.candidates:
        tried.append(p)
        if os.path.isfile(p):
            return p
    shown = tried if len(tried) <= 6 else tried[:6] + [f"... and {len(tried) - 6} more"]
    raise ArmNotFound(f"no MJCF found for {arm}. Fix it with one of: --xml PATH, ${spec.env_var}=PATH, or "
                      f"$REACHLENS_MODELS=DIR (a directory holding e.g. {spec.rel[2] if len(spec.rel) > 2 else spec.rel[0]}). "
                      f"Looked at: {', '.join(shown)}")


def load_arm(arm="ur3", xml=None, site=None):
    """Returns (model, data, site_id, qpos0) for a 6-joint arm with a tool site."""
    path = xml if (xml and arm not in ARMS) else resolve_xml(arm, xml)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    name = site or (ARMS[arm].site if arm in ARMS else "attachment_site")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"site '{name}' not found in {path}")
    q0 = model.key_qpos[0][: model.nv].copy() if model.nkey else model.qpos0.copy()
    return model, data, sid, q0
