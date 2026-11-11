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
    site: str = "attachment_site"
    default_range: float = 0.6      # half-width (m) of the default xy map
    default_z: float = 0.3


ARMS = {
    "ur3": ArmSpec("ur3", "REACHLENS_UR3_XML",
                   (os.path.join(_HOME, "UR3_ROS2_PICK_AND_PLACE/ur_rl_training/models/ur3/ur3.xml"),),
                   default_range=0.55, default_z=0.25),
    "ur5e": ArmSpec("ur5e", "REACHLENS_UR5E_XML",
                    (os.path.join(os.environ.get("MUJOCO_MENAGERIE_PATH", os.path.join(_HOME, "mujoco_menagerie")),
                                  "universal_robots_ur5e/ur5e.xml"),
                     os.path.join(_HOME, "mujoco_menagerie/universal_robots_ur5e/ur5e.xml")),
                    default_range=0.95, default_z=0.3),
}


class ArmNotFound(FileNotFoundError):
    pass


def resolve_xml(arm, xml=None):
    """--xml wins, then $REACHLENS_<ARM>_XML, then the known locations."""
    if xml:
        if not os.path.isfile(xml):
            raise ArmNotFound(f"model file not found: {xml}")
        return xml
    spec = ARMS[arm]
    env = os.environ.get(spec.env_var)
    for p in ([env] if env else []) + list(spec.candidates):
        if p and os.path.isfile(p):
            return p
    raise ArmNotFound(f"no MJCF found for {arm}; pass --xml or set ${spec.env_var} "
                      f"(looked in: {', '.join(spec.candidates)})")


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
