"""The optional C++ refine loop must give the same answer as the Python one, call by call."""
import numpy as np
import pytest

from reachlens import ik, load_arm, map_slice
from reachlens.arms import ArmNotFound, resolve_xml

try:
    resolve_xml("ur3")
    HAVE_UR3 = True
except ArmNotFound:
    HAVE_UR3 = False

pytestmark = [
    pytest.mark.skipif(not ik.have_cpp(), reason="C++ extension not built (pip install pybind11, then rebuild)"),
    pytest.mark.skipif(not HAVE_UR3, reason="UR3 MJCF not found (set REACHLENS_UR3_XML)"),
]


def _targets(n, seed):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(0.0, 0.5)) for _ in range(n)]


def test_refine_matches_python_per_call():
    m, d, s, q0 = load_arm("ur3")
    rng = np.random.default_rng(1)
    Rd = ik.tool_down()
    for pos in _targets(60, 2):
        seed = rng.uniform(-np.pi, np.pi, 6)
        qc, itc, ec = ik._refine(m, d, s, seed.copy(), np.asarray(pos), Rd, 150, 0.05, 0.3)
        qp, itp, ep = ik._refine_py(m, d, s, seed.copy(), np.asarray(pos), Rd, 150, 0.05, 0.3)
        assert itc == itp, pos
        assert np.allclose(qc, qp, atol=1e-6), pos
        assert np.allclose(ec, ep, atol=1e-6), pos


def test_map_is_identical_with_and_without_cpp(monkeypatch):
    m, d, s, q0 = load_arm("ur3")
    fast = map_slice(m, d, s, 0.25, half=0.4, n=9, q_home=q0)
    monkeypatch.setattr(ik, "_fast", None)
    slow = map_slice(m, d, s, 0.25, half=0.4, n=9, q_home=q0)
    assert np.array_equal(fast["grids"]["reachable"], slow["grids"]["reachable"])
    assert np.allclose(np.nan_to_num(fast["grids"]["manipulability"]), np.nan_to_num(slow["grids"]["manipulability"]), atol=1e-6)


def test_bad_arguments_raise_instead_of_crashing():
    m, d, s, q0 = load_arm("ur3")
    with pytest.raises(Exception):
        ik._fast.refine(m._address, d._address, 10_000, q0, np.zeros(3), np.eye(3), 10, 0.05, 0.3, 25)
    with pytest.raises(Exception):
        ik._fast.refine(m._address, d._address, s, q0, np.zeros(2), np.eye(3), 10, 0.05, 0.3, 25)
