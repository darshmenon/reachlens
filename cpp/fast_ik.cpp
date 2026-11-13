// Optional C++ core for reachlens: the damped-least-squares refine loop of reachlens/ik.py (_refine), through the
// MuJoCo C API on the same mjModel / mjData the Python bindings own. Same algorithm and same stopping rules as the
// Python version, so the two can be compared call by call (tests/test_fast.py).
#include <mujoco/mujoco.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// Solve A x = b for a symmetric positive definite 6x6 A (Cholesky). A is J J^T + damping^2 I, so it is always SPD.
static void solve_spd6(const double A[6][6], const double b[6], double x[6]) {
  double L[6][6] = {};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j <= i; ++j) {
      double s = A[i][j];
      for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
      L[i][j] = (i == j) ? std::sqrt(std::max(s, 1e-300)) : s / L[j][j];
    }
  }
  double y[6];
  for (int i = 0; i < 6; ++i) {
    double s = b[i];
    for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  for (int i = 5; i >= 0; --i) {
    double s = y[i];
    for (int k = i + 1; k < 6; ++k) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
}

static void pose_error(const mjtNum * R, const double * Rd, const mjtNum * p, const double * pd, double e[6]) {
  // M = R Rd^T, e_o = 0.5 * [M12 - M21, M20 - M02, M01 - M10]
  double M[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += R[3 * i + k] * Rd[3 * j + k];
      M[i][j] = s;
    }
  for (int i = 0; i < 3; ++i) e[i] = pd[i] - p[i];
  e[3] = 0.5 * (M[1][2] - M[2][1]);
  e[4] = 0.5 * (M[2][0] - M[0][2]);
  e[5] = 0.5 * (M[0][1] - M[1][0]);
}

static double norm3(const double * v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::tuple refine(std::uintptr_t model_addr, std::uintptr_t data_addr, int site, Arr q0, Arr pd, Arr Rd, int iters,
                 double damping, double step, int patience) {
  const mjModel * m = reinterpret_cast<const mjModel *>(model_addr);
  mjData * d = reinterpret_cast<mjData *>(data_addr);
  if (!m || !d) throw std::invalid_argument("refine: null model or data");
  if (q0.size() < 6 || pd.size() != 3 || Rd.size() != 9) throw std::invalid_argument("refine: bad argument shapes");
  if (m->nv < 6 || m->njnt < 6) throw std::invalid_argument("refine: model needs at least 6 joints");
  if (site < 0 || site >= m->nsite) throw std::invalid_argument("refine: bad site id");
  const double * Rdp = Rd.data();
  const double * pdp = pd.data();
  double q[6], lo[6], hi[6], e[6];
  bool lim[6];
  for (int i = 0; i < 6; ++i) {
    q[i] = q0.data()[i];
    lo[i] = m->jnt_range[2 * i];
    hi[i] = m->jnt_range[2 * i + 1];
    lim[i] = m->jnt_limited[i] != 0;
  }
  const int nv = m->nv;
  std::vector<mjtNum> jp(3 * nv), jr(3 * nv);
  double best_e = INFINITY;
  int since = 0, it = 0;
  const double damp2 = damping * damping;
  auto eval = [&]() {
    for (int i = 0; i < 6; ++i) d->qpos[i] = q[i];
    mj_kinematics(m, d);
    mj_comPos(m, d);
    pose_error(d->site_xmat + 9 * site, Rdp, d->site_xpos + 3 * site, pdp, e);
  };
  bool converged = false;
  {
    py::gil_scoped_release release;
    for (it = 1; it <= iters; ++it) {
      eval();
      if (norm3(e) < 1e-4 && norm3(e + 3) < 1e-3) {
        converged = true;
        break;
      }
      const double ne = norm3(e) + 0.1 * norm3(e + 3);
      if (ne < best_e - 1e-5) {
        best_e = ne;
        since = 0;
      } else if (++since >= patience) {
        break;   // stalled at a limit or a singularity
      }
      mj_jacSite(m, d, jp.data(), jr.data(), site);
      double J[6][6];
      for (int c = 0; c < 6; ++c)
        for (int r = 0; r < 3; ++r) {
          J[r][c] = jp[r * nv + c];
          J[r + 3][c] = jr[r * nv + c];
        }
      double A[6][6];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
          double s = 0;
          for (int k = 0; k < 6; ++k) s += J[i][k] * J[j][k];
          A[i][j] = s + (i == j ? damp2 : 0.0);
        }
      double w[6], dq[6], n = 0;
      solve_spd6(A, e, w);
      for (int c = 0; c < 6; ++c) {
        double s = 0;
        for (int r = 0; r < 6; ++r) s += J[r][c] * w[r];
        dq[c] = s;
        n = std::max(n, std::fabs(s));
      }
      const double scale = n > step ? step / n : 1.0;
      for (int c = 0; c < 6; ++c) {
        q[c] += dq[c] * scale;
        if (lim[c]) q[c] = std::min(std::max(q[c], lo[c]), hi[c]);
      }
    }
    if (!converged) {
      if (it > iters) it = iters;
      eval();   // error at the final q, like the Python version
    }
  }
  py::array_t<double> qo(6), eo(6);
  std::copy(q, q + 6, qo.mutable_data());
  std::copy(e, e + 6, eo.mutable_data());
  return py::make_tuple(qo, it, eo);
}

PYBIND11_MODULE(_fast, mod) {
  mod.doc() = "C++ refine loop for reachlens (MuJoCo C API)";
  mod.def("refine", &refine, py::arg("model_addr"), py::arg("data_addr"), py::arg("site"), py::arg("q0"), py::arg("pd"),
          py::arg("Rd"), py::arg("iters"), py::arg("damping"), py::arg("step"), py::arg("patience"));
}
