"""Heatmap rendering for a reachability slice."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_slice(res, arm, out, summ=None, marks=()):
    xs, ys, g = res["xs"], res["ys"], res["grids"]
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    panels = [("IK success", g["reachable"], "Greens", 0, 1, "reachable (1) / not (0)"),
              ("Manipulability (Yoshikawa)", g["manipulability"], "viridis", None, None, "sqrt(det(J J^T))"),
              ("Joint-limit margin", g["limit_margin"], "magma", 0, 1, "0 on a limit, 1 mid-range")]
    for a, (title, arr, cmap, lo, hi, lab) in zip(ax, panels):
        a.set_facecolor("#d9d9d9")
        im = a.imshow(np.ma.masked_invalid(arr) if title != "IK success" else arr, origin="lower", extent=ext,
                      cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        a.plot(0, 0, "k^", ms=9)
        for (mx, my, label) in marks:
            a.plot(mx, my, marker="*", color="#e4572e", mec="k", ms=15, ls="")
            a.annotate(label, (mx, my), textcoords="offset points", xytext=(8, 8), fontsize=9, color="#e4572e",
                       fontweight="bold")
        a.set_title(title)
        a.set_xlabel("x (m)")
        a.set_aspect("equal")
        fig.colorbar(im, ax=a, shrink=0.8, label=lab)
    ax[0].set_ylabel("y (m)")
    s = f"  |  {summ['success_rate']:.0%} of {summ['points']} poses reachable" if summ else ""
    fig.suptitle(f"{arm.upper()} tool-down reach at z = {res['z']:.2f} m{s}   (grey = unreachable, ^ = base)")
    fig.savefig(out, dpi=110)
    plt.close(fig)
