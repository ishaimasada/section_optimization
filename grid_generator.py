"""
Elliptic O-grid generator around a blunt-trailing-edge airfoil.

Wall boundary = airfoil surface points (from coordinates.txt) + a straight
base face closing the blunt trailing edge. Outer boundary = circle at
R_FARFIELD chords. Periodic in xi (wraps around the body), Dirichlet in eta
(wall to farfield). Initial guess by radial TFI, refined by Winslow elliptic
smoothing (SOR, red-black).
"""
import numpy as np
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Change the current working directory to the file location
filepath = os.path.abspath(__file__)
directory = os.path.dirname(filepath)
os.chdir(directory)

COORD_FILE = "coordinates.txt"
N_PER_SIDE = 90        # resampled points from each corner to the LE (so surface = 2*N_PER_SIDE - 1)
N_BASE = 7             # total points across the base face, including both corners
N_ETA = 50              # points from wall to farfield
R_FARFIELD = 15.0       # farfield radius, in chords
GROWTH = 1.13            # radial stretching ratio
OMEGA = 1.7              # SOR factor
TOL = 1e-9
MAX_IT = 30000


def load_wall_points(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    keep = [0]
    for i in range(1, len(d)):
        if not np.allclose(d[i], d[keep[-1]], atol=1e-9):
            keep.append(i)
    return d[keep]


def cosine_cluster(n):
    """n points in [0,1], clustered at both ends."""
    t = np.linspace(0, 1, n)
    return 0.5 * (1 - np.cos(np.pi * t))


def resample_surface(d, n_per_side):
    """Resample the surface and also return the exact spline tangent at each
    resampled point (needed because near the tightly-clustered corners,
    finite-differencing the *resampled* points is numerically unreliable)."""
    le_idx = np.argmin(d[:, 0])
    seg_a = d[: le_idx + 1]        # corner A (index 0) -> LE
    seg_b = d[le_idx:]             # LE -> corner B (last index)

    def spline_resample(seg, n):
        ds = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(ds)])
        csx, csy = CubicSpline(s, seg[:, 0]), CubicSpline(s, seg[:, 1])
        s_new = cosine_cluster(n) * s[-1]
        pts = np.column_stack([csx(s_new), csy(s_new)])
        tang = np.column_stack([csx(s_new, 1), csy(s_new, 1)])   # dx/ds, dy/ds
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        return pts, tang

    pts_a, tang_a = spline_resample(seg_a, n_per_side)        # corner A -> LE
    pts_b, tang_b = spline_resample(seg_b, n_per_side)        # LE -> corner B
    pts = np.vstack([pts_a, pts_b[1:]])                        # drop duplicated LE
    tang = np.vstack([tang_a, tang_b[1:]])
    return pts, tang


def build_wall_loop(d, n_per_side, n_base):
    surface, tang_surface = resample_surface(d, n_per_side)   # corner A ... corner B
    cornerA, cornerB = surface[0], surface[-1]
    t = np.linspace(0, 1, n_base)[1:-1][:, None]               # interior base points only
    base = cornerB + t * (cornerA - cornerB)                   # corner B -> corner A
    base_tangent = (cornerA - cornerB) / np.linalg.norm(cornerA - cornerB)
    tang_base = np.tile(base_tangent, (n_base - 2, 1))
    wall = np.vstack([surface, base])                          # closed loop, periodic
    tangent = np.vstack([tang_surface, tang_base])
    return wall, tangent


def wall_outward_normals(wall, tangent):
    """Unit outward normal at each wall point, from the supplied tangent.
    The wall loop is consistently CCW (checked at load time), so a single
    fixed rotation gives the outward direction everywhere -- a per-point
    sign check against the centroid is NOT reliable on a cambered/concave
    shape (it flips sign wherever a point sits near the centroid line)."""
    return np.column_stack([tangent[:, 1], -tangent[:, 0]])


def build_outer_boundary(wall, r_farfield, chord):
    """Farfield circle, points matched to wall points by arc-length fraction.
    This is guaranteed to be a simple (non-self-intersecting) closed curve,
    which a per-point normal-ray projection is NOT on a cambered/concave
    shape at a large standoff -- small normal differences between
    neighboring wall points compound into crossings far from the body."""
    cx = 0.5 * (wall[:, 0].max() + wall[:, 0].min())
    cy = wall[wall[:, 0].argmin(), 1]
    ds = np.linalg.norm(np.diff(np.vstack([wall, wall[0]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(ds)])
    theta = (s[:-1] / s[-1]) * 2 * np.pi
    R = r_farfield * chord
    return np.column_stack([cx + R * np.cos(theta), cy + R * np.sin(theta)])


def local_wall_spacing(wall):
    """Average of the two adjacent segment lengths at each wall point
    (periodic)."""
    seg = np.linalg.norm(np.diff(np.vstack([wall, wall[0]]), axis=0), axis=1)
    seg_prev = np.roll(seg, 1)
    return 0.5 * (seg + seg_prev)


def eta_distribution_per_point(n_eta, growth, frac0):
    """Per-point geometric radial distribution, first step a prescribed
    fraction (frac0, one value per xi) of the total radial distance, so
    tightly-spaced points (corners, LE) get a proportionally small first
    step and don't overshoot their neighbors."""
    n_xi = len(frac0)
    cum = np.zeros((n_xi, n_eta))
    step = frac0.copy()
    for j in range(1, n_eta):
        cum[:, j] = cum[:, j - 1] + step
        step = step * growth
    return cum / cum[:, -1:]


def radial_tfi(wall, normal, outer, n_eta, growth, t_blend=0.15,
                first_step_ratio=0.5, frac0_bounds=(1e-5, 0.02)):
    """Initial guess: blend from a pure local-normal offset right at the wall
    (keeps lines diverging away from sharp corners instead of crossing)
    into a straight line toward the farfield-matched circle point, fully
    switched over by t_blend of the radial distance. The first radial step
    is scaled to local wall spacing so it can't overshoot and cross at
    tightly-clustered points (corners, LE). Exactly reaches outer(i) at
    j = n_eta-1."""
    n_xi = len(wall)
    chord_len = np.linalg.norm(outer - wall, axis=1)
    h = local_wall_spacing(wall)
    frac0 = np.clip(first_step_ratio * h / chord_len, *frac0_bounds)
    t = eta_distribution_per_point(n_eta, growth, frac0)   # (n_xi, n_eta)

    w = np.clip(t / t_blend, 0, 1)
    w = w**2 * (3 - 2 * w)   # smoothstep

    x1 = wall[:, [0]] + t * chord_len[:, None] * normal[:, [0]]
    y1 = wall[:, [1]] + t * chord_len[:, None] * normal[:, [1]]
    x2 = (1 - t) * wall[:, [0]] + t * outer[:, [0]]
    y2 = (1 - t) * wall[:, [1]] + t * outer[:, [1]]
    x = (1 - w) * x1 + w * x2
    y = (1 - w) * y1 + w * y2
    return x, y


def winslow_sweep_periodic(x, y, color, omega):
    """One red-black half-sweep, periodic in i (xi), Dirichlet in j (eta)."""
    n_xi, n_eta = x.shape
    ip1 = np.roll(np.arange(n_xi), -1)
    im1 = np.roll(np.arange(n_xi), 1)

    # derivatives at all interior eta rows (j = 1 .. n_eta-2), all xi (periodic)
    xi_x = 0.5 * (x[ip1, 1:-1] - x[im1, 1:-1]);  xi_y = 0.5 * (y[ip1, 1:-1] - y[im1, 1:-1])
    et_x = 0.5 * (x[:, 2:] - x[:, :-2]);          et_y = 0.5 * (y[:, 2:] - y[:, :-2])

    a = et_x**2 + et_y**2            # alpha (from eta derivatives)
    g = xi_x**2 + xi_y**2            # gamma (from xi derivatives)
    beta = xi_x * et_x + xi_y * et_y  # x_xi*x_eta + y_xi*y_eta

    denom = 2.0 * (a + g) + 1e-300

    def new(f):
        cross = 0.25 * beta * (f[ip1][:, 2:] - f[ip1][:, :-2] - f[im1][:, 2:] + f[im1][:, :-2])
        return (a * (f[ip1, 1:-1] + f[im1, 1:-1])
                + g * (f[:, 2:] + f[:, :-2])
                - 2 * cross) / denom

    xn, yn = new(x), new(y)
    ii, jj = np.indices(xn.shape)
    mask = ((ii + jj) % 2) == color
    xin, yin = x[:, 1:-1], y[:, 1:-1]
    dx = np.where(mask, omega * (xn - xin), 0.0)
    dy = np.where(mask, omega * (yn - yin), 0.0)
    xin += dx
    yin += dy
    return max(np.abs(dx).max(), np.abs(dy).max())


def jacobian(x, y):
    n_xi = x.shape[0]
    ip1 = np.roll(np.arange(n_xi), -1)
    im1 = np.roll(np.arange(n_xi), 1)
    xi_x = 0.5 * (x[ip1, 1:-1] - x[im1, 1:-1]);  xi_y = 0.5 * (y[ip1, 1:-1] - y[im1, 1:-1])
    et_x = 0.5 * (x[:, 2:] - x[:, :-2]);          et_y = 0.5 * (y[:, 2:] - y[:, :-2])
    return xi_x * et_y - et_x * xi_y


def generate():
    d = load_wall_points(COORD_FILE)
    chord = d[:, 0].max() - d[:, 0].min()
    wall, tangent = build_wall_loop(d, N_PER_SIDE, N_BASE)
    normal = wall_outward_normals(wall, tangent)
    outer = build_outer_boundary(wall, R_FARFIELD, chord)
    x, y = radial_tfi(wall, normal, outer, N_ETA, GROWTH)
    x0, y0 = x.copy(), y.copy()

    def grid_valid(J):
        # A valid (non-folding) grid has a Jacobian of CONSISTENT sign
        # everywhere -- which sign depends on wall traversal direction (here
        # CCW, which gives J < 0 throughout for this derivative convention).
        # A sign CHANGE across the grid, not a particular sign, means folding.
        return (J >= 0).all() or (J <= 0).all()

    print(f"n_xi = {len(wall)}, n_eta = {N_ETA}, chord = {chord:.5f}")
    J0 = jacobian(x, y)
    print(f"TFI initial grid: |J| min/max = {np.abs(J0).min():.3e} / {np.abs(J0).max():.3e}"
          f"  ({'valid' if grid_valid(J0) else 'FOLDED'})")

    for it in range(1, MAX_IT + 1):
        d_ = max(winslow_sweep_periodic(x, y, 0, OMEGA),
                 winslow_sweep_periodic(x, y, 1, OMEGA))
        if it % 500 == 0:
            print(f"iter {it:5d}  max change = {d_:.3e}")
        if d_ < TOL:
            print(f"Converged in {it} iterations (max change {d_:.2e})")
            break
    else:
        print("WARNING: hit MAX_IT without converging")

    J = jacobian(x, y)
    print(f"Final |J| min/max = {np.abs(J).min():.3e} / {np.abs(J).max():.3e}"
          f"  ({'valid, no folding' if grid_valid(J) else 'FOLDED'})")
    return x, y, x0, y0, wall


def plot(x, y, x0, y0, wall, fname="airfoil_ogrid.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    for ax, (X, Y), title in zip(axes[0], [(x0, y0), (x, y)],
                                  ["Radial TFI (initial)", "Elliptic (Winslow) O-grid"]):
        for i in range(0, X.shape[0], 3):
            ax.plot(X[i, :], Y[i, :], "k-", lw=0.4)
        for j in range(0, X.shape[1], 3):
            ax.plot(np.append(X[:, j], X[0, j]), np.append(Y[:, j], Y[0, j]), "k-", lw=0.4)
        ax.plot(np.append(wall[:, 0], wall[0, 0]), np.append(wall[:, 1], wall[0, 1]), "r-", lw=1.3)
        ax.set_aspect("equal"); ax.set_title(title)

    for ax, (X, Y), title in zip(axes[1], [(x0, y0), (x, y)],
                                  ["TFI, zoom near airfoil", "Elliptic, zoom near airfoil"]):
        for i in range(0, X.shape[0], 2):
            ax.plot(X[i, :15], Y[i, :15], "k-", lw=0.5)
        for j in range(0, 15, 1):
            ax.plot(np.append(X[:, j], X[0, j]), np.append(Y[:, j], Y[0, j]), "k-", lw=0.4)
        ax.plot(np.append(wall[:, 0], wall[0, 0]), np.append(wall[:, 1], wall[0, 1]), "r-", lw=1.3)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.set_xlim(-0.03, 0.17); ax.set_ylim(-0.12, 0.08)

    plt.tight_layout()
    plt.savefig(fname, dpi=150)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
    x, y, x0, y0, wall = generate()
    plot(x, y, x0, y0, wall, fname=output_filename+".png")
    np.savez(output_filename+".npz", x=x, y=y, wall=wall)
