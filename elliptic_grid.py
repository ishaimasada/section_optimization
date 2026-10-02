"""
Elliptic (Winslow / Laplace) grid generator on a 100x100 point grid.

Domain: unit square whose bottom edge is bulged by BUMP (set BUMP = 0 for a flat square).
Steps: boundary points -> transfinite interpolation (TFI) initial guess
       -> red-black SOR on the Winslow equations -> Jacobian check -> plot.

Indexing: x[i, j], i along xi (bottom/top edges), j along eta (left/right edges).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 100          # points in each direction
BUMP = 0.25      # bottom-edge bulge (0 -> flat square)
OMEGA = 1.6      # SOR relaxation factor (keep < 2; lower if it diverges)
TOL = 1e-10
MAX_IT = 20000


def boundary_points(n):
    """Return the four edges as (n, 2) arrays. Corners must match exactly."""
    s = np.linspace(0.0, 1.0, n)
    bottom = np.column_stack([s, -BUMP * np.sin(np.pi * s)])   # j = 0
    top    = np.column_stack([s, np.ones(n)])                  # j = n-1
    left   = np.column_stack([np.zeros(n), s])                 # i = 0
    right  = np.column_stack([np.ones(n), s])                  # i = n-1
    # left/right edges run from bottom corner to top corner
    left[:, 1]  = bottom[0, 1]  + s * (top[0, 1]  - bottom[0, 1])
    right[:, 1] = bottom[-1, 1] + s * (top[-1, 1] - bottom[-1, 1])
    return bottom, top, left, right


def tfi(bottom, top, left, right):
    """Transfinite interpolation initial grid."""
    n = len(bottom)
    s = np.linspace(0, 1, n)[:, None]   # xi
    t = np.linspace(0, 1, n)[None, :]   # eta
    P = np.zeros((n, n, 2))
    for k in range(2):
        B, T = bottom[:, k][:, None], top[:, k][:, None]
        L, R = left[:, k][None, :], right[:, k][None, :]
        P[..., k] = ((1 - t) * B + t * T + (1 - s) * L + s * R
                     - ((1 - s) * (1 - t) * bottom[0, k]
                        + s * (1 - t) * bottom[-1, k]
                        + (1 - s) * t * top[0, k]
                        + s * t * top[-1, k]))
    return P[..., 0], P[..., 1]


def winslow_update(x, y, color, omega):
    """One red-black half sweep of the Winslow equations (in place)."""
    xi = 0.5 * (x[2:, 1:-1] - x[:-2, 1:-1]);  yi = 0.5 * (y[2:, 1:-1] - y[:-2, 1:-1])
    xe = 0.5 * (x[1:-1, 2:] - x[1:-1, :-2]);  ye = 0.5 * (y[1:-1, 2:] - y[1:-1, :-2])
    a = xe**2 + ye**2          # alpha
    b = xi * xe + yi * ye      # beta
    g = xi**2 + yi**2          # gamma
    denom = 2.0 * (a + g) + 1e-300

    def new(f):
        return (a * (f[2:, 1:-1] + f[:-2, 1:-1])
                + g * (f[1:-1, 2:] + f[1:-1, :-2])
                - 0.5 * b * (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2])) / denom

    xn, yn = new(x), new(y)
    ii, jj = np.indices(xn.shape)
    mask = ((ii + jj) % 2) == color
    xin, yin = x[1:-1, 1:-1], y[1:-1, 1:-1]      # views into x, y
    dx = np.where(mask, omega * (xn - xin), 0.0)
    dy = np.where(mask, omega * (yn - yin), 0.0)
    xin += dx
    yin += dy
    return max(np.abs(dx).max(), np.abs(dy).max())


def jacobian(x, y):
    xi = 0.5 * (x[2:, 1:-1] - x[:-2, 1:-1]);  yi = 0.5 * (y[2:, 1:-1] - y[:-2, 1:-1])
    xe = 0.5 * (x[1:-1, 2:] - x[1:-1, :-2]);  ye = 0.5 * (y[1:-1, 2:] - y[1:-1, :-2])
    return xi * ye - xe * yi


def generate(n=N):
    x, y = tfi(*boundary_points(n))
    print(f"TFI initial grid: min Jacobian = {jacobian(x, y).min():.3e}")
    for it in range(1, MAX_IT + 1):
        d = max(winslow_update(x, y, 0, OMEGA), winslow_update(x, y, 1, OMEGA))
        if it % 200 == 0:
            print(f"iter {it:5d}  max change = {d:.3e}")
        if d < TOL:
            print(f"Converged in {it} iterations (max change {d:.2e})")
            break
    else:
        print("WARNING: hit MAX_IT without converging")
    Jmin = jacobian(x, y).min()
    print(f"Final min Jacobian = {Jmin:.3e}  ({'valid' if Jmin > 0 else 'FOLDED'})")
    return x, y


def plot(x, y, x0, y0, stride=5, fname="elliptic_grid.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (X, Y), title in zip(axes, [(x0, y0), (x, y)],
                                 ["TFI initial grid", "Elliptic (Winslow) grid"]):
        for i in range(0, X.shape[0], stride):
            ax.plot(X[i, :], Y[i, :], "k-", lw=0.5)
        for j in range(0, X.shape[1], stride):
            ax.plot(X[:, j], Y[:, j], "k-", lw=0.5)
        ax.plot(X[:, 0], Y[:, 0], "r-", lw=1.5)   # bulged edge
        ax.set_aspect("equal"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)


x0, y0 = tfi(*boundary_points(N))
x, y = generate(N)
plot(x, y, x0, y0)
np.savez("elliptic_grid.npz", x=x, y=y)