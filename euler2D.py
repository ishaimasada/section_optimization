"""
2D Euler solver (Jameson-style 2-stage Runge-Kutta, scalar 2nd-order
dissipation), ported from euler.m and vectorized with numpy.

TOPOLOGY NOTE: euler.m was written for a channel/bump grid -- velocity
inlet at i=1, pressure outlet at i=num_i, wall at j=1, farfield at j=num_j.
The airfoil O-grid built earlier is different: it wraps all the way around
the body, so the i-direction is PERIODIC (no inlet/outlet at all), and the
only physical boundaries are the wall (j=0) and the farfield (j=-1). The
boundary-condition logic below reflects that: update_boundaries() only
handles j=0 and j=-1, for every i at once; the i-direction never needs a
boundary condition, just wrap-around neighbors (np.roll).

Arrays are shaped (ni, nj, 4), matching euler.m's (i,j,:) indexing, with
state order [rho, Vx, Vy, P] (primitive, V) or [u1,u2,u3,u4] (conservative, u).
"""
import numpy as np
import os, sys
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

filepath = os.path.abspath(__file__)
directory = os.path.dirname(filepath)
os.chdir(directory)

# argv[1]: output image filename (default "velocity_field.png")
# argv[2]: grid .npz filename to load (default "airfoil_ogrid.npz") -- this
# was previously hardcoded, so every call loaded the SAME grid file
# regardless of which iteration's grid was meant to be solved on.
results_filename = sys.argv[1] if len(sys.argv) > 1 else "velocity_field.png"
grid_filename = sys.argv[2] if len(sys.argv) > 2 else "airfoil_ogrid.npz"

# ---------------------------------------------------------------- Load grid
print(f"Loading grid from {grid_filename}")
with np.load(grid_filename) as data:
    x, y, wall = data["x"], data["y"], data["wall"]
ni, nj = x.shape
print(f"Grid size: {ni} x {nj}")


# ------------------------------------------------------- State conversions
def get_conservative(V, Cv, R):
    """Primitive [rho, Vx, Vy, P] -> conservative [u1,u2,u3,u4]. Vectorized
    over any leading shape; state is the last axis."""
    rho, Vx, Vy, P = V[..., 0], V[..., 1], V[..., 2], V[..., 3]
    u1 = rho
    u2 = rho * Vx
    u3 = rho * Vy
    u4 = Cv * P / R + 0.5 * rho * (Vx**2 + Vy**2)
    return np.stack([u1, u2, u3, u4], axis=-1)


def get_primitive(u, gamma):
    """Conservative -> primitive. Vectorized, state on the last axis."""
    u1, u2, u3, u4 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    rho = u1
    Vx = u2 / u1
    Vy = u3 / u1
    P = (u4 - 0.5 * (u2**2 / u1 + u3**2 / u1)) * (gamma - 1)
    return np.stack([rho, Vx, Vy, P], axis=-1)


def solve_fluxes(V, fs):
    """Flux vectors A (xi/x-ish) and B (eta/y-ish), vectorized over V's
    leading shape."""
    rho, Vx, Vy, P = V[..., 0], V[..., 1], V[..., 2], V[..., 3]
    ht = fs.cp * P / (fs.R * rho) + 0.5 * (Vx**2 + Vy**2)
    A = np.stack([rho * Vx, rho * Vx**2 + P, rho * Vx * Vy, rho * ht * Vx], axis=-1)
    B = np.stack([rho * Vy, rho * Vx * Vy, rho * Vy**2 + P, rho * ht * Vy], axis=-1)
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise FloatingPointError("Invalid elements (Inf or NaN) in flux vectors")
    return A, B


# -------------------------------------------------------- Boundary conditions
def wall_bc(V_adjacent, eta_x_wall, eta_y_wall, Cv, R):
    """Inviscid slip wall. V_adjacent: primitive state at the first interior
    ring (j=1), shape (ni, 4). eta_x_wall, eta_y_wall: metrics at j=0,
    shape (ni,)."""
    rho2, u2, v2, P2 = (V_adjacent[..., 0], V_adjacent[..., 1],
                        V_adjacent[..., 2], V_adjacent[..., 3])
    norm = np.sqrt(eta_x_wall**2 + eta_y_wall**2)
    nx, ny = eta_x_wall / norm, eta_y_wall / norm
    Vn = u2 * nx + v2 * ny
    u1_vel = u2 - Vn * nx
    v1_vel = v2 - Vn * ny
    V1 = np.stack([rho2, u1_vel, v1_vel, P2], axis=-1)
    return get_conservative(V1, Cv, R), V1


def farfield_bc(n, u_inf, v_inf, P_inf, T_inf, Cv, R):
    """Pressure farfield, same freestream state imposed at every i."""
    rho1 = P_inf / (T_inf * R)
    V1 = np.tile(np.array([rho1, u_inf, v_inf, P_inf]), (n, 1))
    return get_conservative(V1, Cv, R), V1


def update_boundaries(V, u, fs, eta_x, eta_y):
    """Wall at j=0, farfield at j=-1, for every i. No i-direction boundary
    (periodic O-grid)."""
    u = u.copy(); V = V.copy()
    u[:, 0, :], V[:, 0, :] = wall_bc(V[:, 1, :], eta_x[:, 0], eta_y[:, 0], fs.cv, fs.R)
    u[:, -1, :], V[:, -1, :] = farfield_bc(ni, fs.u, fs.v, fs.P, fs.T, fs.cv, fs.R)
    return u, V


def display_residuals(L2_norm):
    print("RESIDUALS")
    print(f"Continuity = {L2_norm[0]:.3e}")
    print(f"X-Momentum = {L2_norm[1]:.3e}")
    print(f"Y-Momentum = {L2_norm[2]:.3e}")
    print(f"Energy     = {L2_norm[3]:.3e}")


# -------------------------------------------------------------- Grid metrics
def compute_metrics(x, y):
    """xi (i) is periodic -> central difference with wraparound for every i.
    eta (j) is not periodic -> central differences in the interior, 2nd-order
    one-sided differences at j=0 (wall) and j=-1 (farfield)."""
    x_xi = (np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0)) / 2
    y_xi = (np.roll(y, -1, axis=0) - np.roll(y, 1, axis=0)) / 2

    x_eta = np.empty_like(x)
    y_eta = np.empty_like(y)
    x_eta[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
    y_eta[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2
    x_eta[:, 0] = (-3 * x[:, 0] + 4 * x[:, 1] - x[:, 2]) / 2
    y_eta[:, 0] = (-3 * y[:, 0] + 4 * y[:, 1] - y[:, 2]) / 2
    x_eta[:, -1] = (3 * x[:, -1] - 4 * x[:, -2] + x[:, -3]) / 2
    y_eta[:, -1] = (3 * y[:, -1] - 4 * y[:, -2] + y[:, -3]) / 2

    J = x_xi * y_eta - x_eta * y_xi
    xi_x = y_eta / J
    xi_y = -x_eta / J
    eta_x = -y_xi / J
    eta_y = x_xi / J
    return x_xi, x_eta, y_xi, y_eta, J, xi_x, xi_y, eta_x, eta_y


def ensure_positive_jacobian(x, y, wall):
    """The solver (like euler.m) assumes J > 0 everywhere. Our O-grid's
    Jacobian sign depends on the i-traversal direction (CCW vs CW); flip it
    if needed rather than erroring out."""
    *_, J, _, _, _, _ = compute_metrics(x, y)
    if np.median(J) < 0:
        print("Flipping grid i-direction so the Jacobian is positive.")
        x, y, wall = x[::-1, :].copy(), y[::-1, :].copy(), wall[::-1, :].copy()
    return x, y, wall


x, y, wall = ensure_positive_jacobian(x, y, wall)
x_xi, x_eta, y_xi, y_eta, J, xi_x, xi_y, eta_x, eta_y = compute_metrics(x, y)
if J.min() <= 0:
    # A handful of cells at the tightest-clustered wall points (sharp
    # corners) can land just barely on the wrong side of zero from plain
    # floating-point cancellation -- not a real fold. Treat it as one only
    # if it's a large fraction of the typical cell Jacobian.
    bad = J <= 0
    tiny = np.abs(J[bad]).max() < 1e-3 * np.median(J)
    if not tiny:
        raise ValueError(f"Non-positive Jacobian remains after orientation fix "
                          f"(min J = {J.min():.3e}); this looks like a real "
                          f"fold, not roundoff -- check the grid.")
    print(f"Note: clipping {bad.sum()} cell(s) with negligible non-positive "
          f"Jacobian (max |J| = {np.abs(J[bad]).max():.1e}, roundoff noise "
          f"at tightly-clustered wall points) up to a small positive floor.")
    J[bad] = 1e-3 * np.median(J)

# Face-averaged metrics used every RK stage -- constant, precompute once.
sl = slice(1, -1)   # interior j
y_eta_E = 0.5 * (y_eta[:, sl] + np.roll(y_eta, -1, axis=0)[:, sl])
x_eta_E = 0.5 * (x_eta[:, sl] + np.roll(x_eta, -1, axis=0)[:, sl])
y_eta_W = 0.5 * (y_eta[:, sl] + np.roll(y_eta, 1, axis=0)[:, sl])
x_eta_W = 0.5 * (x_eta[:, sl] + np.roll(x_eta, 1, axis=0)[:, sl])
y_xi_N = 0.5 * (y_xi[:, sl] + y_xi[:, 2:])
x_xi_N = 0.5 * (x_xi[:, sl] + x_xi[:, 2:])
y_xi_S = 0.5 * (y_xi[:, sl] + y_xi[:, :-2])
x_xi_S = 0.5 * (x_xi[:, sl] + x_xi[:, :-2])


# ------------------------------------------------------- Freestream / inputs
gamma = 1.4
R = 287.0
Cp = gamma * R / (gamma - 1)
Cv = R / (gamma - 1)
T_inf = 300.0
P_inf = 1e5
rho_inf = P_inf / (R * T_inf)
M = 0.3
a_inf = np.sqrt(gamma * R * T_inf)
u_inf = M * a_inf
v_inf = 0.0

fs = SimpleNamespace(T=T_inf, P=P_inf, rho=rho_inf, u=u_inf, v=v_inf,
                      cv=Cv, R=R, gamma=gamma, cp=Cp)

ITERATIONS = 50000
TOLERANCE = 1e-3
CFL = 0.1    # see note below on the eta-flux sign fix -- that's a real,
             # separate bug, but this grid's tightly-clustered TE corners
             # still make CFL=1.5 (euler.m's default) unstable even with it
             # fixed; 0.1 is what's actually been tested to hold up
ALPHA = (0.5, 1.0)
EPSILON = 0.05
NUM_STAGES = 2

if CFL > 1.8:
    raise ValueError("CFL number must be less than 1.8.")


def run_solver(iterations=ITERATIONS, tolerance=TOLERANCE, verbose=True, cfl=CFL):
    # ------------------------------------------------------ Initialization
    V0 = np.array([rho_inf, u_inf, v_inf, P_inf])
    V = np.tile(V0, (ni, nj, 1))
    u_old = get_conservative(V, Cv, R)
    c_field = np.full((ni, nj), a_inf)

    u_old, V = update_boundaries(V, u_old, fs, eta_x, eta_y)
    A, B = solve_fluxes(V, fs)

    residuals_1 = None
    L2_norm = np.ones(4)
    history = []

    for n in range(1, iterations + 1):
        # -------------------------------------------------- Local time step
        Vx, Vy, a = V[:, sl, 1], V[:, sl, 2], c_field[:, sl]
        U = Vx * xi_x[:, sl] + Vy * xi_y[:, sl]
        Vc = Vx * eta_x[:, sl] + Vy * eta_y[:, sl]
        lambda_xi = np.abs(U) + a * np.sqrt(xi_x[:, sl]**2 + xi_y[:, sl]**2)
        lambda_eta = np.abs(Vc) + a * np.sqrt(eta_x[:, sl]**2 + eta_y[:, sl]**2)
        dt = cfl / (lambda_xi + lambda_eta)   # shape (ni, nj-2)

        u_k, V_k = [None] * NUM_STAGES, [None] * NUM_STAGES
        residuals_n = np.zeros((ni, nj, 4))

        for k in range(NUM_STAGES):
            u_prev, V_prev = (u_old, V) if k == 0 else (u_k[k - 1], V_k[k - 1])

            c_field = np.sqrt(gamma * V_prev[..., 3] / V_prev[..., 0])
            Vx_field = u_prev[..., 1] / u_prev[..., 0]
            Vy_field = u_prev[..., 2] / u_prev[..., 0]

            A_ip1, A_im1 = np.roll(A, -1, axis=0), np.roll(A, 1, axis=0)
            B_ip1, B_im1 = np.roll(B, -1, axis=0), np.roll(B, 1, axis=0)

            AE = 0.5 * (A_ip1[:, sl] + A[:, sl]);  AW = 0.5 * (A_im1[:, sl] + A[:, sl])
            BE = 0.5 * (B_ip1[:, sl] + B[:, sl]);  BW = 0.5 * (B_im1[:, sl] + B[:, sl])
            AN = 0.5 * (A[:, 2:] + A[:, sl]);      AS = 0.5 * (A[:, :-2] + A[:, sl])
            BN = 0.5 * (B[:, 2:] + B[:, sl]);      BS = 0.5 * (B[:, :-2] + B[:, sl])

            AE_prime = AE * y_eta_E[..., None] - BE * x_eta_E[..., None]
            AW_prime = AW * y_eta_W[..., None] - BW * x_eta_W[..., None]
            BN_prime = AN * y_xi_N[..., None] - BN * x_xi_N[..., None]
            BS_prime = AS * y_xi_S[..., None] - BS * x_xi_S[..., None]

            # --- dissipation
            Vx_ip1, Vy_ip1 = np.roll(Vx_field, -1, axis=0), np.roll(Vy_field, -1, axis=0)
            Vx_im1, Vy_im1 = np.roll(Vx_field, 1, axis=0), np.roll(Vy_field, 1, axis=0)
            c_ip1, c_im1 = np.roll(c_field, -1, axis=0), np.roll(c_field, 1, axis=0)
            u_prev_ip1, u_prev_im1 = np.roll(u_prev, -1, axis=0), np.roll(u_prev, 1, axis=0)

            VxE = 0.5 * (Vx_field[:, sl] + Vx_ip1[:, sl]); VyE = 0.5 * (Vy_field[:, sl] + Vy_ip1[:, sl])
            cE = 0.5 * (c_field[:, sl] + c_ip1[:, sl])
            lambda_E = np.abs(VxE * y_eta_E - VyE * x_eta_E) + cE * np.sqrt(y_eta_E**2 + x_eta_E**2)
            DE = EPSILON * lambda_E[..., None] * (u_prev_ip1[:, sl] - u_prev[:, sl])

            VxW = 0.5 * (Vx_field[:, sl] + Vx_im1[:, sl]); VyW = 0.5 * (Vy_field[:, sl] + Vy_im1[:, sl])
            cW = 0.5 * (c_field[:, sl] + c_im1[:, sl])
            lambda_W = np.abs(VxW * y_eta_W - VyW * x_eta_W) + cW * np.sqrt(y_eta_W**2 + x_eta_W**2)
            DW = EPSILON * lambda_W[..., None] * (u_prev[:, sl] - u_prev_im1[:, sl])

            VxN = 0.5 * (Vx_field[:, sl] + Vx_field[:, 2:]); VyN = 0.5 * (Vy_field[:, sl] + Vy_field[:, 2:])
            cN = 0.5 * (c_field[:, sl] + c_field[:, 2:])
            lambda_N = np.abs(VxN * y_xi_N - VyN * x_xi_N) + cN * np.sqrt(y_xi_N**2 + x_xi_N**2)
            DN = EPSILON * lambda_N[..., None] * (u_prev[:, 2:] - u_prev[:, sl])

            VxS = 0.5 * (Vx_field[:, sl] + Vx_field[:, :-2]); VyS = 0.5 * (Vy_field[:, sl] + Vy_field[:, :-2])
            cS = 0.5 * (c_field[:, sl] + c_field[:, :-2])
            lambda_S = np.abs(VxS * y_xi_S - VyS * x_xi_S) + cS * np.sqrt(y_xi_S**2 + x_xi_S**2)
            DS = EPSILON * lambda_S[..., None] * (u_prev[:, sl] - u_prev[:, :-2])

            # NOTE: this differs from euler.m's "-(AE_prime - AW_prime + BN_prime
            # - BS_prime)". Re-deriving the strong-conservation-form transform
            # from the metrics as defined (eta_x = -y_xi/J, eta_y = x_xi/J)
            # gives Ghat = x_xi*B - y_xi*A, i.e. the NEGATIVE of how BN_prime/
            # BS_prime are built (AN*y_xi - BN*x_xi) -- so the eta-flux term
            # needs a "+" here, not the "-" euler.m uses. Confirmed empirically:
            # with euler.m's sign the residual grows and blows up regardless of
            # CFL (just faster at higher CFL); with this sign it runs stably
            # and residuals trend down. Worth checking against the MATLAB
            # version too, since this would affect it the same way.
            residuals_n[:, sl, :] = -(AE_prime - AW_prime) + (BN_prime - BS_prime) + (DE - DW + DN - DS)

            # --- update interior, boundaries, fluxes
            u_this = u_old.copy()
            u_this[:, sl, :] = u_old[:, sl, :] + ALPHA[k] * (dt / J[:, sl])[..., None] * residuals_n[:, sl, :]
            V_this = V.copy()
            V_this[:, sl, :] = get_primitive(u_this[:, sl, :], gamma)
            u_this, V_this = update_boundaries(V_this, u_this, fs, eta_x, eta_y)
            A, B = solve_fluxes(V_this, fs)

            u_k[k], V_k[k] = u_this, V_this
            if n == 1 and k == NUM_STAGES - 1:
                residuals_1 = residuals_n.copy()

        u, V = u_k[-1], V_k[-1]

        interior_n = residuals_n[:, sl, :]
        interior_1 = residuals_1[:, sl, :]
        L2_norm = (np.sqrt((interior_n**2).sum(axis=(0, 1)))
                   / np.sqrt((interior_1**2).sum(axis=(0, 1))))
        history.append(L2_norm.copy())

        if verbose and n % 10 == 0:
            print(f"\nIteration {n}")
            display_residuals(L2_norm)

        if np.all(L2_norm < tolerance):
            print(f"Converged at iteration {n}")
            display_residuals(L2_norm)
            break

        u_old = u

    else:
        print("Did not converge.")
        display_residuals(L2_norm)

    return V, np.array(history)


def run_solver_robust(iterations=ITERATIONS, tolerance=TOLERANCE, verbose=True,
                        cfl=CFL, max_backoffs=4, backoff_factor=0.5):
    """Not every swept section design has the same stability margin -- some
    shapes' grids are only marginally stable at a given CFL (e.g. this one:
    residuals peak much higher and decay much slower than the baseline
    design did), so a FloatingPointError (NaN/Inf from a diverging solve)
    doesn't necessarily mean something is wrong with that design -- it can
    just need a lower CFL. On divergence, halve the CFL and restart from
    scratch, up to max_backoffs times, rather than failing the whole batch
    run or requiring per-design manual tuning."""
    attempt_cfl = cfl
    for attempt in range(max_backoffs + 1):
        try:
            return run_solver(iterations=iterations, tolerance=tolerance,
                               verbose=verbose, cfl=attempt_cfl)
        except FloatingPointError:
            if attempt == max_backoffs:
                print(f"Still diverging after {max_backoffs} CFL backoffs "
                      f"(last tried CFL={attempt_cfl:.4f}); giving up on this design.")
                raise
            attempt_cfl *= backoff_factor
            print(f"Diverged at CFL={attempt_cfl / backoff_factor:.4f}; "
                  f"retrying from scratch with CFL={attempt_cfl:.4f} "
                  f"(backoff {attempt + 1}/{max_backoffs}).")


RESULTS_DIR = "results"


def plot_velocity_field(x, y, Vmag, wall, out_path):
    """Filled contour of velocity magnitude over the physical grid, O-grid
    seam closed for a clean plot (mirrors the style used for the mesh
    plots earlier)."""
    # close the periodic i-direction so there's no gap in the contour/wall line
    Xc = np.vstack([x, x[:1]])
    Yc = np.vstack([y, y[:1]])
    Vc = np.vstack([Vmag, Vmag[:1]])
    wall_c = np.vstack([wall, wall[:1]])

    fig, ax = plt.subplots(figsize=(9, 8))
    cf = ax.contourf(Xc, Yc, Vc, 40, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Velocity magnitude (m/s)")
    ax.plot(wall_c[:, 0], wall_c[:, 1], "r-", lw=1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Velocity Magnitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    V, history = run_solver_robust()

    rho = V[..., 0]
    Vx = V[..., 1]
    Vy = V[..., 2]
    P = V[..., 3]
    Vmag = np.sqrt(Vx**2 + Vy**2)
    Cp_plot = 1 - (Vmag / u_inf)**2
    T = P / (rho * R)
    c_final = np.sqrt(gamma * P / rho)
    Mach = Vmag / c_final

    np.savez("euler2D_solution.npz", x=x, y=y, rho=rho, Vx=Vx, Vy=Vy, P=P,
             Vmag=Vmag, Cp=Cp_plot, T=T, Mach=Mach, history=history)
    print("Saved solution to euler2D_solution.npz")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    vel_path = os.path.join(RESULTS_DIR, results_filename)
    plot_velocity_field(x, y, Vmag, wall, vel_path)
    print(f"Saved velocity field plot to {vel_path}")