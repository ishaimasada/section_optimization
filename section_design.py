from pyaero.aerodynamics.curves import *
import matplotlib.pyplot as plt
import numpy, json, os


def wrap(a):
    return (a + numpy.pi) % (2*numpy.pi) - numpy.pi

def match_tangency(curve_m, ellipse_m):
    curve_m = curve_m[0:round(len(curve_m)/6)]
    num_curve_points = len(curve_m)
    num_ellipse_points = len(ellipse_m)
    raw_dm = numpy.zeros((num_curve_points, 3))
    for t in range(0, num_curve_points):
        i = t
        j = int(numpy.round(t / (num_curve_points-1) * (num_ellipse_points-1)))
        dm = curve_m[i] - ellipse_m[j]
        raw_dm[t, :] = numpy.array([i, j, dm])
    min_idx = numpy.argmin(abs(raw_dm[:, 2]))
    [curve_idx, ellipse_idx, _] = raw_dm[min_idx, :]
    return int(curve_idx), int(ellipse_idx)


def make_edge(AR, phi, camberline, upper:Bezier, lower:Bezier, upper_end:Bezier=None, side:str="LE"):
    upper_x = [point.x_coord for point in upper.positions]
    upper_y = [point.y_coord for point in upper.positions]
    if upper_end is not None:
        upper_x += [point.x_coord for point in upper_end.positions]
        upper_y += [point.y_coord for point in upper_end.positions]
    lower_x = [point.x_coord for point in lower.positions]
    lower_y = [point.y_coord for point in lower.positions]
    upper_m = [(upper_y[idx+1]-upper_y[idx-1])/(upper_x[idx+1]-upper_x[idx-1]) for idx in range(1, len(upper_x)-1)]
    lower_m = [(lower_y[idx+1]-lower_y[idx-1])/(lower_x[idx+1]-lower_x[idx-1]) for idx in range(1, len(lower_x)-1)]
    if side == "TE":
        phi = numpy.pi - phi

    # Vertex Position (mid-point of endpoints)
    x0 = upper_x[0]
    y0 = upper_y[0]
    x1 = lower_x[0]
    y1 = lower_y[0]
    end_connection = Panel(Point(x0,y0,0), Point(x1,y1,0))
    vertex_x = end_connection.get_position(0.5).x_coord
    vertex_y = end_connection.get_position(0.5).y_coord

    # REFERENCE ELLIPSE (for tangency condition)
    bref = end_connection.length / 2
    aref = AR * bref
    psi_ref = numpy.atan(bref/aref)
    match side:
        case "LE":
            upper_theta_ref = numpy.linspace(phi+psi_ref, phi+numpy.pi/2, 15)
            lower_theta_ref = numpy.linspace(phi+numpy.pi/2, phi+numpy.pi-psi_ref, 15)
        case "TE":
            upper_theta_ref = numpy.linspace(phi+numpy.pi/2, phi+numpy.pi-psi_ref, 15)
            lower_theta_ref = numpy.linspace(phi+psi_ref, phi+numpy.pi/2, 15)
            upper_x = numpy.flip(upper_x)
            upper_y = numpy.flip(upper_y)
            lower_x = numpy.flip(lower_x)
            lower_y = numpy.flip(lower_y)
            lower_m.reverse()
            upper_m.reverse()
    upper_r_ref = 2*aref*numpy.cos(upper_theta_ref-phi) / (numpy.cos(upper_theta_ref-phi)**2 + AR**2 * numpy.sin(upper_theta_ref-phi)**2)
    upper_x_ref = upper_r_ref * numpy.cos(upper_theta_ref) + vertex_x
    upper_y_ref = upper_r_ref * numpy.sin(upper_theta_ref) + vertex_y
    lower_r_ref = 2*aref*numpy.cos(lower_theta_ref-phi) / (numpy.cos(lower_theta_ref-phi)**2 + AR**2 * numpy.sin(lower_theta_ref-phi)**2)
    lower_x_ref = lower_r_ref * numpy.cos(lower_theta_ref) + vertex_x
    lower_y_ref = lower_r_ref * numpy.sin(lower_theta_ref) + vertex_y
    match side:
        case "LE":
            upper_x_ref = numpy.flip(upper_x_ref)
            upper_y_ref = numpy.flip(upper_y_ref)
        case "TE":
            lower_x_ref = numpy.flip(lower_x_ref)
            lower_y_ref = numpy.flip(lower_y_ref)
    upper_m_ref = [(upper_y_ref[idx+1]-upper_y_ref[idx-1])/(upper_x_ref[idx+1]-upper_x_ref[idx-1]) for idx in range(1, len(upper_r_ref)-1)]
    lower_m_ref = [(lower_y_ref[idx+1]-lower_y_ref[idx-1])/(lower_x_ref[idx+1]-lower_x_ref[idx-1]) for idx in range(1, len(lower_r_ref)-1)]
    upper_curve_idx, upper_ellipse_idx = match_tangency(upper_m, upper_m_ref)
    lower_curve_idx, lower_ellipse_idx = match_tangency(lower_m, lower_m_ref)
    xref_upper = upper_x_ref[upper_ellipse_idx]
    yref_upper = upper_y_ref[upper_ellipse_idx]
    xref_lower = lower_x_ref[lower_ellipse_idx]
    yref_lower = lower_y_ref[lower_ellipse_idx]

    # Scale Ellipse to meet positional requirement
    upper_panel_ref = Panel(Point(vertex_x, vertex_y,0), Point(xref_upper, yref_upper,0))
    lower_panel_ref = Panel(Point(vertex_x, vertex_y,0), Point(xref_lower, yref_lower,0))
    match side:
        case "LE":
            psi_upper = wrap(upper_panel_ref.phi - phi)
            psi_lower = wrap(phi - lower_panel_ref.phi)
        case "TE":
            psi_upper = wrap(phi - upper_panel_ref.phi)
            psi_lower = wrap(lower_panel_ref.phi - phi)
            camberline.reverse()
    ref_upper_height = upper_panel_ref.length * abs(numpy.sin(psi_upper))
    ref_lower_height = lower_panel_ref.length * abs(numpy.sin(psi_lower))
    upper_connection = camberline[upper_curve_idx]
    lower_connection = camberline[lower_curve_idx]
    upper_half_thickness = (upper_connection.length/2) * abs(numpy.sin(phi - upper_connection.phi))
    lower_half_thickness = (lower_connection.length/2) * abs(numpy.sin(phi - lower_connection.phi))

    # (Leading/Trailing) Edge Ellipse
    b_upper = bref * upper_half_thickness / ref_upper_height
    b_lower = bref * lower_half_thickness / ref_lower_height
    match side:
        case "LE":
            theta_upper = numpy.linspace(phi+psi_upper, phi+numpy.pi/2, 7)
            theta_lower = numpy.linspace(phi+numpy.pi/2, phi+numpy.pi-psi_lower, 7)
        case "TE":
            theta_upper = numpy.linspace(phi+numpy.pi/2, phi+numpy.pi-psi_upper, 7)
            theta_lower = numpy.linspace(phi+psi_lower, phi+numpy.pi/2, 7)
    a_upper = AR * b_upper
    a_lower = AR * b_lower
    r_upper = 2*a_upper*numpy.cos(theta_upper-phi) / (numpy.cos(theta_upper-phi)**2 + AR**2 * numpy.sin(theta_upper-phi)**2)
    x_upper = r_upper * numpy.cos(theta_upper) + vertex_x
    y_upper = r_upper * numpy.sin(theta_upper) + vertex_y
    r_lower = 2*a_lower*numpy.cos(theta_lower-phi) / (numpy.cos(theta_lower-phi)**2 + AR**2 * numpy.sin(theta_lower-phi)**2)
    x_lower = r_lower * numpy.cos(theta_lower) + vertex_x
    y_lower = r_lower * numpy.sin(theta_lower) + vertex_y

    x = numpy.concatenate((x_upper, x_lower[1:]))
    y = numpy.concatenate((y_upper, y_lower[1:]))
    return x, y, upper_curve_idx, lower_curve_idx

# Change the current working directory to the file location
filepath = os.path.abspath(__file__)
directory = os.path.dirname(filepath)
os.chdir(directory)

# Load parameters from JSON file
with open("parameters.json", "r") as file:
    parameters = json.load(file)

LE_radius = parameters["annulus"]["LE hub"]
section = parameters["sections"][0]

# Curve Objects
upper_control_x = numpy.array(section["upper"]["x"]).flatten()
upper_control_y = numpy.array(section["upper"]["y"]).flatten()
upper = Bezier([Point(x,y,0) for (x,y) in list(zip(upper_control_x, upper_control_y))], resolution=24)
upper_end_control_x = numpy.array(section["upper end"]["x"]).flatten()
upper_end_control_y = numpy.array(section["upper end"]["y"]).flatten()
upper_end = Bezier([Point(x,y,0) for (x,y) in list(zip(upper_end_control_x, upper_end_control_y))], resolution=7)
lower_control_x = numpy.array(section["lower"]["x"]).flatten()
lower_control_y = numpy.array(section["lower"]["y"]).flatten()
lower = Bezier([Point(x,y,0) for (x,y) in list(zip(lower_control_x, lower_control_y))], resolution=upper.resolution+upper_end.resolution)

# Curve Positions
upper_positions = copy.deepcopy(upper.positions)
upper_positions.extend(upper_end.positions[1:])
upper_positions = numpy.asarray(upper_positions)
lower_positions = numpy.asarray(copy.deepcopy(lower.positions))

# LE
camberline = [Panel(upper.positions[idx], lower.positions[idx]) for idx in range(upper.resolution)]
camberline.extend([Panel(upper_end.positions[idx], lower.positions[idx+upper.resolution]) for idx in range(1, upper_end.resolution)])
LE_hub_radius = parameters["annulus"]["LE hub"]
LE_AR = section["LE AR"]
alpha = numpy.deg2rad(section["inlet angle"])
half_wedge = numpy.deg2rad(section["LE angle"])
if LE_AR < 1: e = numpy.sqrt(1 - LE_AR**2)
else: e = numpy.sqrt(1 - (1/LE_AR)**2)
LE_x, LE_y, upper_cutoff_idx, lower_cutoff_idx = make_edge(LE_AR, alpha, camberline, upper, lower, side="LE")

# Results
def first_past(positions, x_target):
    for i, p in enumerate(positions):
        if p.x_coord > x_target:
            return i
    return len(positions) - 1
upper_cutoff_idx = first_past(upper_positions, LE_x[0])
lower_cutoff_idx = first_past(lower_positions, LE_x[-1])
upper_positions = upper_positions[upper_cutoff_idx:]
lower_positions = lower_positions[lower_cutoff_idx:]
x = ([p.x_coord for p in upper_positions][::-1] + list(LE_x) + [p.x_coord for p in lower_positions])
y = ([p.y_coord for p in upper_positions][::-1] + list(LE_y) + [p.y_coord for p in lower_positions])
coordinates = numpy.array([x, y]).T
numpy.savetxt("coordinates.txt", coordinates, delimiter=",", header="x,y")
plt.plot(x, y)
plt.axis("equal")
# plt.show()