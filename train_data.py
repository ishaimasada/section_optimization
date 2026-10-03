from pyaero.aerodynamics.curves import *
import subprocess, json, numpy, sys, os

# Change the current working directory to the file location
filepath = os.path.abspath(__file__)
directory = os.path.dirname(filepath)
os.chdir(directory)

# TEST PARAMETERES
parameter_filename = "param_iter.json"
profile_design_filename = "section_design.py"
grid_generator_filename = "grid_generator.py"
solver_filename = "euler2D.py"
num_samples = 3

# Load parameters from JSON file
with open(parameter_filename, "r") as file:
    section = json.load(file)["sections"][0]

# Determine line slopes of each INNER control point 
upper_LE_phi = numpy.deg2rad(section["inlet angle"]) + numpy.deg2rad(section["LE angle"])
lower_LE_phi = numpy.deg2rad(section["inlet angle"]) - numpy.deg2rad(section["LE angle"])
upper_TE_phi = numpy.deg2rad(section["exit angle"]) + numpy.deg2rad(section["TE angle"])
lower_TE_phi = numpy.deg2rad(section["exit angle"]) - numpy.deg2rad(section["TE angle"])

upper_control_x = numpy.array(section["upper"]["x"]).flatten()
upper_control_y = numpy.array(section["upper"]["y"]).flatten()
upper = Bezier([Point(x,y,0) for (x,y) in list(zip(upper_control_x, upper_control_y))], resolution=24)
upper_end_control_x = numpy.array(section["upper end"]["x"]).flatten()
upper_end_control_y = numpy.array(section["upper end"]["y"]).flatten()
upper_end = Bezier([Point(x,y,0) for (x,y) in list(zip(upper_end_control_x, upper_end_control_y))], resolution=7)
lower_control_x = numpy.array(section["lower"]["x"]).flatten()
lower_control_y = numpy.array(section["lower"]["y"]).flatten()
lower = Bezier([Point(x,y,0) for (x,y) in list(zip(lower_control_x, lower_control_y))], resolution=upper.resolution+upper_end.resolution)

get_x = numpy.vectorize(lambda position: position.x_coord)
get_y = numpy.vectorize(lambda position: position.y_coord)
upper_positions = copy.deepcopy(upper.positions)
upper_positions.extend(upper_end.positions[1:])
upper_positions = numpy.asarray(upper_positions)
lower_positions = numpy.asarray(copy.deepcopy(lower.positions))
upper_x = get_x(upper_positions)
upper_y = get_y(upper_positions)
lower_x = get_x(lower_positions)
lower_y = get_y(lower_positions)
LE_connection = Panel(Point(upper_x[0], upper_y[0], 0), Point(lower_x[0], lower_y[0], 0))
LE = LE_connection.get_position(0.5)
TE_connection = Panel(Point(upper_x[-1], upper_y[-1], 0), Point(lower_x[-1], lower_y[-1], 0))
TE = TE_connection.get_position(0.5)

# Arbitrary choice of extent of each control point variation
chord = Panel(TE, LE).length
extent = chord / 3

# Parameterization of all control points
LE_upper = upper.control_points[0]
LE_lower = lower.control_points[0]
TE_upper = upper.control_points[-1]
TE_lower = lower.control_points[-1]
t0_end = Point(LE_upper.x_coord + extent * numpy.cos(upper_LE_phi), LE_upper.y_coord + extent * numpy.sin(upper_LE_phi), 0)
t1_end = Point(TE_upper.x_coord - extent * numpy.cos(upper_TE_phi), TE_upper.y_coord + extent * numpy.sin(upper_TE_phi), 0)
t2_end = Point(LE_lower.x_coord + extent * numpy.cos(lower_LE_phi), LE_lower.y_coord + extent * numpy.sin(lower_LE_phi), 0)
t3_end = Point(TE_lower.x_coord - extent * numpy.cos(lower_TE_phi), TE_lower.y_coord + extent * numpy.sin(lower_TE_phi), 0)
domain0 = Panel(LE_upper, t0_end)
domain1 = Panel(TE_upper, t1_end)
domain2 = Panel(LE_lower, t2_end)
domain3 = Panel(TE_lower, t3_end)
t0 = numpy.linspace(0, 1, num_samples)
t1 = numpy.linspace(0, 1, num_samples)
t2 = numpy.linspace(0, 1, num_samples)
t3 = numpy.linspace(0, 1, num_samples)

# Domain Space Visualization
plt.plot(upper_x, upper_y)
plt.plot(lower_x, lower_y)
plt.scatter([LE_upper.x_coord, LE_lower.x_coord, TE_upper.x_coord, TE_lower.x_coord], [LE_upper.y_coord, LE_lower.y_coord, TE_upper.y_coord, TE_lower.y_coord])
plt.scatter([t0_end.x_coord, t2_end.x_coord, t1_end.x_coord, t3_end.x_coord], [t0_end.y_coord, t2_end.y_coord, t1_end.y_coord, t3_end.y_coord])
plt.plot([LE_upper.x_coord, t0_end.x_coord], [LE_upper.y_coord, t0_end.y_coord])
plt.plot([LE_lower.x_coord, t2_end.x_coord], [LE_lower.y_coord, t2_end.y_coord])
plt.plot([TE_upper.x_coord, t1_end.x_coord], [TE_upper.y_coord, t1_end.y_coord])
plt.plot([TE_lower.x_coord, t3_end.x_coord], [TE_lower.y_coord, t3_end.y_coord])
# plt.show()

# Overwrite the parameters with the test control points
for i in range(num_samples):
    section["upper"]["x"][1] = domain0.get_position(t0[i]).x_coord
    section["upper"]["y"][1] = domain0.get_position(t0[i]).y_coord
    section["upper"]["x"][2] = domain1.get_position(t1[i]).x_coord
    section["upper"]["y"][2] = domain1.get_position(t1[i]).y_coord
    section["lower"]["x"][1] = domain2.get_position(t2[i]).x_coord
    section["lower"]["y"][1] = domain2.get_position(t2[i]).y_coord
    section["lower"]["x"][2] = domain3.get_position(t3[i]).x_coord
    section["lower"]["y"][2] = domain3.get_position(t3[i]).y_coord

    # Save the modified parameters to a new JSON file for the current iteration
    iter_filename = f"param_iter{i}.json"
    with open(iter_filename, "w") as iter_file:
        json.dump({"sections": [section]}, iter_file, indent=4)

    # Run the section design script to generate airfoil coordinates for the grid generator
    subprocess.run(["python", profile_design_filename, f"param_iter{i}.json"])

    # Run the grid generator
    subprocess.run(["python", grid_generator_filename, f"airfoil_ogrid{i}"])

    # Run the solver to obtain results
    subprocess.run(["python", solver_filename, f"velocity_field{i}.png"])
