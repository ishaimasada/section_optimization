import subprocess, json, os

# Change the current working directory to the file location
filepath = os.path.abspath(__file__)
directory = os.path.dirname(filepath)
os.chdir(directory)

# Load parameters from JSON file
with open("param_iter.json", "r") as file:
    parameters = json.load(file)

# Determine line slopes of each INNER control point 

# Run the section design script to generate coordinates for the grid generator
design_filename = "section_design.py"
result = subprocess.run(["python", design_filename], capture_output=True, text=True)

# Check if results are valid
match result:
    case True:
        # Run the grid generator
        grid_generator_filename = "grid_generator.py"
        subprocess.run(["python", grid_generator_filename])
    case False:
        pass # continue?

# Run the solver to obtain results
subprocess.run(["python", "euler2D.py"])

# Step inner control points


# REPEAT