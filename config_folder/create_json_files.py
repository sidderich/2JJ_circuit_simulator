import json
import numpy as np

# Create multiple JSON files
# Boundaries for critical currents
I_C1_min = 10e-6
I_C1_max = 100e-6
I_C1_step = 5e-6

I_C2_min = 10e-6
I_C2_max = 100e-6
I_C2_step = 5e-6

# Boundaries of resistances
R_min = 30
R_max = 100 
R_step = 5

R1_min = 30
R1_max = 100
R1_step = 5

R2_min = 30
R2_max = 100
R2_step = 5

L_min = 10e-12
L_max = 50e-12
L_step = 1e-12



# Create list of critical currents
I_C1_list = np.arange(I_C1_min, I_C1_max, I_C1_step)
I_C2_list = np.arange(I_C2_min, I_C2_max, I_C2_step)

# Create list of resistances
R_list = np.arange(R_min, R_max, R_step)
R1_list = np.arange(R1_min, R1_max, R1_step)
R2_list = np.arange(R2_min, R2_max, R2_step)
L_list = np.arange(L_min, L_max, L_step)

# Create dictionary of simulation parameters
sim_params = {
    "model": "ind",
    "time_step": 0.1,
    "max_current": 300e-6
}

# iterate over L_list

for L in L_list:
    # Create a dictionary of junction parameters
    junction_parameters = {
        "I_C1": 50e-6,
        "I_C2": 45e-6,
        "R": 50,
        "R_1": 15,
        "R_2": 10,
        "L": L
    }
    # Create a json file with simulation parameters and junction parameters
    with open(f"sim_params_{L}.json", "w") as f:
        json.dump({"simulation_parameters": sim_params, "junction_parameters": junction_parameters}, f, indent=4)