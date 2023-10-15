import csv
import numpy as np
import functions.utility as util
import MiePy
import scipy
import time


# Path of input file
json_path = './input_json/Demo_wavelength_CF.json'

# Read from json file
inputs = util.read_settings_json(json_path)

# Read dielectric data according to the settings in json
settings = inputs['Settings']

# We set calc to a single calculation data
calc = inputs['Settings'].copy()

# Start timing 
start_time = time.time()

# Initialize MiePy solver
MP = MiePy.MiePy(calc)

# Main loop
if settings['ModeName'] == 'wavelength':
    # Preallocation
    CF = np.zeros([401,1], dtype=np.float64)
    for ii in range(401):    
        calc['wavelength'] = (settings['wavelength'])[ii]
        calc['k0'] = (settings['k0'])[ii]
        calc['ni'] = (settings['ni'])[ii]
        calc['k0br'] = (settings['k0br'])[ii]
        MP.refresh(calc)
        CF[ii:ii+1,:] = MP.coupling_factor()

elif settings['ModeName'] == 'angle':
    # Preallocation
    CF = np.zeros([176,1], dtype=np.float64)
    theta = np.linspace(settings['Theta_i'],settings['Theta_f'],settings['ThetaPoints'])
    for ii in range(176):
        (calc['TestDipole']['Pos_Sph'])[1] = theta[ii]
        MP.refresh(calc)
        CF[ii:ii+1,:] = MP.coupling_factor()

mat_dict = {"CF":CF}
scipy.io.savemat('test.mat',mat_dict)

# Stop timing 
end_time = time.time()

# Output elapsed time
MP.output_elapsed_time(end_time - start_time)
