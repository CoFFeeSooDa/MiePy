import csv
import numpy as np
import functions.utility as util
import MiePy
import scipy
import time


# Path of input file
json_path = './input_json/Demo_wavelength_GScat.json'

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
    if inputs['Output']['Quantity'] == 'CF':
        # Preallocation
        output = np.zeros([401,1], dtype=np.float64)
        for ii in range(401):    
            calc['wavelength'] = (settings['wavelength'])[ii]
            calc['k0'] = (settings['k0'])[ii]
            calc['ni'] = (settings['ni'])[ii]
            calc['k0br'] = (settings['k0br'])[ii]
            MP.refresh(calc)
            output[ii:ii+1,:] = MP.coupling_factor()
    elif inputs['Output']['Quantity'] == 'GScat':
        # Preallocation
        output = np.zeros([401, 3, 3], dtype=np.complex128)
        for ii in range(401):    
            calc['wavelength'] = (settings['wavelength'])[ii]
            calc['k0'] = (settings['k0'])[ii]
            calc['ni'] = (settings['ni'])[ii]
            calc['k0br'] = (settings['k0br'])[ii]
            MP.refresh(calc)
            output[ii, :, :] = MP.dyadic_greens_function_scattering()
            
elif settings['ModeName'] == 'angle':
    # Preallocation
    CF = np.zeros([176,1], dtype=np.float64)
    theta = np.linspace(settings['Theta_i'],settings['Theta_f'],settings['ThetaPoints'])
    for ii in range(176):
        (calc['TestDipole']['Pos_Sph'])[1] = theta[ii]
        MP.refresh(calc)
        CF[ii:ii+1,:] = MP.coupling_factor()

mat_dict = {f"{inputs['Output']['Quantity']}": output}
scipy.io.savemat(f"{inputs['Output']['Quantity']}.mat", mat_dict)

# Stop timing 
end_time = time.time()

# Output elapsed time
MP.output_elapsed_time(end_time - start_time)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    with open(json_path) as f:
        # Load inputs to a dictionary
        json_in = json.load(f)
    f.close()
    x = np.linspace(json_in['Settings']['lambda_i'], json_in['Settings']['lambda_f'], 401)
    y = scipy.io.loadmat(f"{json_in['Output']['Quantity']}.mat")[f"{json_in['Output']['Quantity']}"]
    if y.ndim != 2:
        k0 = 2*np.pi / x
        k = MP.ni[MP.source_dipole.region] * k0
        prefactor = 4 * np.pi * k**2
        prefactor = prefactor[:, np.newaxis]
        y = prefactor * np.einsum('ijk, k-> ij', y, MP.source_dipole.ori_sph)
        legend = ['Re_E_x', 'Re_E_y', 'Re_E_z']
    else:
        legend = []
    plt.plot(x * 1e9, np.real(y[:, 0]), linestyle='-')
    plt.plot(x * 1e9, np.real(y[:, 1]), linestyle='-')
    plt.plot(x * 1e9, np.real(y[:, 2]), linestyle='-')
    plt.legend(legend)
    plt.show()
