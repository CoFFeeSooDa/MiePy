import csv
import numpy as np
import functions.utility as util
import MiePy
import scipy
import time
import matplotlib.pyplot as plt

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
    sample_pt = 401
    # Preallocation
    electric_field = np.zeros([sample_pt, 3], dtype=np.complex128)
    wavelength = np.zeros(sample_pt, dtype=np.float64)
    k = np.zeros(sample_pt, dtype=np.complex128)
    electric_field_compare = np.zeros([sample_pt, 3], dtype=np.complex128)
    electric_field_source = np.zeros([sample_pt, 3], dtype=np.complex128)
    GF_scat = np.zeros([sample_pt, 3, 3], dtype=np.complex128)
    
    for ii in range(sample_pt):    
        calc['wavelength'] = (settings['wavelength'])[ii]
        calc['k0'] = (settings['k0'])[ii]
        calc['ni'] = (settings['ni'])[ii]
        calc['k0br'] = (settings['k0br'])[ii]
        MP.refresh(calc)
        wavelength[ii] = calc['wavelength']
        electric_field[ii:ii+1,:] = np.ravel(MP.total_electric_field())
        electric_field_source[ii:ii+1,:] = np.ravel(MP.source_dipole_electric_field())
        GF_scat[ii, :, :] = MP.dyadic_greens_function_scattering()
        # Wavenumber in the dielectric medium
        k[ii] = MP.ni[MP.source_dipole.region] * MP.k0
    
    # Obtain scattering electric field
    # Prefactor (for electric field, cgs but cm -> m)
    prefactor = 4 * np.pi * k**2
    prefactor = prefactor[:, np.newaxis]
    electric_field_scat = prefactor * np.einsum('ijk, k-> ij', GF_scat, MP.source_dipole.ori_sph)
    electric_field -= electric_field_source
    electric_field_compare = electric_field_scat
# mat_dict = {"CF":CF}
# scipy.io.savemat('test.mat',mat_dict)

# Stop timing 
end_time = time.time()

# Output elapsed time
MP.output_elapsed_time(end_time - start_time)

if __name__ == "__main__":
    wavelength *= 1e9
    plt.plot(wavelength, np.real(electric_field[:, 0]), label='Re_E_x')
    # plt.plot(wavelength, np.real(electric_field[:, 1]), label='Re_E_y')
    # plt.plot(wavelength, np.real(electric_field[:, 2]), label='Re_E_z')
    
    plt.plot(wavelength, np.real(electric_field_compare[:, 0]), label='Re_E_compare_x', linestyle='--')
    # plt.plot(wavelength, np.real(electric_field_compare[:, 1]), label='Re_E_compare_y', linestyle='--')
    # plt.plot(wavelength, np.real(electric_field_compare[:, 2]), label='Re_E_compare_z', linestyle='--')
    
    # plt.plot(wavelength, (np.real(electric_field[:, 0]) - np.real(electric_field_compare[:, 0])) / np.real(electric_field[:, 0]),\
    #          label='Re_E_x_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.real(electric_field[:, 1]) - np.real(electric_field_compare[:, 1])) / np.real(electric_field[:, 1]),\
    #          label='Re_E_y_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.real(electric_field[:, 2]) - np.real(electric_field_compare[:, 2])) / np.real(electric_field[:, 2]),\
    #          label='Re_E_z_relative_diff', linestyle='-')
    plt.legend()
    plt.show()
    
    # plt.plot(wavelength, np.imag(electric_field[:, 0]), label='Im_E_x')
    # plt.plot(wavelength, np.imag(electric_field[:, 1]), label='Im_E_y')
    # plt.plot(wavelength, np.imag(electric_field[:, 2]), label='Im_E_z')
    # plt.plot(wavelength, np.imag(electric_field_compare[:, 0]), label='Im_E_compare_x', linestyle='--')
    # plt.plot(wavelength, np.imag(electric_field_compare[:, 1]), label='Im_E_compare_y', linestyle='--')
    # plt.plot(wavelength, np.imag(electric_field_compare[:, 2]), label='Im_E_compare_z', linestyle='--')
    # plt.legend()
    # plt.show()
    # pass
