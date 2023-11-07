import csv
import copy
import numpy as np
import functions.utility as util
import MiePy
import scipy
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat

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
    wavelength = np.zeros(sample_pt, dtype=np.float64)
    k = np.zeros(sample_pt, dtype=np.complex128)
    GF_scat = np.zeros([sample_pt, 3, 3], dtype=np.complex128)
    for ii in range(sample_pt):    
        calc['wavelength'] = (settings['wavelength'])[ii]
        calc['k0'] = (settings['k0'])[ii]
        calc['ni'] = (settings['ni'])[ii]
        calc['k0br'] = (settings['k0br'])[ii]
        MP.refresh(calc)
        wavelength[ii] = calc['wavelength']
        # Green's Function Compare
        GF_scat[ii, :, :] = MP.dyadic_greens_function_scattering()
        # Wavenumber in the dielectric medium
        k[ii] = MP.ni[MP.source_dipole.region] * MP.k0
    # Obtain scattering electric field
    # Prefactor (for electric field, cgs but cm -> m)
    prefactor = 4 * np.pi * k**2
    prefactor = prefactor[:, np.newaxis]
    electric_field_scat = prefactor * np.einsum('ijk, k-> ij', GF_scat, MP.source_dipole.ori_sph)
    electric_field = copy.deepcopy(electric_field_scat)
    
# Path of input file
json_path = './input_json/Demo_wavelength_GScat_OMNN.json'

# Read from json file
inputs = util.read_settings_json(json_path)

# Read dielectric data according to the settings in json
settings = inputs['Settings']

# We set calc to a single calculation data
calc = inputs['Settings'].copy()

# Initialize MiePy solver
MP = MiePy.MiePy(calc)

# Main loop
if settings['ModeName'] == 'wavelength':
    sample_pt = 10001
    # Preallocation
    wavelength_OMNN = np.zeros(sample_pt, dtype=np.float64)
    k = np.zeros(sample_pt, dtype=np.complex128)
    GF_scat = np.zeros([sample_pt, 3, 3], dtype=np.complex128)
    for ii in range(sample_pt):    
        calc['wavelength'] = (settings['wavelength'])[ii]
        calc['k0'] = (settings['k0'])[ii]
        calc['ni'] = (settings['ni'])[ii]
        calc['k0br'] = (settings['k0br'])[ii]
        MP.refresh(calc)
        wavelength_OMNN[ii] = calc['wavelength']
        # Green's Function Compare
        GF_scat[ii, :, :] = MP.dyadic_greens_function_scattering()
        # Wavenumber in the dielectric medium
        k[ii] = MP.ni[MP.source_dipole.region] * MP.k0
    # Obtain scattering electric field
    # Prefactor (for electric field, cgs but cm -> m)
    prefactor = 4 * np.pi * k**2
    prefactor = prefactor[:, np.newaxis]
    electric_field_scat = prefactor * np.einsum('ijk, k-> ij', GF_scat, MP.source_dipole.ori_sph)
    electric_field_compare = copy.deepcopy(electric_field_scat)

# Stop timing 
end_time = time.time()

# Output elapsed time
MP.output_elapsed_time(end_time - start_time)

if __name__ == "__main__":
    # Wavelength Mode Test
    wavelength *= 1e9
    wavelength_OMNN *= 1e9
    # plt.plot(wavelength, np.real(electric_field[:, 0]), label='Re_E_x')
    # plt.plot(wavelength, np.real(electric_field[:, 1]), label='Re_E_y')
    # plt.plot(wavelength, np.real(electric_field[:, 2]), label='Re_E_z')
    
    # plt.plot(wavelength, np.real(electric_field_compare[:, 0]), label='Re_E_OMNN_x', linestyle='--')
    # plt.plot(wavelength, np.real(electric_field_compare[:, 1]), label='Re_E_OMNN_y', linestyle='--')
    # plt.plot(wavelength, np.real(electric_field_compare[:, 2]), label='Re_E_OMNN_z', linestyle='--')
    
    # plt.plot(wavelength, (np.real(electric_field[:, 0]) - np.real(electric_field_compare[:, 0])),\
    #          label='Re_E_x_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.real(electric_field[:, 1]) - np.real(electric_field_compare[:, 1])),\
    #          label='Re_E_y_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.real(electric_field[:, 2]) - np.real(electric_field_compare[:, 2])),\
    #          label='Re_E_z_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.imag(electric_field[:, 0]) - np.imag(electric_field_compare[:, 0])),\
    #          label='Im_E_x_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.imag(electric_field[:, 1]) - np.imag(electric_field_compare[:, 1])),\
    #          label='Im_E_y_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.imag(electric_field[:, 2]) - np.imag(electric_field_compare[:, 2])),\
    #          label='Im_E_z_relative_diff', linestyle='-')
    
    # plt.plot(wavelength, (np.real(electric_field[:, 0]) - np.real(electric_field_compare[:, 0])) / np.real(electric_field[:, 0]),\
    #          label='Re_E_x_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.real(electric_field[:, 1]) - np.real(electric_field_compare[:, 1])) / np.real(electric_field[:, 1]),\
    #          label='Re_E_y_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.real(electric_field[:, 2]) - np.real(electric_field_compare[:, 2])) / np.real(electric_field[:, 2]),\
    #          label='Re_E_z_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.imag(electric_field[:, 0]) - np.imag(electric_field_compare[:, 0])) / np.imag(electric_field[:, 0]),\
    #          label='Im_E_x_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.imag(electric_field[:, 1]) - np.imag(electric_field_compare[:, 1])) / np.imag(electric_field[:, 1]),\
    #          label='Im_E_y_relative_diff', linestyle='-')
    # plt.plot(wavelength, (np.imag(electric_field[:, 2]) - np.imag(electric_field_compare[:, 2])) / np.imag(electric_field[:, 2]),\
    #          label='Im_E_z_relative_diff', linestyle='-')
    
    
    plt.plot(wavelength, np.imag(electric_field[:, 0]), label='Im_E_x')
    plt.plot(wavelength, np.imag(electric_field[:, 1]), label='Im_E_y')
    plt.plot(wavelength, np.imag(electric_field[:, 2]), label='Im_E_z')
    plt.plot(wavelength_OMNN, np.imag(electric_field_compare[:, 0]), label='Im_E_compare_x', linestyle='--')
    plt.plot(wavelength_OMNN, np.imag(electric_field_compare[:, 1]), label='Im_E_compare_y', linestyle='--')
    plt.plot(wavelength_OMNN, np.imag(electric_field_compare[:, 2]), label='Im_E_compare_z', linestyle='--')
    # plt.legend()
    # plt.show()
    plt.rcParams['savefig.dpi'] = 1000 #圖片像素
    plt.rcParams['figure.dpi'] = 1000 #分辨率
    plt.title("Scattering Green's Function Difference $\\overline{\\overline{\\mathbf{G}}}_{\\mathrm{scat}}$")
    plt.xlabel("Wavelength ($\\mathrm{nm}$)")
    plt.ylabel("Rel. Difference in $\\overline{\\overline{\\mathbf{G}}}_{\\mathrm{scat}}$")
    plt.legend()
    plt.show()
    
    # Purcell Factor test
    # matlab_purcell = loadmat("C:/Users/User/Desktop/Code/MieDipole/result_purcell.mat")['result_purcell']
    # wavelength *= 1e9
    # plt.plot(matlab_purcell[0] * 1e9, matlab_purcell[1], label = 'Matlab')
    # plt.plot(wavelength, purcell_factor, '--', label = 'Python')
    # plt.legend()
    # plt.show()
    # Difference
    # matlab_purcell = loadmat("C:/Users/User/Desktop/Code/MieDipole/result_purcell.mat")['result_purcell']
    # wavelength *= 1e9
    # plt.plot(wavelength, purcell_factor - matlab_purcell[1], label = 'python vs matlab')
    # plt.rcParams['savefig.dpi'] = 1000 #圖片像素
    # plt.rcParams['figure.dpi'] = 1000 #分辨率
    # plt.title("Purcell Difference $F_{\\mathrm{p}}$")
    # plt.xlabel("Wavelength ($\\mathrm{nm}$)")
    # plt.ylabel("Rel. Difference in $F_{\\mathrm{p}}$")
    # plt.legend()
    # plt.show()
    pass
