import csv
import numpy as np
import functions.utility as util
import MiePy


# Path of input file
json_path = './input_json/Demo_AngleMode_CF.json'

# Read from json file
inputs = util.read_settings_json(json_path)

# Read dielectric data according to the settings in json
settings = inputs['Settings']
#calc = inputs['Settings'].copy()

# Preprocessing (initialize settings of a calculation)
wavelength = settings['wavelength']
k0 = 2*np.pi / wavelength
boundary_radius = np.array(inputs['Settings']['BoundaryRadius'])
k0br = k0[0] * boundary_radius


calc = {'SourceDipole':inputs['Settings']['SourceDipole'],
        'TestDipole':inputs['Settings']['TestDipole'],
        'ExpansionOrder':inputs['Settings']['ExpansionOrder'],
        'BoundaryCondition':inputs['Settings']['BoundaryCondition'],
        'BoundaryRadius':boundary_radius,
        'Dpstrength':inputs['Settings']['Dpstrength'],
        'wavelength':(settings['wavelength'])[0],
        'k0': k0,
        'k0br': k0br,
        'ni': np.array([1, 1.349872708298668+0.990253984899622j])}

# Initialize MiePy solver
MP = MiePy.MiePy(calc)

for ii in range(10):
    calc['wavelength'] = (settings['wavelength'])[ii]
    calc['k0'] = k0[ii]
    calc['ni'] = (settings['ni'])[ii]
    print(calc['ni'])
    calc['k0br'] = k0[ii] * boundary_radius
    MP.refresh(calc)
    E = MP.total_electric_field()
    print(f'{E=}')

