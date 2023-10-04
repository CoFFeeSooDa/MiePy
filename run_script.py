import numpy as np
import functions.utility as util
import MiePy

# Path of input file
json_path = './input_json/Demo_AngleMode_CF.json'

# Read from json file
inputs = util.read_json(json_path)

# Preprocessing (initialize settings of a calculation)
wavelength = np.float64(300.0e-9)
k = 2*np.pi / wavelength
boundary_radius = inputs['Settings']['BoundaryRadius']
kbr = k * boundary_radius

settings = {'SourceDipole':inputs['Settings']['SourceDipole'],
            'TestDipole':inputs['Settings']['TestDipole'],
            'ExpansionOrder':3,#inputs['Settings']['ExpansionOrder'],
            'BoundaryCondition':inputs['Settings']['BoundaryCondition'],
            'BoundaryRadius':boundary_radius,
            'Dpstrength':inputs['Settings']['Dpstrength'],
            'wavelength':wavelength,
            'k': k,
            'kbr': kbr,
            'nr': np.array([1, 1.3499+0.9903j])}

# Initialize MiePy solver
MP = MiePy.MiePy(settings)

M, N = MP._vector_spherical_function(dipole_type='source', function_type='3')


