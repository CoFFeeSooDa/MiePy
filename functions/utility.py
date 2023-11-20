import json
import numpy as np
import csv
import sys
import functions.text_color as tc
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, interp1d

def find_attribute(target_object:object, parent:str):
            # Get attributes (list) for the target object
            all_attributes = dir(target_object)
            # Get the attributes of an base object
            base_object_attributes = dir(object)
            # Missing attributes in a base object
            base_object_attributes.extend(['__dict__', '__module__', '__weakref__'])
            # Filter the attributes
            user_defined_attributes = [attr for attr in all_attributes if attr not in base_object_attributes]

            for attr in user_defined_attributes:
                #print(type(getattr(target_object,attr)))
                if isinstance(getattr(target_object,attr), list) \
                        or isinstance(getattr(target_object,attr), str) \
                        or isinstance(getattr(target_object,attr), float) \
                        or isinstance(getattr(target_object,attr), int) \
                        or isinstance(getattr(target_object,attr), np.ndarray):
                    print(parent + '.' + str(attr))
                    continue
                elif isinstance(getattr(target_object,attr), object):
                    parent1 = parent + '.' + str(attr)
                    find_attribute(getattr(target_object,attr), parent1)

def read_JnC_csv(csv_file_path):
    data = np.genfromtxt(csv_file_path, delimiter=',')
    data = data[~np.isnan(data)].reshape((-1, 2))
    N = int(np.shape(data)[0])
    data = np.hstack((data[:(N//2), :], data[(N//2):, 1, np.newaxis]))
    # Get wavelength
    wavelength = data[:, 0] * 1e-6

    refractive_index = data[:, 1].astype(np.complex128) + 1j * data[:, 2].astype(np.complex128)    
    dielectrics = np.square(refractive_index)

    return wavelength, dielectrics

def generate_dielectric_csv(lambda_i: float, 
                            lambda_f: float,
                            dielectric_type: str, 
                            N=400):
    if dielectric_type == "JPCL":
        if N <= 400:
            csv_file_path = f"./dielectric_data/Ag_{dielectric_type}.csv"
            return csv_file_path
        else:
            print("More data points required then expected. The maximum is 400.")
            sys.exit('Error in generate_dielectric_csv.')
    else:
        wavelength_interpolate = np.linspace(lambda_i,\
            lambda_f, N + 1)
        wavelength, epsi_JnC = read_JnC_csv("./dielectric_data/Ag_J&C.csv")
        # elif dielectric_type == "OMNN":
        if dielectric_type == "Akima":        
            ## Akima
            interpolate_real = Akima1DInterpolator(wavelength, np.real(epsi_JnC))
            interpolate_imag = Akima1DInterpolator(wavelength, np.imag(epsi_JnC))
        elif dielectric_type == "Pchip":
            ## Pchip
            interpolate_real = PchipInterpolator(wavelength, np.real(epsi_JnC))
            interpolate_imag = PchipInterpolator(wavelength, np.imag(epsi_JnC))
        elif dielectric_type == "Linear":  
            ## Linear
            interpolate_real = interp1d(wavelength, np.real(epsi_JnC), kind='linear')
            interpolate_imag = interp1d(wavelength, np.imag(epsi_JnC), kind='linear')
        else:
            print("Wrong Dielectric Type. Valid keywords are \'JPCL\', \'Akima\', \'Pchip\', and \'Linear\'.")
            sys.exit('Error in generate_dielectric_csv.')
        result = np.zeros((N+1, 3), dtype=np.float64)
        result[:, 0] = wavelength_interpolate
        result[:, 1] = interpolate_real(wavelength_interpolate)
        result[:, 2] = interpolate_imag(wavelength_interpolate)
        np.savetxt(f"./dielectric_data/Ag_{dielectric_type}.csv", result, delimiter=",")
        csv_file_path = f"./dielectric_data/Ag_{dielectric_type}.csv"
    return csv_file_path

def read_dielectrics_csv(csv_file_path, csv_data='refractive_index'):
    # Change csv to numpy array
    data = np.genfromtxt(csv_file_path, delimiter=',', dtype=np.float64)
    # Get wavelength
    wavelength = data[:,0]
        
    if csv_data == 'refractive_index':
        # Get refractive indices
        refractive_index = data[:,1].astype(np.complex128) + 1j * data[:,2].astype(np.complex128)    
    elif csv_data == 'dielectric_constant':
        # Get dielectric constants
        dielectrics = data[:,1].astype(np.complex128) + 1j * data[:,2].astype(np.complex128)
        # Convert to complex refractive indices
        refractive_index = np.sqrt(dielectrics)

    return wavelength, refractive_index

def read_settings_json(json_file_path):
    # Open json setting file
    with open(json_file_path) as f:
        # Load inputs to a dictionary
        inputs = json.load(f)
        
    # Read dielectrics according to the loaded json
    if isinstance(inputs['Settings']['epsi0'], str):
        if 'grid_pts' in inputs['Settings']:
            inputs['Settings']['epsi0'] =\
                generate_dielectric_csv(inputs['Settings']['lambda_i'],
                                        inputs['Settings']['lambda_f'],
                                        inputs['Settings']['epsi0'], 
                                        N=inputs['Settings']['grid_pts'])
        else:
            inputs['Settings']['epsi0'] =\
                generate_dielectric_csv(inputs['Settings']['lambda_i'],
                                        inputs['Settings']['lambda_f'], 
                                        inputs['Settings']['epsi0'])
        lambda0, n0 = read_dielectrics_csv(inputs['Settings']['epsi0'], csv_data='dielectric_constant')
    else:
        lambda0, n0 = None, np.array(inputs['Settings']['epsi0'], dtype=np.complex128)

    if isinstance(inputs['Settings']['epsi1'], str):
        if 'grid_pts' in inputs['Settings']:
            inputs['Settings']['epsi1'] =\
                generate_dielectric_csv(inputs['Settings']['lambda_i'],
                                        inputs['Settings']['lambda_f'],
                                        inputs['Settings']['epsi1'], 
                                        N=inputs['Settings']['grid_pts'])
        else:
            inputs['Settings']['epsi1'] =\
                generate_dielectric_csv(inputs['Settings']['lambda_i'],
                                        inputs['Settings']['lambda_f'], 
                                        inputs['Settings']['epsi1'])
        lambda1, n1 = read_dielectrics_csv(inputs['Settings']['epsi1'], csv_data='dielectric_constant')
    else:
        lambda1, n1 = None, np.array(inputs['Settings']['epsi1'], dtype=np.complex128)
        
    if inputs['Settings']['BoundaryCondition'] == 'coreshell':
        if isinstance(inputs['Settings']['epsi2'], str):
            if 'grid_pts' in inputs['Settings']:
                inputs['Settings']['epsi2'] =\
                    generate_dielectric_csv(inputs['Settings']['lambda_i'],
                                            inputs['Settings']['lambda_f'],
                                            inputs['Settings']['epsi2'], 
                                            N=inputs['Settings']['grid_pts'])
            else:
                inputs['Settings']['epsi2'] =\
                    generate_dielectric_csv(inputs['Settings']['lambda_i'],
                                            inputs['Settings']['lambda_f'],
                                            inputs['Settings']['epsi2'])
            lambda2, n2 = read_dielectrics_csv(inputs['Settings']['epsi2'], csv_data='dielectric_constant')
        else:
            lambda2, n2 = None, np.array(inputs['Settings']['epsi2'], dtype=np.complex128)

    # Create refractive index array of the whole space
    if inputs['Settings']['ModeName'] == 'wavelength':
        try:
            n_length = np.max([n0.size,n1.size,n2.size])
            n_max = np.argmax([n0.size,n1.size,n2.size])
            # Relative complex refractive index
            ni = np.zeros([n_length,3], dtype=np.complex128)
            ni[:,0] = n0
            ni[:,1] = n1
            ni[:,2] = n2
            wavelength = [lambda0,lambda1,lambda2][n_max]
        except:
            n_length = np.max([n0.size,n1.size])
            n_max = np.argmax([n0.size,n1.size])
            # Relative complex refractive index
            ni = np.zeros([n_length,2], dtype=np.complex128)
            ni[:,0] = n0
            ni[:,1] = n1
            wavelength = [lambda0,lambda1][n_max]
    elif inputs['Settings']['lambda_i'] == inputs['Settings']['lambda_f']:
        if lambda0 is not None:
            n0 = n0[lambda0 == inputs['Settings']['lambda_i']]
        if lambda1 is not None:
            n1 = n1[lambda1 == inputs['Settings']['lambda_i']]
        try:
            if lambda2 is not None:
                n2 = n2[lambda2 == inputs['Settings']['lambda_i']]
            ni = np.array([n0,n1,n2],dtype=np.complex128)
        except:
            ni = np.array([n0,n1],dtype=np.complex128)
    else:
        print(tc.str_red('Error occurs in creating refractive index array.'))
        print(tc.str_red('lambda_i = lambda_f. Otherwise, the calculation mode should be the wavelength mode.'))
        sys.exit('Error in read_settings_json.')   
    
    inputs['Settings']['ni'] = ni
    if inputs['Settings']['ModeName'] == 'wavelength':
        inputs['Settings']['wavelength'] = wavelength
    else:
        inputs['Settings']['wavelength'] = inputs['Settings']['lambda_i']

    inputs['Settings'].pop('epsi0', None)
    inputs['Settings'].pop('epsi1', None)
    inputs['Settings'].pop('epsi2', None)
    inputs['Settings'].pop('lambda_i', None)
    inputs['Settings'].pop('lambda_f', None)

    if inputs['Settings']['ModeName'] == 'angle':
        inputs['Settings']['Theta_i'] *= (np.pi/180)
        inputs['Settings']['Theta_f'] *= (np.pi/180)
        (inputs['Settings']['TestDipole']['Pos_Sph'])[1] *= (np.pi/180)
        (inputs['Settings']['TestDipole']['Pos_Sph'])[2] *= (np.pi/180)

    # Preprocessing (initialize settings of a calculation)
    k0 = 2*np.pi / np.array(inputs['Settings']['wavelength'])
    if k0.size==1:
        k0br = k0*np.array(inputs['Settings']['BoundaryRadius'])
    else:
        k0br = np.outer(k0, np.array(inputs['Settings']['BoundaryRadius']))

    inputs['Settings']['k0'] = k0
    inputs['Settings']['k0br'] = k0br

    return inputs

if __name__ == '__main__':
    test = read_settings_json('./input_json/Demo_AngleMode_CF.json')
    print(test)