# Standard imports:
import logging
from pathlib import Path
import json
import inspect

# External imports:
import numpy as np

# My functions
import functions.coordinate_transformation as ct
import functions.basis_function as bf
import functions.text_color as tc


# Main class to compute properties of Mie scattering 
class MiePy(object):
    """MiePy

    Args:
        theta (float): Polar angle (rad)
        nmax (int): Maximum expansion order
        order (string): Ordering of tables ('normal' or 'reversed')
        log_message (object): object of logging standard module (for the use of MiePy logging only)
        
    Returns:
        Tuple: Normalized Tau , Pi , and P functions
        NTau (ndarray[float], n x (2n+1)): Normalized Tau array
        NPi (ndarray[float], n x (2n+1)): Normalized Pi array
        NP (ndarray[float], n x (2n+1)): Normalized P array

    Calling functions:
        Wigner_d (ndarray[float], (2j+1)x(2j+1)): Wigner d matrix
    """
    def __init__(self,input_path='./input_json/Demo_AngleMode_CF.json',debug_folder=None,output_debug_file=False):
        # Logger setup
        self._output_debug_file = output_debug_file
        if output_debug_file == True:
            self._debug_folder = None
            if debug_folder is None:
                self.debug_folder = Path(__file__).parent / 'debug_log'
            else:
                self.debug_folder = debug_folder
        self._log = logging.getLogger('MiePy')
        if not self._log.hasHandlers():
            # Create new handlers to _log
            self._log.setLevel(logging.DEBUG)
            # Console handler at INFO level
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # Log file handler at DEBUG level 
            if output_debug_file == True:
                lf = logging.FileHandler(self._debug_folder / 'MiePy_log.txt')
                lf.setLevel(logging.DEBUG)
            # Format
            format = logging.Formatter('%(asctime)s:%(name)s-%(levelname)s: %(message)s')
            ch.setFormatter(format)
            self._log.addHandler(ch)
            if output_debug_file == True:
                lf.setFormatter(format)
                self._log.addHandler(lf)
        
        # Load input file (.json)
        self._log.info('Read json file: ' + input_path)
        inputs = read_json(input_path)
        
        # Setting objects of the source dipole and test dipole
        self.source_dipole = Dipole(inputs['Settings']['SourceDipole'])
        self.test_dipole   = Dipole(inputs['Settings']['TestDipole'])
        
        # Setting expansion order
        self.expansion_order = inputs['ExpansionOrder']
        self._log.info(f'Maximum multipole: {self.expansion_order}')

        # Set boundary condition
        self.boundary_condition = inputs['BoundaryCondition']

        # Set radii of the boundary
        self.boundary_radius = inputs['BoundaryRadius']

        # Other variables that need to be initialized...

    @property
    def debug_folder(self):
        #Get or set the path for debug logging. Will create folder if not existing.
        return self._debug_folder

    @debug_folder.setter
    def debug_folder(self, path):
        # Do not do logging in here! This will be called before the logger is set up
        assert isinstance(path, Path), 'Must be pathlib.Path object'
        if self._output_debug_file == True:
            if path.is_file():
                path = path.parent
            if not path.is_dir():
                path.mkdir(parents=True)
            self._debug_folder = path
    
    '''
    # Display attributes
    def display_inputs_attribute(self):    
        find_attribute(self.inputs,'inputs')
        #for attr in user_defined_attributes:
        #    print(f"{attr}: {getattr(self.inputs, attr)}")
    
    def display_attribute(self):
        find_attribute(self.settings,'settings')
    '''
    
    
    # Construct vector spherical functions (M and N)
    #def vector_spherical_function(self,)
    
    #Test block
    def test(self):
        #self._log.info('It is a info text')
        #self._log.debug('It is a debug text')
        #self.DPos = Position([2,2,0],type='cartesian',log_message=self._log)
        #print(f'{self.DPos.cartesian=}')
        #print(f'{self.DPos.spherical=}')
        #ct.spherical_to_spherical([3,0,1.57],1,0,self._log)
        #bf.normTauPiP(1,11,'normal',self._log)
        print(str(self.source_dipole.pos_cart))
        print(str(self.source_dipole.pos_sph))
        print(str(self.test_dipole.pos_cart))
        print(str(self.test_dipole.pos_sph))
        

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

def read_json(path):
            with open(path) as f:
                # Load inputs
                inputs = json.load(f)
                # Convert inputs dictionaries to an object
                return inputs

def vector_spherical_function(inputs: object, dipole_type: str, function_type: str):
    """vector_spherical_function

    Args:
        input (object): object defined by MiePy
        dipole_type: string to determine the type of dipole
        function_type : type of the vector spherical functions
        log_message (object): object of logging standard module (for the use of MiePy logging only)
        
    Returns:
        Tuple: Normalized M and N functions
        M (ndarray[float], n x (2n+1) x 3): vector spherical function M
        N (ndarray[float], n x (2n+1) x 3): vector spherical function N
        
    """
    # Assign the max order of n
    n_max = np.int16(input.expansion_order)
    # Assign r, theta, and phi determined by the type of dipole
    if dipole_type == 'source_dipole':
         r = input.source_dipole.pos_sph[0]
         theta = input.source_dipole.pos_sph[1]
         phi = input.source_dipole.pos_sph[2]
    elif dipole_type == 'test_dipole':
         r = input.test_dipole.pos_sph[0]
         theta = input.test_dipole.pos_sph[1]
         phi = input.test_dipole.pos_sph[2]
    # Assign the dimensionless radial variable
    k = input.tmp['k']
    kr = k * r

    # Preallocation
    M = np.zeros([n_max,2*n_max+1,3])
    N = np.zeros([n_max,2*n_max+1,3])

    # Radial function
    if function_type == '1':
        z = bf.spherical_bessel_function_1(kr, n_max, input._log)
        raddz = bf.riccati_bessel_function_S(kr, n_max, input._log)[1] / kr
    elif function_type == '3':
        z = bf.spherical_hankel_function_1(kr, n_max, input._log)
        raddz = bf.riccati_bessel_function_xi(kr, n_max, input._log)[1] / kr
    
    z = z[1:]
    raddz = z[1:]

    # Angular function
    n = np.linspace(1,n_max,n_max).reshape(n_max,1)

    # Calculate Radz (z_n(kr)/kr)
    if kr == 0:
         Radz = np.zeros(n_max,1)
         Radz[0] = 1/3
    else:
         Radz = z / kr

    return M, N
    



class Dipole(object):
    """Dipole

    Args:
        input (list): inputs of dipoles from .json files
        log_message (object): object of logging standard module (for the use of MiePy logging only)
        
    Attributes:
        Pos_Cart : position vector in Cartesian coordinates
        Pos_Sph  : position vector in spherical coordinates
        Ori_Cart : orientation vector in Cartesian coordinates

    Calling functions:
        cartesian_to_spherical and spherical_to_cartesian in basis_function
    """
    def __init__(self, inputs:list,log_message=None):
        # Set position vector
        if 'Pos_Cart' in inputs.keys():
            self.pos_cart = np.array(inputs['Pos_Cart'], dtype=np.float64)
            self.pos_sph  = ct.cartesian_to_spherical(inputs['Pos_Cart'], log_message)
        elif 'Pos_Sph' in inputs.keys():
            self.pos_cart = ct.spherical_to_cartesian(inputs['Pos_Sph'], log_message)
            self.pos_sph  = np.array(inputs['Pos_Sph'], dtype=np.float64)
        else:
            log_message.error('Incorrect attribute from position vector.')
            log_message.error('Attribute must be "Cart" or "Sph"')

        # Set orientation vector
        '''
        if 'Ori_Cart' in inputs.keys():
            self.Pos_Cart = np.array(inputs['Ori_Cart'], dtype=np.float64)
            self.Pos_Sph  = ct.cartesian_to_spherical(inputs['Ori_Cart'], log_message)
        elif 'Ori_Sph' in inputs.keys():
            self.Pos_Cart = ct.spherical_to_cartesian(inputs['Ori_Sph'], log_message)
            self.Pos_Sph  = np.array(inputs['Pos_Sph'], dtype=np.float64)
        else:
            log_message.error('Incorrect attribute from position vector.')
            log_message.error('Attribute must be "Cart" or "Sph"')
        '''          



if __name__ == '__main__':
    #MP = MiePy('./input_json/Demo_AngleMode_CF.json')
    MP = MiePy(output_debug_file=False)
    MP.test()
    #MP.display_attribute()
    
    