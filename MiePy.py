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
    def __init__(self, settings:dict, debug_folder=None, output_debug_file=False):
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
        
        # Return MiePy status
        self._log.info('Loading settings...')

        # Setting objects of the source dipole and test dipole
        self.source_dipole = Dipole(settings['SourceDipole'])
        self.test_dipole  = Dipole(settings['TestDipole'])

        # Setting expansion order
        self.expansion_order = settings['ExpansionOrder']
        self._log.info(f'Maximum multipole: {self.expansion_order}')

        # Set boundary condition
        self.boundary_condition = settings['BoundaryCondition']

        # Set radii of the boundary
        self.boundary_radius = settings['BoundaryRadius']

        # Other variables that need to be initialized...
        self.k0 = settings['k0']
        self.k0br = settings['k0br']
        self.nr = settings['nr']
        
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
    
    def _vector_spherical_function(self, dipole_type: str, function_type: str):
        """_vector_spherical_function

        Args:
            self (object): object defined by MiePy
            dipole_type: string to determine the type of dipole ('source' or 'test')
            function_type : type of the vector spherical functions ('1' or '3')
            log_message (object): object of logging standard module (for the use of MiePy logging only)
            
        Returns:
            Tuple: Normalized M and N functions
            M (ndarray[float], n x (2n+1) x 3): vector spherical function M
            N (ndarray[float], n x (2n+1) x 3): vector spherical function N
            
        """
        # Assign the max order of n
        n_max = np.int16(self.expansion_order)
        # Assign r, theta, and phi determined by the type of dipole
        if dipole_type == 'source':
            r = self.source_dipole.pos_sph[0]
            theta = self.source_dipole.pos_sph[1]
            phi = self.source_dipole.pos_sph[2]
            # Calculate normalized Tau, Pi and P angular functions
            NPi, NTau, NP = bf.normTauPiP(theta, n_max, 'reversed', self._log)
            # Calculate azimuthal function
            azi_func = bf.exp_imphi(phi, n_max, 'reversed',self._log)
        elif dipole_type == 'test':
            r = self.test_dipole.pos_sph[0]
            theta = self.test_dipole.pos_sph[1]
            phi = self.test_dipole.pos_sph[2]
            # Calculate normalized Tau, Pi and P angular functions
            NPi, NTau, NP = bf.normTauPiP(theta, n_max, 'normal', self._log)
            # Calculate azimuthal function
            azi_func = bf.exp_imphi(phi, n_max, 'normal',self._log)

        # Assign the dimensionless radial variable
        kr = self.k * r

        # Preallocation
        M = np.zeros([n_max,2*n_max+1,3],dtype=np.complex128)
        N = np.zeros([n_max,2*n_max+1,3],dtype=np.complex128)

        # Radial function
        if function_type == '1':
            z = bf.spherical_bessel_function_1(kr, n_max, self._log)
            raddz = bf.riccati_bessel_function_S(kr, n_max, self._log)[1] / kr
        elif function_type == '3':
            z = bf.spherical_hankel_function_1(kr, n_max, self._log)
            raddz = bf.riccati_bessel_function_xi(kr, n_max, self._log)[1] / kr
        
        
        z = z[1:]
        raddz = raddz[1:]

        # Angular function
        n = np.linspace(1,n_max,n_max).reshape(n_max,1)

        # Calculate Radz (z_n(kr)/kr)
        if kr == 0:
            radz = np.zeros(n_max,1)
            radz[0] = 1/3
        else:
            radz = z / kr

        # M field
        M[:,:,1] = 1j * z * NPi *  azi_func
        M[:,:,2] = - z * NTau * azi_func

        # N field
        n = np.reshape(np.arange(1,n_max+1),[n_max,1])
        N[:,:,0] = radz * n*(n+1) * NP * azi_func
        N[:,:,1] = raddz * NTau * azi_func
        N[:,:,2] = 1j * raddz * NPi * azi_func

        return M, N

    def _source_coefficient(self):
        """_source_coefficient

        Args:
            self (object): object defined by MiePy
            
        Returns:
            Tuple: Normalized M and N functions
            M (ndarray[float], n x (2n+1) x 3): vector spherical function M
            N (ndarray[float], n x (2n+1) x 3): vector spherical function N
            
        """
        pass

    def _source_dipole_electric_field(self):
        """_source_dipole_electric_field

        Args:
            self (object): object defined by MiePy
            
        Returns:
            EdipS (3x1 array): electric field of source dipole in the secondary coordinate
            
        """

        r, theta, phi = self.source_dipole.pos_sph
        # Find the region index of the source dipole
        index = find_r_region(self.boundary_radius,r)
        # Wavenumber in dielectrics
        k = self.k0 * self.nr[index]
        # Preallocation (N is the vector spherical function)
        Nx, Ny, Nz = np.zeros([3,3,1], dtype=np.complex128)
        # Radial function 
        rad1 = np.exp(1j * k * r)/r * (r**(-2) - 1j*k/r)
        rad2 = np.exp(1j * k * r)/r * (k**2 + 1j*k/r - r**(-2))
        # X-component electric field
        Nx[0] =  rad1 * np.sin(theta) * np.cos(phi) * 2
        Nx[1] =  rad2 * np.cos(theta) * np.cos(phi)
        Nx[2] = -rad2 * np.sin(theta)
        # Y-component electric field
        Ny[0] =  rad1 * np.sin(theta) * np.sin(phi) * 2
        Ny[1] =  rad2 * np.cos(theta) * np.sin(phi)
        Ny[2] =  rad2 * np.cos(theta)
        # Z-component electric field
        Nz[0] =  rad1 * np.cos(theta) * 2
        Nz[1] = -rad2 * np.sin(theta)
        # Electric Dipole Field (Gaussian Unit)
        EdipS = (Nx*self.source_dipole.ori_Cart[0] + \
                 Ny*self.source_dipole.ori_Cart[1] + \
                 Nz*self.source_dipole.ori_Cart[2]) * self.nr[index]

        return EdipS


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
        if 'Ori_Cart' in inputs.keys():
            self.ori_Cart = np.array(inputs['Ori_Cart'], dtype=np.float64)
            # self.Pos_Sph  = ct.cartesian_to_spherical(inputs['Ori_Cart'], log_message)
        elif 'Ori_Sph' in inputs.keys():
            #self.Pos_Cart = ct.spherical_to_cartesian(inputs['Ori_Sph'], log_message)
            self.ori_Sph  = np.array(inputs['Pos_Sph'], dtype=np.float64)
        else:
            log_message.error('Incorrect attribute from position vector.')
            log_message.error('Attribute must be "Cart" or "Sph"')



def find_r_region(boundary_radius:np.ndarray, r:np.float64):
    """find_r_region

    Args:
        boundary_radius (array): descending array of radii for sphere(s)
        r: radius which needs to determine
        
    Output: 
        ind (int): index of the dielectric region
    """
    left, right = 0, len(boundary_radius) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if r == boundary_radius[mid]:
            return mid
        elif r > boundary_radius[mid]:
            right = mid - 1
        else:
            left = mid + 1
    
    return left


if __name__ == '__main__':
    #MP = MiePy('./input_json/Demo_AngleMode_CF.json')
    #MP = MiePy(output_debug_file=False)
    #MP.display_attribute()

    
    ind = find_r_region(np.array([70]), 90)
    print(ind)