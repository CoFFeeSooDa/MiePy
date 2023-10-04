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
import functions.mie_coefficient as mc


# Main class to compute properties of Mie scattering 
class MiePy(object):
    """MiePy

    Args:
        theta (float): Polar angle (rad)
        n_max (int): Maximum expansion order
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
        self.source_dipole = Dipole(settings['SourceDipole'],self._log)
        self.test_dipole   = Dipole(settings['TestDipole'],self._log)

        self.test_dipole.pos_sph2 = \
            ct.cartesian_to_spherical(np.array(settings['TestDipole']['Pos_Cart']) - \
                                      np.array(settings['SourceDipole']['Pos_Cart']), self._log)

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
    
    def total_electric_field(self):
        # Calculate the electric dipole field (in the secondary spherical coordinate)
        electric_dipole_field = self._dipole_source_electric_field()
        # Transform the electric dipole field to the primary spherical coordinate
        electric_dipole_field = ct.vector_spherical_to_spherical(electric_dipole_field, 
                                  self.test_dipole.pos_sph2[1] - self.test_dipole.pos_sph[1], 
                                  self.test_dipole.pos_sph2[2] - self.test_dipole.pos_sph[1],self._log)
        # Calculate source coefficients
        p, q = self._dipole_source_coefficient()
        p, q = p[:,:,np.newaxis], q[:,:,np.newaxis]
        # Find the region in which the test dipole is located
        ind = find_r_region(self.boundary_radius, self.test_dipole.pos_sph[0])
        if ind == 0:
            # Calculate Mie coefficients
            alpha, beta = self._mie_coefficient()[:2]
            alpha, beta = alpha[:,:,np.newaxis], beta[:,:,np.newaxis]
            # Calculate vector spherical function at the test dipole position
            M_test, N_test = self._vector_spherical_function('test', '3')
            # Scattering electric field
            E_N = np.einsum('ijk->k', p * alpha * N_test).reshape([3,1])
            E_M = np.einsum('ijk->k', q * beta  * M_test).reshape([3,1])
            # Total Electric field
            electric_field = electric_dipole_field + E_N + E_M
        elif ind == 1: # temporary for single sphere only
            # Calculate source coefficient
            gamma, delta = self._mie_coefficient()[2:]
            gamma, delta = gamma[:,:,np.newaxis], delta[:,:,np.newaxis]
            M_test, N_test = self._vector_spherical_function('test', '1')
            # Scattering electric field
            E_N = np.einsum('ijk->k', p * delta * N_test).reshape([3,1])
            E_M = np.einsum('ijk->k', q * gamma * M_test).reshape([3,1])
            # Total Electric field
            electric_field = electric_dipole_field + E_N + E_M 

        return electric_field
        
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
            r, theta, phi = self.source_dipole.pos_sph
            # Calculate normalized Tau, Pi and P angular functions
            NPi, NTau, NP = bf.normTauPiP(theta, n_max, 'reversed', self._log)
            # Calculate azimuthal function
            azi_func = bf.exp_imphi(phi, n_max, 'reversed',self._log)
            # Find the region index of the source dipole
            index = find_r_region(self.boundary_radius,r)
            # Define a dimensionless radial variable
            kr = self.nr[index] * self.k0 * r
        elif dipole_type == 'test':
            r, theta, phi = self.test_dipole.pos_sph
            # Calculate normalized Tau, Pi and P angular functions
            NPi, NTau, NP = bf.normTauPiP(theta, n_max, 'normal', self._log)
            # Calculate azimuthal function
            azi_func = bf.exp_imphi(phi, n_max, 'normal',self._log)
            # Find the region index of the test dipole
            index = find_r_region(self.boundary_radius,r)
            # Define a dimensionless radial variable
            kr = self.nr[index] * self.k0 * r

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

    def _dipole_source_coefficient(self):
        """_dipole_source_coefficient

        Args:
            self (object): object defined by MiePy
            
        Returns:
            Tuple: p, q, r, s (returned values are determined by the location of the source dipole)
                source in region 0: returns p and q
                source in region -1: returns r and s
                others: returns p, q, r, and s
            
        """
        n = self.expansion_order
        
        # Find the location (region) of the source dipole
        index = find_r_region(self.boundary_radius, self.source_dipole.pos_sph[0])
        # Wavenumber in the dielectric medium
        k = self.nr[index] * self.k0
        # Prefactor (for electric field, cgs but cm -> m)
        prefactor = 4 * np.pi * 1j * k**3

        # Calculate M and N fields for the source dipole according to the location
        
        if index == 0:
            M, N = self._vector_spherical_function(dipole_type='source', function_type='3')
            p = prefactor*np.einsum('ijk,k->ij',N,self.source_dipole.ori_sph)
            q = prefactor*np.einsum('ijk,k->ij',M,self.source_dipole.ori_sph)
            return p, q
        elif index == self.nr.size-1:
            M, N = self._vector_spherical_function(self, dipole_type='source', function_type='1')
            r = prefactor*np.einsum('ijk,k->ij',N,self.source_dipole.ori_sph)
            s = prefactor*np.einsum('ijk,k->ij',M,self.source_dipole.ori_sph)
            return r, s
        else:
            M1, N1 = self._vector_spherical_function(self, dipole_type='source', function_type='1')
            r = prefactor*np.einsum('ijk,k->ij',N1,self.source_dipole.ori_sph)
            s = prefactor*np.einsum('ijk,k->ij',M1,self.source_dipole.ori_sph)
            M3, N3 = self._vector_spherical_function(self, dipole_type='source', function_type='3')
            p = prefactor*np.einsum('ijk,k->ij',N3,self.source_dipole.ori_sph)
            q = prefactor*np.einsum('ijk,k->ij',M3,self.source_dipole.ori_sph)
            return p, q, r, s

    def _dipole_source_electric_field(self):
        """_dipole_source_electric_field

        Args:
            self (object): object defined by MiePy
            
        Returns:
            electric_dipole_field (3x1 array): electric field of source dipole in the secondary coordinate
            
        """

        r, theta, phi = self.test_dipole.pos_sph2
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
        electric_dipole_field = (Nx*self.source_dipole.ori_cart[0] + \
                                 Ny*self.source_dipole.ori_cart[1] + \
                                 Nz*self.source_dipole.ori_cart[2]) * self.nr[index]

        return electric_dipole_field

    def _mie_coefficient(self):
        alpha, beta, gamma, delta = mc.mie_single0(self.nr, self.k0br, self.expansion_order, self._log)
        return alpha, beta, gamma, delta



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
            self.ori_cart = np.array(inputs['Ori_Cart'], dtype=np.float64)
            self.ori_sph  = ct.vector_transformation(inputs['Ori_Cart'], self.pos_sph[1:], 'c2s', log_message)
        elif 'Ori_Sph' in inputs.keys():
            self.ori_cart = ct.vector_transformation(inputs['Ori_Sph'], self.pos_sph[1:], 's2c', log_message)
            self.ori_sph  = np.array(inputs['Ori_Sph'], dtype=np.float64)
        elif log_message:
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