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
        
        self._log.info('Function spherical_to_spherical currently only supports z-direction shifting.')

        # Return MiePy status
        self._log.info('Loading settings...')

        # Read mode name
        self.mode_name = settings['ModeName']
        self._log.info(f'Mode is set to "{self.mode_name}"')

        # Set objects of the source dipole and test dipole
        self.source_dipole = Dipole(settings['SourceDipole'],self._log)
        self.test_dipole   = Dipole(settings['TestDipole'],self._log)

        self.test_dipole.pos_sph2 = \
            ct.cartesian_to_spherical(self.test_dipole.pos_cart - self.source_dipole.pos_cart, self._log)

        # Set expansion order
        self.expansion_order = settings['ExpansionOrder']
        self._log.info(f'Maximum multipole: {self.expansion_order}')

        # Set boundary condition
        self.boundary_condition = settings['BoundaryCondition']
        self._log.info(f'Boundary condition: {self.boundary_condition}')

        # Set radii of the boundary
        self.boundary_radius = settings['BoundaryRadius']
        
        # location of source dipole
        self.source_dipole.region = find_r_region(self.boundary_radius,self.source_dipole.pos_sph[0])
        self._log.info(f'Source dipole in REGION {self.source_dipole.region}')

        # Other variables that need to be initialized...
        self.k0 = settings['k0']
        self.k0br = settings['k0br']
        self.ni = settings['ni']

        # Speed Optimization according to the selected mode
        # (prevent from redundant computation)
        if self.mode_name == 'wavelength':
            # Calculate normalized Tau, Pi and P angular functions for the source dipole
            NPi_s, NTau_s, NP_s = \
                bf.normTauPiP(self.source_dipole.pos_sph[1], self.expansion_order, 'reversed', self._log)
            # Calculate azimuthal function for the source dipole
            azi_func_s = \
                bf.exp_imphi(self.source_dipole.pos_sph[2], self.expansion_order, 'reversed',self._log)
            # Calculate normalized Tau, Pi and P angular functions for the source dipole
            NPi_t, NTau_t, NP_t = \
                bf.normTauPiP(self.test_dipole.pos_sph[1], self.expansion_order, 'normal', self._log)
            # Calculate azimuthal function for the source dipole
            azi_func_t = \
                bf.exp_imphi(self.test_dipole.pos_sph[2], self.expansion_order, 'normal',self._log)
            
            self.speed_up = {"NPi_s": NPi_s, "NTau_s": NTau_s, "NP_s": NP_s, "azi_func_s": azi_func_s,
                             "NPi_t": NPi_t, "NTau_t": NTau_t, "NP_t": NP_t, "azi_func_t": azi_func_t}
        elif self.mode_name == 'angle':
            # Calculate source coefficients
            p, q = self._source_dipole_coefficient()
            p, q = p[:,:,np.newaxis], q[:,:,np.newaxis]
            # Find the region in which the test dipole is located
            ind = find_r_region(self.boundary_radius, self.test_dipole.pos_sph[0])
            kr = self.ni[ind] * self.k0 * self.test_dipole.pos_sph[0]
            if ind == 0:
                # Calculate radial functions of test dipole
                z_t = bf.spherical_hankel_function_1(kr, self.expansion_order, self._log)
                raddz_t = bf.riccati_bessel_function_xi(kr, self.expansion_order, self._log)[1] / kr
                # Calculate Mie coefficients
                alpha, beta = self._mie_coefficient()[:2]
                alpha, beta = alpha[:,:,np.newaxis], beta[:,:,np.newaxis]
                self.speed_up = {'p': p, 'q': q, 'alpha':alpha, 'beta':beta}
            elif ind == 1:
                # Calculate radial functions of test dipole
                z_t = bf.spherical_bessel_function_1(kr, self.expansion_order, self._log)
                raddz_t = bf.riccati_bessel_function_S(kr, self.expansion_order, self._log)[1] / kr
                # Calculate source coefficient
                gamma, delta = self._mie_coefficient()[2:]
                gamma, delta = gamma[:,:,np.newaxis], delta[:,:,np.newaxis]
                self.speed_up = {'p': p, 'q': q, 'gamma':gamma, 'delta':delta}
        else:
            self.speed_up = {}
            
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
    
    def refresh(self, calc:dict):
        # Other variables that need to be refresh...
        if self.mode_name == 'wavelength':
            self.k0 = calc['k0']
            self.k0br = calc['k0br']
            self.ni = calc['ni']
        elif self.mode_name == 'angle':
            # Recalculate the test dipole
            self.test_dipole   = Dipole(calc['TestDipole'],self._log)

            self.test_dipole.pos_sph2 = \
                ct.cartesian_to_spherical(self.test_dipole.pos_cart - self.source_dipole.pos_cart, self._log)

    def output_elapsed_time(self, elapsed_time):
        self._log.info('Job Completed.')
        self._log.info(f'Elapsed time: {elapsed_time} seconds')
    
    def coupling_factor(self):
        # cgs unit but cm is replaced by m (CF unit: m^-6)
        return np.abs(self.test_dipole.ori_sph @ self.total_electric_field())**2
    
    def source_dipole_electric_field(self):
        """source_dipole_electric_field

        Args:
            self.expansion_order (np.int16): Expansion order of the vector spherical functions
            self.self.test_dipole.pos_sph2 (ndarray[np.float64] 1x3): Position of the test dipole in the secondary spherical coordinate
            self.self.test_dipole.pos_sph  (ndarray[np.float64] 1x3): Position of the test dipole in the primary spherical coordinate
            self._log (object): Object of logging standard module (for the use of MiePy logging only)
            self.speed_up (dict): For reducing redundant computations

        Methods:
            self._source_dipole_electric_field (ndarray[np.complex128] 3x1): Electric field of the source dipole in the secondary coordinate
           
        Returns:
            electric_dipole_field (ndarray[np.complex128] 3x1): Electric field of the source dipole in the primary coordinate
            
        Calling functions:
            ct.vector_spherical_to_spherical (ndarray[np.float64], 3x1): Trasform to the primary coordinate
            
        """

        try:
            electric_dipole_field = self.speed_up['electric_dipole_field']
        except:
            # Calculate the electric dipole field (in the secondary spherical coordinate)
            electric_dipole_field = self._source_dipole_electric_field()
            # Transform the electric dipole field to the primary spherical coordinate
            electric_dipole_field = ct.vector_spherical_to_spherical(electric_dipole_field, 
                                      self.test_dipole.pos_sph2[1] - self.test_dipole.pos_sph[1], 
                                      self.test_dipole.pos_sph2[2] - self.test_dipole.pos_sph[2],self._log)
        
        return electric_dipole_field

    def total_electric_field(self):
        """total_electric_field

        Args:
            self.expansion_order (np.int16): Expansion order of the vector spherical functions
            self.source_dipole.pos_sph (ndarray[np.float64] 1x3): Position of the source dipole (spherical coordiante)
            self.source_dipole.region (np.int16): Index of the region where the source dipole is located
            self.k0 (np.float64): Wavenumber in vacuum
            self.ni (ndarray[np.complex128] 1xm): Complex refractive indices in each region (m = 2, 3)
            log_message (object): Object of logging standard module (for the use of MiePy logging only)

        Methods:
            self.source_dipole_electric_field (ndarray[np.complex128] 3x1): Electric field of the source dipole in the primary spherical coordinate
            self._source_dipole_coefficient(tuple): Expansion coefficients of a electric dipole (p, q, r, s)
            self._mie_coefficient (tuple): Mie coefficients (alpha, beta, gamma and delta)
            self._vector_spherical_function (tuple): Calculate normalized vector spherical functions (M and N)
           
        Returns:
            Tuple: Normalized M and N functions
                M (ndarray[float], n x (2n+1) x 3): vector spherical function M
                N (ndarray[float], n x (2n+1) x 3): vector spherical function N
        
        Calling functions:
            ct.vector_spherical_to_spherical (ndarray[np.float64], 3x1): Trasform to the primary coordinate
            
        """
        
        # Calculate source-dipole electric field in the primary spherical coordinate
        electric_dipole_field = self.source_dipole_electric_field()
        # Calculate source coefficients
        try:
            p = self.speed_up['p']
            q = self.speed_up['q']
        except:
            p, q = self._source_dipole_coefficient()
            p, q = p[:,:,np.newaxis], q[:,:,np.newaxis]
        # Find the region in which the test dipole is located
        ind = find_r_region(self.boundary_radius, self.test_dipole.pos_sph[0])
        if ind == 0:
            # Calculate Mie coefficients
            try:
                alpha = self.speed_up['alpha']
                beta  = self.speed_up['beta']
            except:
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
            try:
                gamma = self.speed_up['gamma']
                delta = self.speed_up['delta']
            except:
                gamma, delta = self._mie_coefficient()[2:]
                gamma, delta = gamma[:,:,np.newaxis], delta[:,:,np.newaxis]
            M_test, N_test = self._vector_spherical_function('test', '1')
            # Scattering electric field
            E_N = np.einsum('ijk->k', p * delta * N_test).reshape([3,1])
            E_M = np.einsum('ijk->k', q * gamma * M_test).reshape([3,1])
            # Total Electric field
            electric_field = E_N + E_M 

        return electric_field
    
    def dyadic_greens_function_scattering(self):
        """dyadic_greens_function_scattering
        Args:
            self.expansion_order (np.int16): Expansion order of the vector spherical functions
            self.source_dipole.pos_sph (ndarray[np.float64] 1x3): Position of the source dipole (spherical coordiante)
            self.source_dipole.region (np.int16): Index of the region where the source dipole is located
            self.k0 (np.float64): Wavenumber in vacuum
            self.ni (ndarray[np.complex128] 1xm): Complex refractive indices in each region (m = 2, 3)
            log_message (object): Object of logging standard module (for the use of MiePy logging only)

        Methods:
            self._mie_coefficient (tuple): Mie coefficients (alpha, beta, gamma and delta)
            self._vector_spherical_function (tuple): 
                Calculate normalized vector spherical functions (M and N)
                For Source Dipole Moiety:
                    self._vector_spherical_function(dipole_type='source', function_type='3')
                For Test Dipole Moiety:
                    ind = 0: self._vector_spherical_function(dipole_type='test', function_type='3')
                    ind = 1: self._vector_spherical_function(dipole_type='test', function_type='1')
                Returns:
                    Tuple: Normalized M and N functions
                        M (ndarray[float], n x (2n+1) x 3): vector spherical function M
                        N (ndarray[float], n x (2n+1) x 3): vector spherical function N
                        
        Returns:
            GF_scat(ndarray[complex], 3 x 3): scattering dyadic Green's Function for a certain frequency
            
        Note:
            How to obtain scattering electric field for a certain frequency using GF_scat?
            
            prefactor = 4 * np.pi * k**2
            electric_field_scat = prefactor
                                * np.einsum('ij, j -> i', GF_scat, self.source_dipole.ori_sph)
        """
        # Preliminaries: Variables from MiePy object
        ## Wavenumber in the dielectric medium
        k = self.ni[self.source_dipole.region] * self.k0
        ## Prefactor (for electric field, cgs but cm -> m)
        prefactor = 1j * k
        
        # Source Dipole Moiety
        # Calculate vector spherical function at the source dipole position
        M_source, N_source = self._vector_spherical_function(dipole_type='source', function_type='3')
        #Dim: Axis: (0, 1, 2) = (n, -m(total 2n+1), 3)
        
        # Test Dipole Moiety
        # Find the region in which the test dipole is located
        ind = find_r_region(self.boundary_radius, self.test_dipole.pos_sph[0])
        if ind == 0:
            # Calculate Mie coefficients
            try:
                alpha = self.speed_up['alpha']
                beta  = self.speed_up['beta']
            except:
                alpha, beta = self._mie_coefficient()[:2]
                #Dim: Axis: (0, 1) = (n, 1)
                alpha = alpha[:, 0] #Dim: Axis: (0) = (n), same effect as np.ravel()
                beta = beta[:, 0] #Dim: Axis: (0) = (n), same effect as np.ravel() 
            # Calculate vector spherical function at the test dipole position
            M_test, N_test = self._vector_spherical_function('test', '3')
            #Dim: Axis: (0, 1, 2) = (n, m(total 2n+1), 3)
            GF_scat_N = prefactor * np.einsum('n, nmk, nml -> kl', alpha, N_test, N_source)
            #Dim: Axis: (0, 1) = (3, 3)
            GF_scat_M = prefactor * np.einsum('n, nmk, nml -> kl', beta, M_test, M_source)
            #Dim: Axis: (0, 1) = (3, 3)
            GF_scat = GF_scat_N + GF_scat_M
            
        elif ind == 1: # temporary for single sphere only
            # Calculate source coefficient
            try:
                gamma = self.speed_up['gamma']
                delta = self.speed_up['delta']
            except:
                gamma, delta = self._mie_coefficient()[2:]
                #Dim: Axis: (0, 1) = (n, 1)
                gamma = gamma[:, 0] #Dim: Axis: (0) = (n), same effect as np.ravel()
                delta = delta[:, 0] #Dim: Axis: (0) = (n), same effect as np.ravel()
            # Calculate vector spherical function at the test dipole position
            M_test, N_test = self._vector_spherical_function('test', '1')
            #Dim: Axis: (0, 1, 2) = (n, m(total 2n+1), 3)
            GF_scat_N = prefactor * np.einsum('n, nmk, nml -> kl', delta, N_test, N_source)
            #Dim: Axis: (0, 1) = (3, 3)
            GF_scat_M = prefactor * np.einsum('n, nmk, nml -> kl', gamma, M_test, M_source)
            #Dim: Axis: (0, 1) = (3, 3)
            GF_scat = GF_scat_N + GF_scat_M
        
        return GF_scat
        
    def _vector_spherical_function(self, dipole_type: str, function_type: str):
        """_vector_spherical_function

        Args:
            dipole_type (str): Type of dipole ('source' or 'test')
            function_type (str): Type of function ('1' or '3') 
            self.expansion_order (np.int16): Expansion order of the vector spherical functions
            self.source_dipole.pos_sph (ndarray[np.float64] 1x3): Position of the source dipole (spherical coordiante)
            self.source_dipole.region (np.int16): Index of the region where the source dipole is located
            self.k0 (np.float64): Wavenumber in vacuum
            self.ni (ndarray[np.complex128] 1xm): Complex refractive indices in each region (m = 2, 3)
            log_message (object): Object of logging standard module (for the use of MiePy logging only)
            
        Returns:
            Tuple: Normalized M and N functions
                M (ndarray[float], n x (2n+1) x 3): Vector spherical function M
                N (ndarray[float], n x (2n+1) x 3): Vector spherical function N
            
        """
        # Assign the max order of n
        n_max = self.expansion_order
        # Assign r, theta, and phi determined by the type of dipole
        if dipole_type == 'source':
            r, theta, phi = self.source_dipole.pos_sph
            # Calculate normalized Tau, Pi and P angular functions
            try:
                NPi = self.speed_up['NPi_s']
                NTau = self.speed_up['NTau_s']
                NP = self.speed_up['NP_s']
            except:
                NPi, NTau, NP = bf.normTauPiP(theta, n_max, 'reversed', self._log)
            # Calculate azimuthal function
            try:
                azi_func = self.speed_up['azi_func_s']
            except:
                azi_func = bf.exp_imphi(phi, n_max, 'reversed',self._log)
            # Define a dimensionless radial variable
            kr = self.ni[self.source_dipole.region] * self.k0 * r
        elif dipole_type == 'test':
            r, theta, phi = self.test_dipole.pos_sph
            # Calculate normalized Tau, Pi and P angular functions
            try:
                NPi = self.speed_up['NPi_t']
                NTau = self.speed_up['NTau_t']
                NP = self.speed_up['NP_t']
            except:
                NPi, NTau, NP = bf.normTauPiP(theta, n_max, 'normal', self._log)
            # Calculate azimuthal function
            try:
                azi_func = self.speed_up['azi_func_t']
            except:
                azi_func = bf.exp_imphi(phi, n_max, 'normal',self._log)
            # Define a dimensionless radial variable
            ind = find_r_region(self.boundary_radius, self.test_dipole.pos_sph[0])
            kr = self.ni[ind] * self.k0 * r

        # Preallocation
        M = np.zeros([n_max,2*n_max+1,3],dtype=np.complex128)
        N = np.zeros([n_max,2*n_max+1,3],dtype=np.complex128)

        # Radial function
        if dipole_type == 'test' and ('z' and 'raddz' in self.speed_up):
            z = self.speed_up['z_t']
            raddz = self.speed_up['raddz_t']
        else:
            if function_type == '1':
                z = bf.spherical_bessel_function_1(kr, n_max, self._log)
                raddz = bf.riccati_bessel_function_S(kr, n_max, self._log)[1] / kr
            elif function_type == '3':
                z = bf.spherical_hankel_function_1(kr, n_max, self._log)
                raddz = bf.riccati_bessel_function_xi(kr, n_max, self._log)[1] / kr
        
        # Exclude the zeroth-order spherical Bessel (Hankel) function
        z = z[1:]
        # Exclude the zeroth-order Riccati-Bessel function
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

    def _source_dipole_coefficient(self):
        """_source_dipole_coefficient

        Args:
            self.ni (ndarray[np.complex128] 1xm): Complex refractive indices in each region (m = 2, 3)
            self.source_dipole.region (np.int16): Index of the region where the source dipole is located
            self.k0 (np.float64): Wavenumber in vacuum
            self.source_dipole.ori_cart (ndarray[np.float64] 1x3): Orientation of the source dipole (Cartesian coordinate)
        
        Methods:
            self._vector_spherical_function (tuple): Calculate normalized vector spherical functions (M and N)
                M (ndarray[np.complex128], n x (2n+1) x 3): Vector spherical function M
                N (ndarray[np.complex128], n x (2n+1) x 3): Vector spherical function N
            
        Returns:
            Tuple: Expansion coefficients (p, q, r, s) of a electric dipole (returned values are based on the location of the source dipole)
                Source in region  0 (outermost region): Returns p and q
                Source in region -1 (innermost region): Returns r and s
                Others: Returns p, q, r, and s
            
        """

        # Wavenumber in the dielectric medium
        k = self.ni[self.source_dipole.region] * self.k0
        # Prefactor (for electric field, cgs but cm -> m)
        prefactor = 4 * np.pi * 1j * k**3

        # Calculate M and N fields based on the location of the source dipole
        if self.source_dipole.region == 0:
            M, N = self._vector_spherical_function(dipole_type='source', function_type='3')
            p = prefactor*np.einsum('ijk,k->ij',N,self.source_dipole.ori_sph)
            q = prefactor*np.einsum('ijk,k->ij',M,self.source_dipole.ori_sph)
            return p, q
        elif self.source_dipole.region == self.ni.size-1:
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

    def _source_dipole_electric_field(self):
        """_source_dipole_electric_field

        Args:
            self.test_dipole.pos_sph2 (ndarray[np.float64] 1x3): Position of the test dipole in the secondary spherical coordinate
            self.k0 (np.float64): Wavenumber in vacuum
            self.ni (ndarray[np.complex128] 1xm): Complex refractive indices in each region (m = 2, 3)
            self.source_dipole.region (np.int16): Index of the region where the source dipole is located
            self.source_dipole.ori_cart (ndarray[np.float64] 1x3): Orientation of the source dipole (Cartesian coordinate)

        Returns:
            electric_dipole_field (ndarray[np.complex128] 3x1): Electric field of the source dipole in the secondary coordinate
            
        """

        r, theta, phi = self.test_dipole.pos_sph2
        # Wavenumber in dielectrics
        k = self.k0 * self.ni[self.source_dipole.region]
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
                                 Nz*self.source_dipole.ori_cart[2]) * self.ni[self.source_dipole.region]

        return electric_dipole_field

    def _mie_coefficient(self):
        """_mie_coefficient

        Args:
            self.boundary_condition (str): Either 'single' or 'coreshell'
            self.source_dipole.region (np.int16): Index of the region where the source dipole is located
            self.ni (ndarray[np.complex128], 1 x m): Complex refractive indices of each region (m = 2, 3)
            self.boundary_radius (ndarray[np.float64], 1 x (m-1)): Radius of spherical boundary
            self.k0br (ndarray[np.float64], 1 x (m-1)): k0 * boundary_radious (m = 2, 3)
            self.expansion_order (np.int16): Expansion order of vector spherical functions
            log_message (object): Object of logging standard module (for the use of MiePy logging only)
            
        Returns:
            Tuple: Mie coefficients of alpha0, beta0, gamma1 and delta1 (Only for single sphere currently)
                alpha0 (ndarray[float], n x 1): alpha0 coefficient
                beta0  (ndarray[float], n x 1): beta0 coefficient
                gamma1 (ndarray[float], n x 1): gamma1 coefficient
                delta1 (ndarray[float], n x 1): delta1 coefficient
            Tuple: Mie coefficients of alpha0, beta0, gamma1 and delta1 (For core/shell sphere, TODO)
                alpha0 (ndarray[float], n x 1): alpha0 coefficient
                beta0  (ndarray[float], n x 1): beta0 coefficient
                alpha1 (ndarray[float], n x 1): alpha1 coefficient
                beta1  (ndarray[float], n x 1): beta1 coefficient
                gamma1 (ndarray[float], n x 1): gamma1 coefficient
                delta1 (ndarray[float], n x 1): delta1 coefficient
                gamma2 (ndarray[float], n x 1): gamma2 coefficient
                delta2 (ndarray[float], n x 1): delta2 coefficient

        Calling functions:
            functions.mie_coefficient: Functions of Mie coefficients
        """
        # Calculate coefficients based on the boundary condition(s)
        if self.boundary_condition == 'single':
            # Find the region at which the source dipole is located
            if self.source_dipole.region == 0:
                # Call the function for the source dipole located at region 0
                alpha0, beta0, gamma1, delta1 = mc.mie_single0(self.ni, self.k0br, self.expansion_order, self._log)
            else:
                # Call the function for the source dipole located at region 1
                alpha0, beta0, gamma1, delta1 = mc.mie_single1(self.ni, self.k0br, self.expansion_order, self._log)

            return alpha0, beta0, gamma1, delta1
        elif self.boundary_condition == 'coreshell':
            # Find the region at which the source dipole is located
            if self.source_dipole.region == 0:
                # Call the function for the source dipole located at region 0
                alpha0, beta0, alpha1, beta1, gamma1, delta1, gamma2, delta2 = \
                    mc.mie_coreshell0(self.ni, self.k0br, self.expansion_order, self._log)
            elif self.source_dipole.region == 1:
                # Call the function for the source dipole located at region 1
                alpha0, beta0, alpha1, beta1, gamma1, delta1, gamma2, delta2 = \
                    mc.mie_coreshell1(self.ni, self.k0br, self.expansion_order, self._log)
            else:
                # Call the function for the source dipole located at region 2
                alpha0, beta0, alpha1, beta1, gamma1, delta1, gamma2, delta2 = \
                    mc.mie_coreshell2(self.ni, self.k0br, self.expansion_order, self._log)
            
            return alpha0, beta0, alpha1, beta1, gamma1, delta1, gamma2, delta2
        
    def _imag_dyadic_greens_function_single_0(self):
        
        ImGF_0 = 6 * np.pi * np.eyes(3) / self.k0
        return ImGF_0


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
    #print(ind)