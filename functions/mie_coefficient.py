import sys
import numpy as np
import functions.basis_function as bf

def mie_single0(nr:np.ndarray, k0br:np.ndarray, n_max:np.int16, log_message=None):
    """mie_single0 -- source dipole is located at region 0 

    Args:
        nr (ndarray[np.complex128], 1x2): Complex refractive indices for each region
        k0br (ndarray[float], 1x1): k0 * boundary radius
        n_max (np.int16): Maximum expansion order
        log_message (object): Object of logging standard module (for the use of MiePy logging only)
        
    Returns:
        Tuple: Mie coefficients for a source dipole located at Region 0.
            alpha0 (ndarray[np.complex128], n x 1): Mie alpha0 coefficient
            beta0  (ndarray[np.complex128], n x 1): Mie beta0 coefficient
            gamma1 (ndarray[np.complex128], n x 1): Mie gamma1 coefficient
            delta1 (ndarray[np.complex128], n x 1): Mie delta1 coefficient
    """
    if ((nr.size != 2) or (k0br.size != 1)):
        if log_message:
            log_message.error('illegal inputs of "nr" or "k0br" to mie_single0')
        sys.exit('Error occurs in mie_single0.')
    
    n0, n1 = nr
    n0kr1, n1kr1 = k0br * nr
    # Radial functions
    n0psi1  = bf.riccati_bessel_function_S(n0kr1, n_max, log_message)[0][1:]
    n1psi1  = bf.riccati_bessel_function_S(n1kr1, n_max, log_message)[0][1:]
    n0Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n0kr1, n_max, log_message)[1:]
    n1Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n1kr1, n_max, log_message)[1:]
    n0xi1   = bf.riccati_bessel_function_xi(n0kr1, n_max, log_message)[0][1:]
    n0Dxi1  = bf.logarithmic_derivative_riccati_bessel_function_xi(n0kr1, n_max, log_message)[1:]

    # coefficients
    alpha0 = -(n1 * n0Dpsi1 - n0 * n1Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n0xi1
    beta0  = -(n0 * n0Dpsi1 - n1 * n1Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n0xi1
    gamma1 =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n1psi1
    delta1 =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n1psi1

    return alpha0, beta0, gamma1, delta1

# TODO
def mie_single1(nr:np.ndarray, k0br:np.ndarray, n_max:np.int16, log_message=None):
    """mie_single1 -- source dipole is located at region 1 

    Args:
        nr (ndarray[np.complex128], 1x2): Complex refractive indices for each region
        k0br (ndarray[np.float64], 1x1): k0 * boundary radius
        n_max (int): Maximum expansion order
        log_message (object): Object of logging standard module (for the use of MiePy logging only)
        
    Returns:
        Tuple: Mie coefficients for a source dipole located at Region 1.
            alpha0 (ndarray[np.complex128], n x 1): Mie alpha coefficient
            beta0  (ndarray[np.complex128], n x 1): Mie beta coefficient
            gamma1 (ndarray[np.complex128], n x 1): Mie gamma coefficient
            delta1 (ndarray[np.complex128], n x 1): Mie delta coefficient
    """
    if ((nr.size != 2) or (k0br.size != 1)):
        if log_message:
            log_message.error('illegal inputs of "nr" or "k0br" to mie_single1')
        sys.exit('Error occurs in mie_single1.')
    
    n0, n1 = nr
    n0kr1, n1kr1 = k0br * nr
    # Radial functions
    '''
    n0psi1  = bf.riccati_bessel_function_S(n0kr1, n_max, log_message)[0][1:]
    n1psi1  = bf.riccati_bessel_function_S(n1kr1, n_max, log_message)[0][1:]
    n0Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n0kr1, n_max, log_message)[1:]
    n1Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n1kr1, n_max, log_message)[1:]
    n0xi1  = bf.riccati_bessel_function_xi(n0kr1, n_max, log_message)[0][1:]
    n0Dxi1  = bf.logarithmic_derivative_riccati_bessel_function_xi(n0kr1, n_max, log_message)[1:]
    '''
    # coefficients
    '''
    alpha = -(n1 * n0Dpsi1 - n0 * n1Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n0xi1
    beta  = -(n0 * n0Dpsi1 - n1 * n1Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n0xi1
    gamma =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n1psi1
    delta =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n1psi1
    '''
    pass
    #return alpha0, beta0, gamma1, delta1

# TODO
def mie_coreshell0(nr:np.ndarray, k0br:np.ndarray, n_max:np.int16, log_message=None):
    """mie_coreshell0 -- source dipole is located at region 0

    Args:
        nr (ndarray[np.complex128], 1x3): Complex refractive indices for each region
        k0br (ndarray[np.float64], 1x2): k0 * boundary radii (descending order, i.e., shell radius then core radius)
        n_max (int): Maximum expansion order
        log_message (object): Object of logging standard module (for the use of MiePy logging only)
        
    Returns:
        Tuple: Mie coefficients for a source dipole located at Region 0.
            alpha0 (ndarray[np.complex128], n x 1): Mie alpha0 coefficient
            beta0  (ndarray[np.complex128], n x 1): Mie beta0 coefficient
            alpha1 (ndarray[np.complex128], n x 1): Mie alpha1 coefficient
            beta1  (ndarray[np.complex128], n x 1): Mie beta1 coefficient
            gamma1 (ndarray[np.complex128], n x 1): Mie gamma1 coefficient
            delta1 (ndarray[np.complex128], n x 1): Mie delta1 coefficient
            gamma2 (ndarray[np.complex128], n x 1): Mie gamma2 coefficient
            delta2 (ndarray[np.complex128], n x 1): Mie delta2 coefficient
    """
    if ((nr.size != 3) or (k0br.size != 2)):
        if log_message:
            log_message.error('illegal inputs of "nr" or "k0br" to mie_coreshell0')
        sys.exit('Error occurs in mie_coreshell0.')
    
    n0, n1, n2 = nr
    n0kr1, n1kr1, n2kr1 = k0br[0] * nr
    n0kr2, n1kr2, n2kr2 = k0br[1] * nr
    # Radial functions
    '''
    n0psi1  = bf.riccati_bessel_function_S(n0kr1, n_max, log_message)[0][1:]
    n1psi1  = bf.riccati_bessel_function_S(n1kr1, n_max, log_message)[0][1:]
    n0Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n0kr1, n_max, log_message)[1:]
    n1Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n1kr1, n_max, log_message)[1:]
    n0xi1  = bf.riccati_bessel_function_xi(n0kr1, n_max, log_message)[0][1:]
    n0Dxi1  = bf.logarithmic_derivative_riccati_bessel_function_xi(n0kr1, n_max, log_message)[1:]
    '''
    # coefficients
    '''
    alpha = -(n1 * n0Dpsi1 - n0 * n1Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n0xi1
    beta  = -(n0 * n0Dpsi1 - n1 * n1Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n0xi1
    gamma =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n1psi1
    delta =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n1psi1
    '''
    pass
    #return alpha0, beta0, alpha1, beta1, gamma1, delta1, gamma2, delta2

# TODO
def mie_coreshell1(nr:np.ndarray, k0br:np.ndarray, n_max:np.int16, log_message=None):
    """mie_coreshell1 -- source dipole is located at region 1

    Args:
        nr (ndarray[np.complex128], 1x3): Complex refractive indices for each region
        k0br (ndarray[np.float64], 1x2): k0 * boundary radii (descending order, i.e., shell radius then core radius)
        n_max (int): Maximum expansion order
        log_message (object): Object of logging standard module (for the use of MiePy logging only)
        
    Returns:
        Tuple: Mie coefficients for a source dipole located at Region 1.
            alpha0 (ndarray[np.complex128], n x 1): Mie alpha0 coefficient
            beta0  (ndarray[np.complex128], n x 1): Mie beta0 coefficient
            alpha1 (ndarray[np.complex128], n x 1): Mie alpha1 coefficient
            beta1  (ndarray[np.complex128], n x 1): Mie beta1 coefficient
            gamma1 (ndarray[np.complex128], n x 1): Mie gamma1 coefficient
            delta1 (ndarray[np.complex128], n x 1): Mie delta1 coefficient
            gamma2 (ndarray[np.complex128], n x 1): Mie gamma2 coefficient
            delta2 (ndarray[np.complex128], n x 1): Mie delta2 coefficient
    """
    if ((nr.size != 3) or (k0br.size != 2)) and log_message:
        log_message.error('illegal inputs of "nr" or "k0br" to mie_coreshell1')
    
    n0, n1, n2 = nr
    n0kr1, n1kr1, n2kr1 = k0br[0] * nr
    n0kr2, n1kr2, n2kr2 = k0br[1] * nr
    # Radial functions
    '''
    n0psi1  = bf.riccati_bessel_function_S(n0kr1, n_max, log_message)[0][1:]
    n1psi1  = bf.riccati_bessel_function_S(n1kr1, n_max, log_message)[0][1:]
    n0Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n0kr1, n_max, log_message)[1:]
    n1Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n1kr1, n_max, log_message)[1:]
    n0xi1  = bf.riccati_bessel_function_xi(n0kr1, n_max, log_message)[0][1:]
    n0Dxi1  = bf.logarithmic_derivative_riccati_bessel_function_xi(n0kr1, n_max, log_message)[1:]
    '''
    # coefficients
    '''
    alpha = -(n1 * n0Dpsi1 - n0 * n1Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n0xi1
    beta  = -(n0 * n0Dpsi1 - n1 * n1Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n0xi1
    gamma =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n1psi1
    delta =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n1psi1
    '''
    pass
    #return alpha0, beta0, alpha1, beta1, gamma1, delta1, gamma2, delta2

# TODO
def mie_coreshell2(nr:np.ndarray, k0br:np.ndarray, n_max:np.int16, log_message=None):
    """mie_coreshell2 -- source dipole is located at region 2

    Args:
        nr (ndarray[np.complex128], 1x3): Complex refractive indices for each region
        k0br (ndarray[np.float64], 1x2): k0 * boundary radii (descending order, i.e., shell radius then core radius)
        n_max (int): Maximum expansion order
        log_message (object): Object of logging standard module (for the use of MiePy logging only)
        
    Returns:
        Tuple: Mie coefficients for a source dipole located at Region 2.
            alpha0 (ndarray[np.complex128], n x 1): Mie alpha0 coefficient
            beta0  (ndarray[np.complex128], n x 1): Mie beta0 coefficient
            alpha1 (ndarray[np.complex128], n x 1): Mie alpha1 coefficient
            beta1  (ndarray[np.complex128], n x 1): Mie beta1 coefficient
            gamma1 (ndarray[np.complex128], n x 1): Mie gamma1 coefficient
            delta1 (ndarray[np.complex128], n x 1): Mie delta1 coefficient
            gamma2 (ndarray[np.complex128], n x 1): Mie gamma2 coefficient
            delta2 (ndarray[np.complex128], n x 1): Mie delta2 coefficient
    """
    if ((nr.size != 3) or (k0br.size != 2)):
        if log_message:
            log_message.error('illegal inputs of "nr" or "k0br" to mie_coreshell2')
        sys.exit('Error occurs in mie_coreshell2.')
    
    n0, n1, n2 = nr
    n0kr1, n1kr1, n2kr1 = k0br[0] * nr
    n0kr2, n1kr2, n2kr2 = k0br[1] * nr
    # Radial functions
    '''
    n0psi1  = bf.riccati_bessel_function_S(n0kr1, n_max, log_message)[0][1:]
    n1psi1  = bf.riccati_bessel_function_S(n1kr1, n_max, log_message)[0][1:]
    n0Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n0kr1, n_max, log_message)[1:]
    n1Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n1kr1, n_max, log_message)[1:]
    n0xi1  = bf.riccati_bessel_function_xi(n0kr1, n_max, log_message)[0][1:]
    n0Dxi1  = bf.logarithmic_derivative_riccati_bessel_function_xi(n0kr1, n_max, log_message)[1:]
    '''
    # coefficients
    '''
    alpha = -(n1 * n0Dpsi1 - n0 * n1Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n0xi1
    beta  = -(n0 * n0Dpsi1 - n1 * n1Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n0xi1
    gamma =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n1psi1
    delta =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n1psi1
    '''
    pass
    #return alpha0, beta0, alpha1, beta1, gamma1, delta1, gamma2, delta2

if __name__ == '__main__':
    alpha, beta, gamma, delta = mie_single0(np.array([1,3+0.4j]), np.array([1.1]), 5)
    print(f'{alpha=}')
    print(f'{beta=}')
    print(f'{gamma=}')
    print(f'{delta=}')
    
