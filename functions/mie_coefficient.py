import numpy as np
import functions.basis_function as bf

def mie_single0(nr:np.ndarray, k0br:np.ndarray, n_max:np.int16, log_message=None):
    """mie_single0

    Args:
        nr (ndarray[float], 1x2): complex refractive index for each region
        k0br (ndarray[float], 1x1): k0 * boundary radius
        n_max (int): maximum expansion order
        log_message (object): object of logging standard module (for the use of MiePy logging only)
        
    Returns:
        Tuple: Mie coefficients for a source dipole located at Region 0.
            alpha (ndarray[float], n x 1): Mie alpha coefficient
            beta (ndarray[float], n x 1): Mie beta coefficient
            gamma (ndarray[float], n x 1): Mie gamma coefficient
            delta (ndarray[float], n x 1): Mie delta coefficient
    """
    if ((nr.size != 2) or (k0br.size != 1)) and log_message:
        log_message.error('illegal inputs of "nr" or "k0br" to mie_single')
    
    n0, n1 = nr
    n0kr1, n1kr1 = k0br * nr
    # Radial functions
    n0psi1  = bf.riccati_bessel_function_S(n0kr1, n_max, log_message)[0][1:]
    n1psi1  = bf.riccati_bessel_function_S(n1kr1, n_max, log_message)[0][1:]
    n0Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n0kr1, n_max, log_message)[1:]
    n1Dpsi1 = bf.logarithmic_derivative_riccati_bessel_function_S(n1kr1, n_max, log_message)[1:]
    n0xi1  = bf.riccati_bessel_function_xi(n0kr1, n_max, log_message)[0][1:]
    n0Dxi1  = bf.logarithmic_derivative_riccati_bessel_function_xi(n0kr1, n_max, log_message)[1:]

    # coefficients
    alpha = -(n1 * n0Dpsi1 - n0 * n1Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n0xi1
    beta  = -(n0 * n0Dpsi1 - n1 * n1Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n0xi1
    gamma =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n0 * n0Dxi1 - n1 * n1Dpsi1) * n0psi1 / n1psi1
    delta =  (n1 * n0Dxi1  - n1 * n0Dpsi1) / (n1 * n0Dxi1 - n0 * n1Dpsi1) * n0psi1 / n1psi1

    return alpha, beta, gamma, delta




if __name__ == '__main__':
    alpha, beta, gamma, delta = mie_single0(np.array([1,3+0.4j]), np.array([1.1]), 5)
    print(f'{alpha=}')
    print(f'{beta=}')
    print(f'{gamma=}')
    print(f'{delta=}')
    
