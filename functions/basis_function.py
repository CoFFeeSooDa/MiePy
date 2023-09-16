import numpy as np

def _envj(z: np.complex128, n: np.int16, log_message=None) -> float:
    """_envj

    Args:
        n (int): order of spherical Bessel functions
        z (float): argument of spherical Bessel functions
        log_message (object): object of logging standard module (for the use of MiePy logging only)
        

    Returns:
        _envj (float): threshold for functions _msta1 and _msta2

    Reference:
        Computation of Special Functions by Shanjie Zhang, Jianming Jin (1996)
    """
    n = np.max([1, np.abs(n)])
    return .5 * np.log10(6.28 * n) - n * np.log10(1.36 * z/n)

def _msta1(z: np.complex128, mp: np.int16, log_message=None) -> np.int16:
    """_msta1

    Args:
        z (float): argument of spherical Bessel functions
        mp (int): inital order of spherical Bessel functions (Please see also the reference)
        log_message (object): object of logging standard module (for the use of MiePy logging only)
        

    Returns:
        nn (int): the order for recursive computations

    Reference:
        Computation of Special Functions by Shanjie Zhang, Jianming Jin (1996)
    """
    a0 = np.abs(z)
    n0 = np.fix(1.1 * a0) + 1
    f0 = _envj(a0, n0) - mp
    n1 = n0 + 5
    f1 = _envj(a0, n1) - mp
    
    for ii in range(20):
        nn = n1 - (n1 - n0) / (1.0 - f0/f1)
        nn = np.fix(nn)
        f = _envj(a0, nn) - mp
        if np.abs(nn - n1) < 1:
            break

        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
    
    return nn.astype(np.int16)

def _msta2(z: np.complex128, n: np.int16, mp: np.int16, log_message=None) -> np.int16:
    """_msta2

    Args:
        z (float): argument of spherical Bessel functions
        n (float): order of spherical Bessel functions need to be calculated
        mp (int): inital order of spherical Bessel functions (Please see also the reference)
        log_message (object): object of logging standard module (for the use of MiePy logging only)
        

    Returns:
        nn (int): the order for recursive computations

    Reference:
        Computation of Special Functions by Shanjie Zhang, Jianming Jin (1996)
    """
    a0 = np.abs(z)
    hmp = .5 * mp
    ejn = _envj(a0, n)
    if ejn <= hmp:
        obj = mp
        n0 = np.fix(1.1 * a0)
    else:
        obj = hmp + ejn
        n0 = n
    f0 = _envj(a0, n0) - obj
    n1 = n0 + 5
    f1 = _envj(a0, n1) - obj
    for ii in range(20):
        nn = n1-(n1-n0)/(1.0-f0/f1)
        nn = np.fix(nn)
        f = _envj(a0, nn) - obj
        if np.abs(nn-n1) < 1:
           break
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
    return (nn + 10).astype(np.int16)

def spherical_bessel_function(z: np.complex128, n: np.int16, log_message=None) -> np.ndarray:
    """spherical_bessel_function

    Args:
        z (np.complex128): complex argument of spherical Bessel functions
        n (int): order of spherical Bessel functions
        log_message (object): object of logging standard module (for the use of MiePy logging only)

    Returns:
        j_complex (ndarray[np.complex128] ((n+1)x1)): complex values of spherical Bessel functions
                                                      up to n-th order
        n_complex (ndarray[np.complex128] ((n+1)x1)): complex values of spherical Neumann functions
                                                      up to n-th order

    Reference:
        Computation of Special Functions by Shanjie Zhang, Jianming Jin (1996)
    """
    
    a0 = np.abs(z)
    nm = n
    # Cases for arguments approach to zero
    if a0 < 1e-60:
        j_complex = np.zeros([n+1,1], dtype=np.complex128)
        y_complex = np.ones([n+1,1], dtype=np.complex128) * (-1e300)
        if log_message is not None:
            log_message.warning('Reduced accuracy of spherical Neumann functions!')
        y_complex[0] = 1e0
        return j_complex, y_complex
    
    # Initialize the array of spherical Bessel functions
    j_complex = np.zeros([n+1,1], dtype=np.complex128)
    # Zeroth-order spherical Bessel function
    j0 = np.sin(z)/z
    # First-order spherical Bessel function
    j1 = (j_complex[0] - np.cos(z)) / z
    # Compute spherical Bessel functions of order >= 2
    if n >= 2:
        # Set the starting order for the backward recursive computations
        m = _msta1(a0,200)
        if m < n:
            nm = m
        else:
            m = _msta2(a0,n,15)
        # Define the initial condition
        cf0 = 0.0
        cf1 = -99.0
        # Recursive computations
        for kk in range(m, -1, -1):
            cf = (2.0*kk + 3.0) * cf1/z - cf0
            if kk <= nm :
                j_complex[kk] = cf

            cf0=cf1
            cf1=cf
        if np.abs(j0) > np.abs(j1):
            cs = j0/cf
        else:
            cs = j1/cf0
        # Fill in the results of spherical Bessel functions
        for kk in range(np.min([nm,n]) + 1):
            j_complex[kk] = cs * j_complex[kk]
    
    # Initialize the array of spherical Neumann functions
    y_complex = np.zeros([n + 1,1]).astype(np.complex128)
    # Zeroth-order spherical Neumann function
    y_complex[0] = -np.cos(z) / z
    # First-order spherical Neumann function
    y_complex[1] = (y_complex[0] - np.sin(z)) / z
    # Fill in the results of spherical Bessel functions
    for kk in range(2, min(nm, n) + 1):
        if abs(j_complex[kk-1]) >= abs(j_complex[kk-2]):
            y_complex[kk] = (j_complex[kk] * y_complex[kk - 1] - 1.0 / z**2) / j_complex[kk - 1]
        else:
            y_complex[kk] = (j_complex[kk] * y_complex[kk - 2] - (2.0 * kk - 1.0) / z**3) / j_complex[kk - 2]


    return j_complex, y_complex

def riccati_bessel_function_S(z: np.complex128, n: np.int16, log_message=None) -> np.ndarray:
    """riccati_bessel_function_S

    Args:
        z (np.complex128): complex argument of Riccati-Bessel S functions
        n (int): order of Riccati-Bessel S functions
        log_message (object): object of logging standard module (for the use of MiePy logging only)

    Returns:
        S_complex (ndarray[np.complex128] ((n+1)x1))           : complex values of Riccati-Bessel S functions
                                                                 (psi functions) up to n-th order 
        S_derivative_complex (ndarray[np.complex128] ((n+1)x1)): complex values of the derivatives of Riccati-
                                                                 Bessel S functions (psi functions) up to n-th
                                                                 order
        
        ** Notation adopted from Wikipedia: Bessel function

    Reference:
        Computation of Special Functions by Shanjie Zhang, Jianming Jin (1996)
    """
    S_complex = np.zeros([n+1,1], dtype=np.complex128)
    S_derivative_complex = np.zeros([n+1,1], dtype=np.complex128)

    nm = n
    if np.abs(z) < 1e-60:
        S_derivative_complex[0] = 1.0  # zeroth order
        return S_complex, S_derivative_complex
    else:
        S_complex0 =  np.sin(z)
        S_complex1 =  S_complex0 / z - np.cos(z)
        
        S_complex[0] = S_complex0
        S_complex[1] = S_complex1

        if n >= 2:
            M = _msta1(z,200)
            if M < n:
                nm = M
            else:
                M = _msta2(z,n,15)
            tmp0 = 0
            tmp1 = 1.0e-100
            for jj in range(M,-1,-1):
                tmp2 = (2.0*jj +3.0) * tmp1/z - tmp0
                if jj <= nm:
                    S_complex[jj] = tmp2
                tmp0 = tmp1
                tmp1 = tmp2
        
            if np.abs(S_complex0) > np.abs(S_complex1):
                CS = S_complex0/tmp1
            else:
                CS = S_complex1/tmp0

            for jj in range(nm+1):
                S_complex[jj] = CS * S_complex[jj]
        
        S_derivative_complex[0] = np.cos(z)
        S_derivative_complex[1] = -S_complex1/z + S_complex0

        for jj in range(2,nm+1):
            S_derivative_complex[jj] = -jj*S_complex[jj]/z + S_complex[jj-1]
        
        return S_complex, S_derivative_complex
    
def riccati_bessel_function_C(z: np.complex128, n: np.int16, log_message=None) -> np.ndarray:
    """riccati_bessel_function_C

    Args:
        z (np.complex128): complex argument of Riccati-Bessel C functions 
        n (int): order of Riccati-Bessel C functions
        log_message (object): object of logging standard module (for the use of MiePy logging only)

    Returns:
        C_complex (ndarray[np.complex128] ((n+1)x1))           : complex values of Riccati-Bessel C functions
                                                                 up to n-th order
        C_derivative_complex (ndarray[np.complex128] ((n+1)x1)): complex values of Riccati-Bessel C functions
                                                                 Riccati-Bessel C functions up to n-th order
        
        ** Notation adopted from Wikipedia: Bessel function

    Reference:
        Computation of Special Functions by Shanjie Zhang, Jianming Jin (1996)
    """
    C_complex = np.zeros([n+1,1], dtype=np.complex128)
    C_derivative_complex = np.zeros([n+1,1], dtype=np.complex128)

    nm = n
    if np.abs(z) < 1e-60:
        if log_message is not None:
            log_message.warning('Reduced accuracy of Riccati-Bessel C functions!')
        C_complex = np.full([n+1,1],-1.0e300, dtype=np.complex128)
        C_derivative_complex = np.full([n+1,1],1.0e300, dtype=np.complex128)
        C_complex[0] = -1.0
        C_derivative_complex[0] = 0.0

        return C_complex, C_derivative_complex
    else:
        C_complex0 = -np.cos(z)
        C_complex1 =  C_complex0 / z - np.sin(z)

        C_complex[0] = C_complex0
        C_complex[1] = C_complex1

        tmp0 = C_complex0
        tmp1 = C_complex1

        for jj in range(2,n+1):
            tmp2 = (2.0 * jj - 1.0) * tmp1/z - tmp0
            if np.abs(tmp2) > 1e300:
                max_ind = jj
                continue
            C_complex[jj] = tmp2
            tmp0 = tmp1
            tmp1 = tmp2
            max_ind = jj

        # Calculate and fill in data in C_derivative_complex
        C_derivative_complex[0] = np.sin(z)
        C_derivative_complex[1] = -C_complex1/z + C_complex0
        for jj in range(2,max_ind+1):
            C_derivative_complex[jj] = -jj*C_complex[jj]/z + C_complex[jj-1]
        
        return C_complex, C_derivative_complex

def logarithmic_derivative_riccati_bessel_function_S(z: np.complex128, n: np.int16, log_message=None) -> np.ndarray:
    """logarithmic_derivative_riccati_bessel_function_S

    Args:
        z (np.complex128): complex argument of logarithmic derivatives of Riccati-Bessel S functions
        n (int): order of logarithmic derivatives of Riccati-Bessel S functions
        log_message (object): object of logging standard module (for the use of MiePy logging only)

    Returns:
        DS_complex (ndarray[np.complex128] ((n+1)x1))       : complex values of logarithmic derivatives of 
                                                              Riccati-Bessel S functions up to n-th order
        
        ** Notation adopted from Wikipedia: Bessel function

    Reference:
        J. Mod. Opt., 63, 2348-2355 (2016)
    """
    # From experience
    nex = (n + np.floor(np.abs(1.0478 * z + 18.692))).astype(np.int16)
    DS_complex = np.zeros([nex+1,1],dtype=np.complex128)
    for jj in range(nex,1,-1):
        DS_complex[jj-1] = jj/z - 1/(jj/z + DS_complex[jj])

    DS_complex[1] = (z**2 * np.tan(z) + z - np.tan(z)) / (-z**2 + z*np.tan(z))
    DS_complex[0] = 1 / np.tan(z)
    
    return DS_complex[:n+1]

def logarithmic_derivative_riccati_bessel_function_xi(z: np.complex128, n: np.int16, log_message=None) -> np.ndarray:
    """logarithmic_derivative_riccati_bessel_function_xi

    Args:
        z (np.complex128): complex argument of logarithmic derivatives of Riccati-Bessel xi functions
        n (int): order of logarithmic derivatives of Riccati-Bessel xi functions
        log_message (object): object of logging standard module (for the use of MiePy logging only)

    Returns:
        Dxi_complex (ndarray[np.complex128] ((n+1)x1))       : complex values of logarithmic derivatives of 
                                                               Riccati-Bessel xi functions up to n-th order
        
        ** Notation adopted from Wikipedia: Bessel function

    Reference:
        J. Mod. Opt., 63, 2348-2355 (2016)
    """
    Dxi_complex = np.zeros([n+1,1],dtype=np.complex128)
    Dxi_complex[0] = 1j
    Dxi_complex[1] = (1j * z**2 - z - 1j) / (z**2 + 1j*z)
    for jj in range(2,n+1):
        Dxi_complex[jj] = -jj/z + 1/(jj/z - Dxi_complex[jj-1])
    
    return Dxi_complex
            
        

if __name__ == '__main__':
    #S_complex, S_derivative_complex = riccati_bessel_function_S(4+5j,5)
    #C_complex, C_derivative_complex = riccati_bessel_function_C(4+5j,5)
    #print(f'{S_complex = }')
    #print(f'{S_derivative_complex = }')
    #print(f'{C_complex = }')
    #print(f'{C_derivative_complex = }')
    #DS_complex = logarithmic_derivative_riccati_bessel_function_S(4+5j,5)
    Dxi_complex = logarithmic_derivative_riccati_bessel_function_xi(4+5j,5)
    print(f'{Dxi_complex=}')
    