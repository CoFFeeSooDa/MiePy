# Special Thanks to Yi-Ting Chuang for the code of interpolation
# Modules
import csv
import numba
from numba import cfunc, carray
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

#Self-Written Scripts

#### Constants
InvM2J = 1.986445857e-25 #Unit: J
c = 2.99792458e8  # m/s
hbar = 1.0545718e-34  # m**2 kg / s
Q = 1.602176634e-19
epsilon0 = 8.8541878128e-12

def jit_integrand_function(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)
    
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx, n)
        return jitted_function(values)
    return LowLevelCallable(wrapped.ctypes)


@jit_integrand_function
def epsi_Integrand(args_arr):
    w = args_arr[0]
    omega_p = args_arr[1]
    gamma = args_arr[2]
    sigma = args_arr[3]
    omega2 = args_arr[4]
    ReIm_index = args_arr[5]

    integrand = ((omega_p * w)**0.5 * np.exp(-w / sigma)) / (w**2 - omega2**2 - 1j * gamma * omega2)

    if ReIm_index == np.float_(0):
        integrand_result = np.float_(np.real(integrand))
    elif ReIm_index == np.float_(1):
        integrand_result = np.float_(np.imag(integrand))
    else:
        print("ReIm_index error in epsi!!")
    
    return integrand_result

## Fit dielectric function by Opt. Mem. Neural Networks 23 (2014)
def Dielectric_OMNN(N, omega):
    omega_p = 9.042
    gamma_d = 0.022
    delta = 4.050
    gamma = 0.260
    sigma = 9.935
    f = 2.994

    eps = np.zeros(N, dtype=np.complex128);
    for jj in range(N):
        omega2 = omega[jj];

        argsR = (omega_p, gamma, sigma, omega2, np.float_(0))
        argsI = (omega_p, gamma, sigma, omega2, np.float_(1))

        # integ = lambda w: ((omega_p * w)**0.5 * np.exp(-w / sigma)) / (w**2 - omega2**2 - 1j * gamma * omega2)
        I1 = scipy.integrate.quad(epsi_Integrand, delta, 100, args=argsR, epsabs=1e-10, epsrel=1e-10, limit=int(1e7))[0] \
            + scipy.integrate.quad(epsi_Integrand, 100, np.inf, args=argsR, epsabs=1e-10, epsrel=1e-10, limit=int(1e7))[0] \
            + 1j * (scipy.integrate.quad(epsi_Integrand, delta, 100, args=argsI, epsabs=1e-10, epsrel=1e-10, limit=int(1e7))[0] \
            + scipy.integrate.quad(epsi_Integrand, 100, np.inf, args=argsI, epsabs=1e-10, epsrel=1e-10, limit=int(1e7))[0])
        eps[jj] = 1 - omega_p**2 / (omega2 * (omega2 + 1j * gamma_d)) + f * I1;

    return eps

def read_dielectrics_csv(csv_file_path,csv_data='refractive_index'):
    # Open the CSV file
    with open(csv_file_path, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file, delimiter=',')
        # Read each row
        data = []
        for _ in csv_reader:
            data.append(_)
        # Casting the list to a numpy array
        data = np.array(data, dtype=np.float64)
        # Get wavelength
        wavelength = data[:,0]
        
        if csv_data == 'refractive_index':
            # Get refractive indices
            refractive_index = data[:,1].astype(np.complex128) + 1j * data[:,2].astype(np.complex128)    
        elif csv_data == 'dielectric_constant':
            # Get dielectric constants
            dielectrics = data[:,1].astype(np.complex128) + 1j * data[:,2].astype(np.complex128)
            # Convert to complex refractive indices
            # refractive_index = np.sqrt(dielectrics)

    return wavelength, dielectrics
       
if __name__ == "__main__":
    csv_filepath = "./dielectric_data/Ag_JPCL.csv"
    wavelength, epsi_csv =\
        read_dielectrics_csv(csv_filepath,
                             csv_data='dielectric_constant')
    pts = 10000
    wavelength_YiTing = np.linspace(300, 700, pts + 1) * 1e-9
    omega = np.reciprocal(wavelength_YiTing) * InvM2J / Q
    N = len(omega)
    epsi_YiTing = Dielectric_OMNN(N, omega)
    result = np.zeros((N, 3), dtype=np.float64)
    result[:, 0] = wavelength_YiTing
    result[:, 1] = np.real(epsi_YiTing)
    result[:, 2] = np.imag(epsi_YiTing)
    np.savetxt("./dielectric_data/Ag_OMNN.csv", result, delimiter=",")
    # print(wavelength)
    
    wavelength *= 1e9
    wavelength_YiTing *= 1e9
    plt.plot(wavelength_YiTing, np.real(epsi_YiTing), label="OMNN Model: Re$[\\varepsilon_{r}(\\omega)]$")
    plt.plot(wavelength, np.real(epsi_csv), '--', label="Interpolation by Ming Wei: Re$[\\varepsilon_{r}(\\omega)]$")
    plt.plot(wavelength_YiTing, np.imag(epsi_YiTing), label="OMNN Model: Im$[\\varepsilon_{r}(\\omega)]$")
    plt.plot(wavelength, np.imag(epsi_csv), '--', label="Interpolation by Ming Wei: Im$[\\varepsilon_{r}(\\omega)]$")
    plt.rcParams['savefig.dpi'] = 1000 #圖片像素
    plt.rcParams['figure.dpi'] = 1000 #分辨率
    plt.title("Dielectric Constant Difference")
    plt.xlabel("Wavelength ($\\mathrm{nm}$)")
    plt.ylabel("Dielectric Constant Difference $\\varepsilon_{r}(\\omega)$")
    plt.legend()
    plt.show()
    pass
