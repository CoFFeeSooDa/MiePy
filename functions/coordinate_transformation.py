import sys
import numpy as np

# import local functions
from functions.text_color import str_green

global first_time_flag 
first_time_flag = True

def cartesian_to_spherical(cartesian_coord: np.ndarray, log_message=None) -> np.ndarray:
    """cartesian_to_spherical

    Args:
        cartesian_coord (ndarray[float] (3x1)): Cartesian coordinate components (x, y, z)
        log_message (object): object of logging standard module (for the use of MiePy logging only)

    Returns:
        spherical_coord (ndarray[float] (3x1)): spherical coordinate components (r, theta, phi)
    """
    # Note:
    #   x = cartesian_coord[0]
    #   y = cartesian_coord[1]
    #   z = cartesian_coord[2]
    
    # Radial distance
    r = np.linalg.norm(cartesian_coord)
    # Polar Angle: theta
    theta = 0 if (r == 0) else np.arccos(cartesian_coord[2] / r) # Assigning polar angle = 0 when r = 0
    # Azimuthal angle: phi
    phi = 0 if ((cartesian_coord[0] == 0) and (cartesian_coord[1] == 0)) \
                else np.arctan2(cartesian_coord[1], cartesian_coord[0]) # Assigning azimuthal angle when x = 0 and y = 0
    
    spherical_coord = np.array([r, theta, phi], dtype=np.float64)

    return spherical_coord

def spherical_to_cartesian(spherical_coord: np.ndarray, log_message=None) -> np.ndarray:
    """cartesian_to_spherical

    Args:
        spherical_coord (ndarray[float] (3x1)): spherical coordinate components (r, theta, phi)
        log_message (object): object of logging standard module (for the use of MiePy logging only)

    Returns:
        cartesian_coord (ndarray[float] (3x1)): Cartesian coordinate components (x, y, z)
    """
    x = spherical_coord[0] * np.sin(spherical_coord[1]) * np.cos(spherical_coord[2])
    y = spherical_coord[0] * np.sin(spherical_coord[1]) * np.sin(spherical_coord[2])
    z = spherical_coord[0] * np.cos(spherical_coord[1])

    cartesian_coord = np.array([x, y, z], dtype=np.float64)

    return cartesian_coord

def vector_spherical_to_spherical(spherical_2: np.ndarray, 
                                  spherical_1_theta: np.float64, 
                                  spherical_1_phi: np.float64,
                                  log_message=None) -> np.ndarray:
    """vector_spherical_to_spherical

    Args:
        spherical_2 (ndarray[np.float64] (3x1)): Secondary spherical coordinate
        spherical_1_theta (np.float64): theta in the S1 coordinate
        spherical_1_phi (np.float64): phi in the S1 coordinate
        log_message (object): Object of logging standard module (for the use of MiePy logging only)

    Returns:
        spherical_1 (ndarray[np.float64] (3x1)): Primary spherical coordinate
    """

    '''
    if log_message is not None and first_time_flag is True:
        log_message.info(str_green('Note: Function spherical_to_spherical currently only supports z-direction shifting.'))
    '''

    cos_theta = np.cos(spherical_1_theta)
    sin_theta = np.sin(spherical_1_theta)
    cos_phi = np.cos(spherical_1_phi)
    # coordinate transformation matrix: Eq.(95)
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta, cos_theta,  0],
                  [0,         0,          cos_phi]], dtype=np.float64)
    
    spherical_1 = R @ spherical_2

    return spherical_1

def vector_transformation(vector: np.ndarray, solid_angle : np.ndarray, type: str, log_message=None) -> np.ndarray:
    """vector_transformation

    Args:
        vector (ndarray[np.float64] (3x1)): Vector expressed in Cartesian coordinate
        solid_angle(ndarray[np.float64] (2x1)): Solid angle of the vector position (theta and phi) in radian
        type (str): Either 'c2s' (Cartesian to spherical) or 's2c' (spherical to Cartesian)
        log_message (object): Object of logging standard module (for the use of MiePy logging only)

    Returns:
        spherical_1 (ndarray[float] (3x1)): Primary spherical coordinate
    """
    theta, phi = solid_angle
    # Transformation matrix
    T = np.array([[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
                  [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
                  [np.cos(theta)            ,-np.sin(theta)            ,  0          ]], dtype=np.float64)
    
    if type == 'c2s':
        return T.T @ vector
    elif type == 's2c':
        return T @ vector
    else:
        if log_message:
            log_message.error('illegal type assignment in vector_tranformation.')
        sys.exit('Error occurs in vector_transformation.')

if __name__ == '__main__':
        print(vector_transformation(np.array([0,0,1]),np.array([0,0])))