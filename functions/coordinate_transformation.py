import numpy as np

# import local functions
from functions.text_color import str_green



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

def spherical_to_spherical(spherical_2: np.ndarray, 
                           spherical_1_theta: float, 
                           spherical_1_phi: float,
                           log_message=None) -> np.ndarray:
    """spherical_to_spherical

    Args:
        spherical_2 (ndarray[float] (3x1)): secondary spherical coordinate
        spherical_1_theta (float): theta in the S1 coordinate
        spherical_1_phi (float): phi in the S1 coordinate
        log_message (object): object of logging standard module (for the use of MiePy logging only)

    Returns:
        spherical_1 (ndarray[float] (3x1)): primary spherical coordinate
    """
    if log_message is not None:
        log_message.info(str_green('Function spherical_to_spherical currently only supports z-directional shift.'))
    cos_theta = np.cos(spherical_1_theta)
    sin_theta = np.sin(spherical_1_theta)
    cos_phi = np.cos(spherical_1_phi)
    # coordinate transformation matrix: Eq.(95)
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta, cos_theta,  0],
                  [0,         0,          cos_phi]], dtype=np.float64)
    
    spherical_1 = R @ spherical_2

    return spherical_1