import numpy as np
import functions.coordinate_transformation as ct

# A class to calculate the position vectors in Cartesian/spherical coordinates
# type = 'cartesian' or 'spherical'
class Position(object):
    def __init__(self,position_vector: object,log_message=None) -> None:
        Cart = getattr(position_vector, 'Cart', None)
        Sph = getattr(position_vector, 'Sph', None)
        if (Cart is not None) and (Sph == None):
            self.cartesian = np.array(Cart,dtype=np.float64)
            # From Cartesian coordinates to spherical coordinates
            self.spherical = ct.cartesian_to_spherical(self.cartesian,log_message)
        elif (Cart == None) and (Sph is not None):
            self.spherical = np.array(Sph)
            # From spherical coordinates to Cartesian coordinates
            self.cartesian = ct.spherical_to_cartesian(self.spherical,log_message)
        else:
            log_message.error('Incorrect attribute from position vector.')
            log_message.error('Attribute must be "Cart" or "Sph"')

# A class to calculate the orientation of a dipole in Cartesian/spherical coordinates
# type = 'cartesian' or 'spherical'
class Orientation(object):
    def __init__(self,orientation_vector: list,type='cartesian') -> None:
        if type == 'cartesian':
            pass
        elif type == 'spherical':
            pass

# Convert dictionaries to an object
class Dictionary_To_Object(object):
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dictionary_To_Object(value))
            else:
                setattr(self, key, value)


def position_vector(Cartesian:list, Spherical:list, log_message=None):
    if (Cartesian is not None) and (Spherical == None):
        Cart = np.array(Cartesian, dtype=np.float64)
        # From Cartesian coordinates to spherical coordinates
        Sph = ct.cartesian_to_spherical(Cartesian,log_message)

    elif (Cartesian == None) and (Spherical is not None):
        Sph = np.array(Spherical)
        # From spherical coordinates to Cartesian coordinates
        Cart = ct.spherical_to_cartesian(Spherical,log_message)
    else:
        log_message.error('Incorrect attribute from position vector.')
        log_message.error('Attribute must be "Cart" or "Sph"')       
    
    return Cart, Sph