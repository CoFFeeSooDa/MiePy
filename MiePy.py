# Standard imports:
import logging
from pathlib import Path
import json

# External imports:
import numpy as np

# My functions
import functions.coordinate_transformation as ct
import functions.basis_function as bf


# Main class to compute properties of Mie scattering 
class MiePy(object):
    def __init__(self,input_path='./input_json/Demo_AngleMode_CF.json',debug_folder=None,output_debug_file=False):
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
        
        # load input file (.json)
        self.intput_path = input_path
        # Other variables that need to be initialized...

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

    # Read json files
    def read_json(self):
        with open(self.intput_path) as f:
            data = json.load(f)
            self.settings = data['Settings']['ModeName']
            print(f'{self.settings = }')
    
    
    #test block
    def test(self):
        self._log.info('It is a info text')
        self._log.debug('It is a debug text')
        self.DPos = Position([2,2,0],type='cartesian',log_message=self._log)
        print(f'{self.DPos.cartesian=}')
        print(f'{self.DPos.spherical=}')
        ct.spherical_to_spherical([3,0,1.57],1,0,self._log)
        bf.normTauPiP(1,11,'normal',self._log)
        

# A class to calculate the position vectors in Cartesian/spherical coordinates
# type = 'cartesian' or 'spherical'
class Position(object):
    def __init__(self,position_vector: list,type='cartesian',log_message=None) -> None:
        if type == 'cartesian':
            self.cartesian = np.array(position_vector,dtype=np.float64)
            # From Cartesian coordinates to spherical coordinates
            self.spherical = ct.cartesian_to_spherical(self.cartesian,log_message)
        elif type == 'spherical':
            self.spherical = np.array(position_vector)
            # From spherical coordinates to Cartesian coordinates
            self.cartesian = ct.spherical_to_cartesian(self.spherical,log_message)


#A class to calculate the orientation of a dipole in Cartesian/spherical coordinates
# type = 'cartesian' or 'spherical'
class Orientation(object):
    def __init__(self,orientation_vector: list,type='cartesian') -> None:
        if type == 'cartesian':
            pass
        elif type == 'spherical':
            pass

if __name__ == '__main__':
    #MP = MiePy('./input_json/Demo_AngleMode_CF.json')
    MP = MiePy(output_debug_file=False)
    MP.test()
    