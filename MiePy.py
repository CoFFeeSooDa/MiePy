# Standard imports:
import logging
from pathlib import Path

# External imports:
import numpy as np


# Class
class MiePy():
    def __init__(self,debug_folder=None,output_debug_file=False):
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


    #output example
    def test(self):
        self._log.info('It is a info text')
        self._log.debug('It is a debug text')


if __name__ == '__main__':
    MP = MiePy(debug_folder=None,output_debug_file=True)
    #MP = MiePy()
    MP.test()
    