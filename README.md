# MiePy

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

It is a collaboratory repository for developing a Python version [MieScatteringForDipoleSource](https://github.com/CoFFeeSooDa/MieScatteringForDipoleSource))

Plans:
Construct a class including attributes of:
1. Green's functions (single point)
2. Green's functions (two points)
3. Electric fields
4. Magnetic fields (not implemented in the MATLAB version)
5. Purcell factor
6.  ... feel free to add desired features (wish pool)

## Log

2023/09/16: Ming-Wei complete the radial basis functions. Now attempts to establish a class for radial functions.

### 2023/09/16: IMPORTANT NOTE - BY Ming-Wei

I have finished the issue of logging problem. To keep the consistency, please follow the rules below:
1. The functions (.py files) should be saved in the directory of "functions".
2. Functions are categorized by their functionality, e.g.,
	
   (1) coordinate_transformation: C2S, S2C, S2S, VecTrans, etc.
   
   (2) basis_function: envj, msta1, msta2, sbesselc, rcbesselc, SphBessel, Dlog, Wigner_d, NormTauPiP, etc.
   
   (3) green_function: SingleGR0, SingleGR1, TwoGR0, TwoGR1, etc.
   
   (4) keep updating ...
4. Arguments of functions should include ```log_message=None``` for the purpose of logging.
   Take ```coordinate_transformation.spherical_to_spherical``` for example,
```python
def spherical_to_spherical(spherical_2: np.ndarray, spherical_1_theta: float, spherical_1_phi: float, log_message=None) -> np.ndarray:
# main tasks of this function .....
if log_message is not None:
        log_message.info(str_green('Function spherical_to_spherical currently only supports z-directional shift.'))
#still computations ....
return spherical_1
```

PS. I also created a module ```text_color``` to change the color of strings in the console. Here is an example:
```python
print(text_color.str_green('Function spherical_to_spherical currently only supports z-directional shift.'))
```

---

2023/09/14: Hung-Sheng Tsai constructed the code to transform between cartesian coordinate and spherical coordinate. Please use the following code to use it
```python
import Fn_cartesian_to_spherical as Fn_C2S
spherical_coord = Fn_C2S.cartesian_to_spherical(**kwargs)
```
Note: Some common used constants are included in constant.py. To see or modify the .py file, please use the following command
```
git checkout basis_and_coord
```
To use the constant variables in your python script, please use the following code
```python
from constant import *
```

2023/09/11: Hung-Sheng Tsai constructed the code to transform between spherical coordinates. Please use the following code to use it
```python
import Fn_spherical_to_spherical as Fn_S2S
spherical_2 = Fn_S2S.spherical_to_spherical(**kwargs)
```

2023/09/07: Ming-Wei initialized a debug tool in the class MiePy. If one needs to output debugging logs, please use the following normative command
```python
self._log.info('It is a info text')    # for the level of info
self._log.debug('It is a debug text')  # for the level of debugging
```




## License

[MIT](LICENSE) © Richard Littauer
