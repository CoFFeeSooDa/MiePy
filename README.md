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

---------------------------------(just a template, please ignore it)------------------------------------------

This repository contains:

1. [MyTest_Tetra3_xyls.py](https://github.com/CoFFeeSooDa/StarTrackerTest/blob/main/MyTest_Tetra3_xyls.py): main program to test the robustness of tetra3.
2. [Centroid_Gen.py](https://github.com/CoFFeeSooDa/StarTrackerTest/blob/main/Centroid_Gen.py): generation of (star) centroids in a specified boresight.
3. [Database_Gen.py](https://github.com/CoFFeeSooDa/StarTrackerTest/blob/main/Database_Gen.py): generation of tetra3 database.
4. [Star_Catalog_to_mat.py](https://github.com/CoFFeeSooDa/StarTrackerTest/blob/main/Star_Catalog_to_mat.py): generation of .mat files from star catalog (Tycho-2) for [Centroid_Gen.py](https://github.com/CoFFeeSooDa/StarTrackerTest/blob/main/Centroid_Gen.py)
5. [My_Tetra3.py](https://github.com/CoFFeeSooDa/StarTrackerTest/blob/main/My_Tetra3.py): modified tetra3 solving engine of the compatibility of Tycho-2 catalog 

## Table of Contents

- [Generate Centroids](#Generate-Centroids)
- [Generate Trimmed Star Table](#Generate-Trimmed-Star-Table)
- (to be continued...)[Install](#install)
- [Usage](#usage)
	- [Generator](#generator)

### Generate Centroids

### Generate Trimmed Star Table
You can create your customrized trimmed star table (.mat files) by using Star_Catalog_to_mat.py. This script reads a star catalog (in ./catalogs) and trim the stars by setting the maximum Vmag. In the current version, we provide the Tycho-2 catalog. Further inoformation about the [Tycho-2 catalog](https://cdsarc.u-strasbg.fr/ftp/cats/I/259/) can be found in ./catalog/tycho2/README.md.

## Usage

To use this repository, you can either download the zipped file or clone the respository (you need to install git).
```sh
git clone https://github.com/CoFFeeSooDa/StarTrackerTest.git
```
After the download, the script in ./catalogs/tycho2/combine.py should be run frist if you choose Tycho-2 as your star catalog.
Also, Hipparcos catalog are also available in the catalog directory. (pending)

### Centroid_Gen.py
Centroid_Gen.py can be run independently (usually served as debugging) or imported in other Python codes.
Cetnroid_Gen.py contains two functions, generate_xyls and calculate_vectors. The latter function is called by generate_xyls.
Here is an example to use the function,
```python
import Centroid_Gen
xyls = generate_xyls(star_table_path, boresight, debug=False, fov=16.7, width=1124, height=1124)
```
where xyls is a numpy array of star-centroid pixels, star_table_path is the path of trimmed star table (.mat files, placed in ./trimmed table), and boresight is set to (RA, Dec) in degrees. 

### Star_Catalog_to_mat.py

This is only a documentation package. You can print out [spec.md](spec.md) to your console:

```sh
$ standard-readme-spec
# Prints out the standard-readme spec
```

### Generator

To use the generator, look at [generator-standard-readme](https://github.com/RichardLitt/generator-standard-readme). There is a global executable to run the generator in that package, aliased as `standard-readme`.

## Badge

If your README is compliant with Standard-Readme and you're on GitHub, it would be great if you could add the badge. This allows people to link back to this Spec, and helps adoption of the README. The badge is **not required**.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

To add in Markdown format, use this code:

```
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
```

## Example Readmes

To see how the specification has been applied, see the [example-readmes](example-readmes/).

## Related Efforts

- [Art of Readme](https://github.com/noffle/art-of-readme) - 💌 Learn the art of writing quality READMEs.
- [open-source-template](https://github.com/davidbgk/open-source-template/) - A README template to encourage open-source contributions.

## Maintainers

[@RichardLitt](https://github.com/RichardLitt).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/RichardLitt/standard-readme/issues/new) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

### Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/RichardLitt/standard-readme/graphs/contributors"><img src="https://opencollective.com/standard-readme/contributors.svg?width=890&button=false" /></a>


## License

[MIT](LICENSE) © Richard Littauer
