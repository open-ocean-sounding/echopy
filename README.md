# echopy

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

<img src="logos/echopy_color.png" alt="Echopy logo" width="300"/>


## Introduction

Echopy is a multipurpose library containing common algorithms for
fisheries acoustic data processing including: noise removal; binning;
seabed detection; and target detection. The library provides building
blocks that can be assembled for a variety of fisheries acoustics
applications and is aimed towards unsupervised, automatic and large-scale analyses.

## Background

Echopy was created by [Alejandro Ariza](https://github.com/alejandro-ariza) at the [British Antarctic Survey](https://www.bas.ac.uk/) as part of the project [RapidKrill](https://www.bas.ac.uk/project/rapidkrill), which aims to build an automatic system to deliver acustic-based krill estimates from fishing vessels. The project was coordinated by [Sophie Fielding](https://www.bas.ac.uk/profile/sof/) with contributions from [Robert Blackwell](https://github.com/RobBlackwell), both members of the BAS-acoustics team. Due to its flexible structure,  echopy now aims to be a general purpose library for a variety of acoustic processing routines and welcomes acousticians from around the world to get involved.

## Contributing

Echopy is in its early stages, having implemented only a small part of the countless processing and analysis routines in fisheries acoustics. There is still much to do, and we encourage acousticians to audit the code, provide advice, and implement new algorithms.

If contributing, please read [DESIGN.md](DESIGN.md) to adhere to our coding style.

We also provide a [telegram channel](https://t.me/echopy_group) where beginners can find support, and experts can bring ideas and constructive criticism.

## Prerequisites

Echopy requires Python 3.6.

Echopy is usually used in conjunction with [PyEcholab2](https://github.com/CI-CMG/PyEcholab2). PyEcholab is used
to read RAW files and echopy applies common fisheries acoustic data processing techniques. 

In addition, echopy uses the following packages
* [matplotlib](https://matplotlib.org/) for plotting echograms.
* [numpy](http://www.numpy.org/) for large, multi-dimensional arrays.
* [scipy](https://www.scipy.org/) for scientific computing.
* [toml](https://pypi.org/project/toml/) for configuration files.
* [opencv-python](https://pypi.org/project/opencv-python/) for image processing.
* [scikit-image](https://scikit-image.org/) for image processing.

## Packaging

Install echopy:
```
pip install echopy
```
 or download the installer in [https://pypi.org/project/echopy](https://pypi.org/project/echopy).

Notes for maintainers - `echopy` is packaged acccording to the
[Python Packaging User Guide](https://packaging.python.org/tutorials/packaging-projects/). For
a new release, update the version number in `setup.py` and `echopy/__init__.py` and then:

```
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel
python3 -m pip install --user --upgrade twine
python3 -m twine upload  dist/*
```

This allows for installation using `pip install echopy`.

## Maintainer
* **Alejandro Ariza** - *British Antarctic Survey* - [alejandro-ariza](https://github.com/alejandro-ariza)

## Contributors
* **Robert Blackwell** - *British Antarctic Survey* - [RobertBlackwell](https://github.com/RobBlackwell)
* **Sophie Fielding** - *British Antarctic Survey* - [SophieFielding](https://github.com/bas-sof)

## Acknowledgements

Our thanks to the officers, crew and scientists onboard the RRS James
Clark Ross for their assistance in collecting the data. The Western
Core Box cruises and SF are funded as part of the Ecosystems Programme
at the British Antarctic Survey, Natural Environment Research Council,
a part of UK Research and Innovation.

This work was supported by the Antarctic Wildlife Research Fund and
the Natural Environment Research Council grant NE/N012070/1.

## License

This software is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details

## Contact

* [Alejandro Ariza](mailto:alejandro.ariza@ird.fr)

