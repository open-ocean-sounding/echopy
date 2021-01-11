[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/echopy?style=plastic)](https://www.python.org/)
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg?style=plastic)
[![GitHub Download](https://img.shields.io/github/repo-size/open-ocean-sounding/echopy?style=plastic)](https://github.com/open-ocean-sounding/echopy/archive/main.zip)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/echopy?label=PyPI%20downloads&style=plastic)](https://badge.fury.io/py/echopy)
[![GitHub](https://img.shields.io/github/license/open-ocean-sounding/echopy?style=plastic)](LICENSE.md)
<!--[![PyPI](https://img.shields.io/pypi/v/echopy?style=plastic)](https://badge.fury.io/py/echopy)-->

<img src="logo.png" alt="Echopy logo" width="570"/>

Echopy is a multipurpose library containing common algorithms for fisheries acoustic data processing, such as background noise correction, removal of seabed and corrupted pings, target detection, multifrequency analysis, and binning. The library provides building blocks that can be assembled for a variety of fisheries acoustics applications and is committed to transparency, cooperation, and universal access in fisheries acoustics software.

## Installation
Echopy requires [Python 3.6](https://www.python.org/)

To install echopy and all its [requirements](requirements.md) type the following in your preferred command-line terminal:
```
pip install echopy
```

or install the latest version available in the develop branch by cloning the repository: 
```
git clone https://github.com/open-ocean-sounding/echopy.git
pip install ./echopy
```

## Contributing
Echopy has essentially grown from code that was initially implemented to support specific research projects, and later made publicly available for the benefit of all. If you work in fisheries acoustics, there are several ways you can contribute to improve Echopy. You might create an [issue](https://github.com/open-ocean-sounding/echopy/issues), propose an improvement or report a bug, or even create your own repository branch, work on it, and [send pull requests](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/working-with-forks) to the main branch. By doing so, you can  give visibility to your algorithms in fisheries acoustics, while making life easier for your colleagues. Echopy has an [acknowledgments policy](contribute.md#acknowledgements-policy) to make sure that your work is credited. Check out the [contribution](contribute.md) section if you want to be part of Echopy.

## Maintainers
* [**Alejandro Ariza**](https://github.com/alejandro-ariza) - *French National Research Institute for Sustainable Development (IRD)*

## Contributors
* [**Alejandro Ariza**](https://github.com/alejandro-ariza) - *French National Research Institute for Sustainable Development (IRD)*
* [**Robert Blackwell**](https://github.com/RobBlackwell) - *British Antarctic Survey*
* [**Sophie Fielding**](https://github.com/bas-sof) - *British Antarctic Survey*
* **Xinliang WANG** - *Yellow Sea Fisheries Research institute*

## Acknowledgments
Echopy does not currently have any direct funding, but benefits from the hard work, time  and support of researchers and their institutions. We acknowledge [all contributors and institutions involved](README.md#contributors) with special mention of the British Antarctic Survey project [RapidKrill](https://github.com/bas-acoustics/rapidkrill), and [The Antarctic Wildlife Research Fund (AWR)](http://www.antarcticfund.org/) for releasing the first version and giving life to Echopy.  

## License
This software is licensed under the [MIT License](LICENSE).

## Contact
[echopy@protonmail.com](mailto:echopy@protonmail.com)
