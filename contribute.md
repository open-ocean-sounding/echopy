# Notes for contributors
Guidelines to contribute with your code in Echopy.

## Script header
The first line of each file shoud be `#!/usr/bin/env python3`, followed by the docstring with the description and license:

```python3
#!/usr/bin/env python3
"""
Algorithms to filter noise.

Copyright (c) 2020 EchoPY

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
```

## Acknowledgements policy
We like to give credit to everyone contributing to EchoPY.

We use the standard notations `__authors__` and `__credits__`, right after the [header](contribute.md#script-header), to indicate the people who worked in the code. `__authors__` refers to who actually wrote the code, while `__credits__` refers to people who provided ideas or made suggestions to implement the code. Authors and credits might be broken up in different lines to provide further details about the contribution. Example:

```python
__authors__ = ['Peter Parker' , # filter_noise()
               'Mary Jane'    , # binning(), smoothing()
               ]
__credits__ = ['Otto Octavius', # filter_noise()
               ]
```

If you are implementing someone else's algorithm or idea, you should add the full citation in the function docstring. Add the name, followed by "(unpub.)". If not published yet, then provide as many details as you can, explaining how the algorithm works. Example:

```python3
def ryan(Sv, m=5, n=1, thr=10):
    """
    Mask impulse noise following the two-sided comparison method described in:        
        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in 
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    Args:
        Sv  (float)    : 2D array with Sv data to be masked (dB).
        m   (int)      : vertical binning length (n samples or range).
        n   (int)      : number of pings to compare with, either side.
        thr (float)    : user-defined threshold value (dB).
    """
```
 
## Structuring functions
 
EchoPY contains scripts for every type of operation (E.g mask impulse noise, estimate background noise, etc.). Functions to perform the same operation but with different approaches should be together in the same script. Example for `mask_impulse.py`:

```python3
"""
Impulse noise algorithms.

Copyright (c) 2020 EchoPY

Permission is ...
"""

def ryan(Sv, r):
    """
    Ryan's algorithm.
    """
    # TODO

def smith(Sv, r):
    """
    Smith's algorithm.
    """
    # TODO
```
 
This structure enables switching between algorithms in an easy way:
```python3
import mask_impulse
 
# mask impulse noise with Ryan et al. (2015) and Smith's (unpub.) algorithms
mask_ryan  = mask_impulse.ryan (Sv, r)
mask_smith = mask_impulse.smith(Sv, r)
```
 
## Documenting functions
 
Functions must be documented using the [python docstring convention](https://www.python.org/dev/peps/pep-0257/) and the [google docstring format](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings). For example:
```python3
def find_cool(Sv, r):
    """
    Look for cool things in Sv data and returns a mask with the cool things identified.
   
    Args:
        Sv (float): 2D array with Sv data (dB).
        r  (float): 1D array with range data (m).
       
    Returns:
        bool: 2D array with True values pointing at cool things.
    """
    # TODO: code something smart here
   
    return mask
```
 
## Variables names
 
Variable names and units adhere to the standards in Fisheries acoustics:
 
* Maclennan et al. (2002) ‘A consistent approach to definitions and symbols in fisheries acoustics’, ICES Journal of Marine Science 59: 365–369.
* Simmonds and MacLennan (2005) ‘Fisheries Acoustics: Theory and Practice’, London: Blackwell Science Ltd.
* ICES (2016) ‘A metadata convention for processed acoustic data from active acoustic systems’, SISP 4 TG-AcMeta, Version 1.10, ICES WGFAST Topic Group, TG-AcMeta. 47 pp.
 
with minor adaptations to simplify the code:
```
sv    : volume backscattering coefficient (linear, m-1)
Sv    : volume backscatter strength (logarithmic, dB re m-1)
sa    : area backscattering coefficient (linear, m2 m-2)
Sa    : area backscattering strength (logarithmic, dB re m2 m-2)
nasc  : nautical area scattering coefficient (linear, m2 nmi-2)
NASC  : nautical area scattering strength (logarithmic, dB re m2 nmi-2)
ts    : target strength (linear, m-2)
TS    : target strength (logarithmic, dB re m-2)
theta : alongship angle (degrees)
phi   : atwartship angle (degrees)
r     : range (m)
t     : time (numpy datetime format, yyyy—mm-ddTHH:MM:SS.SSS)
c     : sound speed (m s-1)
f     : frequency (kHz)
alpha : absorption (dB m-1)
T     : temperature (degrees Celsius)
S     : salinity (psu)
```

Other common variable names are:
```
mask : boolean 2D array commonly used to mask Sv
thr  : threshold value
bgn  : background noise
r0   : upper range interval
r1   : lower range interval
t0   : start time interval
t1   : end time interval
f0   : start frequency interval
f1   : end time interval
```

Variable frequency is identified by a suffix, e.g:
```
Sv18, nasc38, theta70, r120, t200, mask333
``` 

Further information is added after the frequency, e.g:
```
Sv38masked   :  38 kHz Sv               (masked)
Sv70clean    :  70 kHz Sv               (clean)
bgn120binned : 120 kHz Background noise (clean)
mask38in     :  38 kHz mask             (impulse noise)
mask70as     :  70 kHz mask             (attenuated signal)
mask120tn    : 120 kHz mask             (transient noise)
mask200sb    : 200 kHz mask             (seabed)
```
 
## Matrix operations
 
We use [standard index notation](https://en.wikipedia.org/wiki/Matrix_(mathematics))
for matrices and for-loop operations:
```
i   : 1st dimension iterator             (iterates along-range    )
j   : 2nd dimension iterator             (iterates along-time     )
k   : 3rd dimension iterator             (iterates along-frequency)
m   : number of rows, or matrix height   (range extent            )
n   : number of columns, or matrix width (time extent             )
o   : number of layers, or matrix depth  (frequency extent        )
idx : i indexes                          (along-range indexes     )
jdx : j indexes                          (along-time indexes      )
kdx : k indexes                          (along-frequency indexes )
```

# Notes for maintainers

`echopy` is packaged acccording to the [Python Packaging User Guide](https://packaging.python.org/tutorials/packaging-projects/). For
a new release, update the version number in `setup.py` and `echopy/__init__.py` and then:
```
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel
python3 -m pip install --user --upgrade twine
python3 -m twine upload  dist/*
```

This allows for installation using `pip install echopy`.
