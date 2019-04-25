# echopy design conventions
 
## Structuring new algorithms
 
Echopy contains scripts for every type of operation (E.g mask impulse noise, estimate background noise, etc.). Algorithms to perform the same operation but with different approaches should be together in the same script. For example:
```
*mask_impulse.py
  *ryan
  *smith
  *...
 
*get_background.py
  *derobertis
  *smith
  *...
```
 
This structure enables switching between algorithms in an easy way:
```
import mask_impulse as maskIN
import get_background as getBGN
 
# mask impulse noise with Ryan et al. (2015) and Smith's algorithms
mask_ryan  = maskIN.ryan(Sv, r)
mask_smith = maskIN.smith(Sv, r)
 
# get background noise with the De Robertis and Higginbottom (2007) and Smith's algorithms
bgn_derobertis = getBGN.derobertis(Sv, r, alpha)
bgn_smith      = getBGN.smith(Sv, r, alpha)
```
 
## Documenting new algorithms
 
New processing algorithms must be documented using the [python docstring convention](https://www.python.org/dev/peps/pep-0257/) and the [google docstring format](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings). For example:
```
def find_cool(Sv, r):
    """
    Look for cool things in Sv data and returns a mask with the cool things identified.
   
    Args:
        Sv (float): 2D array with Sv data (dB).
        r (float):  1D array with range data (m).
       
    Returns:
        bool: 2D array with mask pointing at cool things.
    """
    # TODO: code something smart here
   
    return mask
```
 
**We strongly recommend adding the full citation in the docstring if the algorithm is published**. Please add "unpub." otherwise and provide as many details as you can, explaining how the algorithm works.
 
## Acoustic and oceanographic variables
 
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
 
## Matrix operations
 
We use [standard index notation](https://en.wikipedia.org/wiki/Matrix_(mathematics))
for matrices and for-loop operations:
```
i   : vertical or 1st dimension iterator (normally, the along-range iterator)
j   : horizontal or 2nd dimension iterator (normally, the along-time iterator)
m   : number of rows, or bin height (normally, along-range)
n   : number of columns, or bin width (normally, along-time)
idx : i indexes
jdx : j indexes
```
 
## Other variables:
 
Other common variable names are:
```
mask : boolean 2D array commonly used to mask Sv
thr  : threshold value
bgn  : background noise
r0   : upper range interval
r1   : lower range interval
t0   : start time interval
t1   : end time interval
```
 
## Adding information to variables
 
Frequencies are identified by a suffix, e.g:
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
