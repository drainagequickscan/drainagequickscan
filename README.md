# pydijkdrainage
_Using TimML for geohydrological modelling of dike cross-sections_

## Introduction
pydijkdrainage is a python module for creating cross-sectional geohydrological models for dikes near rivers. The models are built using [TimML](https://github.com/mbakker7/timml).

**Note:** _This repository is private and will remain so unless express permission is given by Witteveen+Bos and the POV-Piping._

## Description
The tool is essentially a wrapper around [TimML](https://github.com/mbakker7/timml). This module makes certain assumptions about model structure that it uses to create a TimML model. It includes a bunch of useful functions for creating and visualizing this cross-sectional models.

## Example usage
The model requires quite a few parameters to be entered, which are collected in dictionaries prior to passing them to the model. To see the expected parameters contained within the different dictionaries (`xdict`, `zdict`, `kdict` and `wdict`), take a look at the dictionaries stored in the XsecDikeModel object:

```{python}
>>> import pydijkdrainage as pdd
>>> pdd.XsecDikeModel._xdict

{'breedte_voorland': nan,
 'breedte_dijk_totaal': nan,
 'breedte_dijk_kruin': nan,
 'breedte_dijk_berm': nan,
 'breedte_kwelsloot': nan}
```

Once all the parameters have been defined, the model is built by calling the XsecDikeModel object.

```{python}
>>> xsm = pdd.XsecDikeModel(xdict, zdict, kdict, wdict)
```

Solve the reference model:

```{python}
>>> xsm.solve()

Number of elements, Number of equations: 8 , 40
........
solution complete
```
Visualize a pretty representation of the cross-section with

```{python}
>>> xsm.plots.xsec()
```
or to see the actual model structure use

```{python}
>>> xsm.plots.inhoms()
```

Adding drainage to a cross-section is done by calling one of the drainage objects:
- HorizontalDrainXsec
- GrindkofferXsec
- VerticalWellsXsec

These objects take an XsecDikeModel as input. Note that providing a XsecDikeModel object that was initialized earlier will alter that object as well! To avoid altering earlier models, just create a new one when creating one with drainage, e.g.

```{python}
>>> xsm_ref = pdd.XsecDikeModel(xdict, zdict, kdict, wdict)
>>> xsm_drn = pdd.HorizontalDrainXsec(xsm_ref, x_drn=5., drawdown=2.0)
>>> xsm_drn.solve()

Number of elements, Number of equations: 9 , 41
........
solution complete
``` 

Some example scripts are included in the examples directory.

## Example output

The following images show examples of the output.

### Model structure (with `xsm.plots.inhoms()`)
![model_structure](https://github.com/ArtesiaWater/pydijkdrainage/blob/master/examples/figures/model_structure_grindkoffer.png)

### Model output in cross section (with `xsm.plots.xsec()`)
![cross_section](https://github.com/ArtesiaWater/pydijkdrainage/blob/master/examples/figures/xsec_stromingen_figure.png)

### Model output top view (with `xsm.plots.topview()`)
![topview_example](https://github.com/ArtesiaWater/pydijkdrainage/blob/master/examples/figures/verticale_bronnen_eindresultaat.png)

## Background
The code is based on the tool that is available on [www.drainagequickscan.nl](http://www.drainagequickscan.nl). This tool was developed by Witteveen+Bos and commissioned by the POV-Piping. 


