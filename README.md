# drainagequickscan
_Using TimML for geohydrological modelling of dike cross-sections_

## Introduction
drainagequickscan contains python code for creating cross-sectional geohydrological models for river dikes. The models are built using [TimML](https://github.com/mbakker7/timml). The python code is based on the code behind the [Quickscan Drainagetechnieken website](#www.drainagequickscan.nl).

## Description
The tool is essentially a wrapper around [TimML](https://github.com/mbakker7/timml). This module makes certain assumptions about model structure which are used to create a TimML model. It includes a bunch of useful functions for creating and visualizing cross-sectional models.

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
![model_structure](https://github.com/drainagequickscan/drainagequickscan/blob/master/examples/figures/model_structure_grindkoffer.png)

### Model output in cross section (with `xsm.plots.xsec()`)
![cross_section](https://github.com/drainagequickscan/drainagequickscan/blob/master/examples/figures/xsec_stromingen_figure.png)

### Model output top view (with `xsm.plots.topview()`)
![topview_example](https://github.com/drainagequickscan/drainagequickscan/blob/master/examples/figures/verticale_bronnen_eindresultaat.png)

## Background
The code is based on the tool that is available on [www.drainagequickscan.nl](http://www.drainagequickscan.nl). This tool was developed by Witteveen+Bos and commissioned by the POV-Piping. 

<img height=100 src="data:image/png;base64,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" alt="WB Logo">
&emsp; <img height=100, src="data:image/jpeg;base64,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" alt="POVPiping Logo>