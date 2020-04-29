# Using Monte Carlo synthetic modes to test methods of pattern discovery in climate data
### code and supporting materials

This repo includes a library `pyCliMine` for generating climate-like synthetic modes and mixing them into data sets. It also contains code for applying slow feature analysis (SFA), dynamic mode decomposition (DMD), and principal component analysis (PCA) to these mixed datasets and quantifying how accurate the matches are.

For example experiments on using this code see the `scripts` folder.

### Gernated modes

Modes generated with this package can linear dense or linear sparse like below

![](images/linear_modes_spatial_example.png)

and we can generate accompanying time series

![](images/linear_modes_time_example.png)

We can also create non-linear Modes

![](images/nonlin.gif)

and non-linear cyclic modes

![](images/nonlincyc.gif)

### Environment

This repo relies on a significant amount of external libraries for full functionality. An overcomplete environment file is included here as an example.
