# Scripts

Each of the subfolders contains scripts used to carry out tests of each of the algorithms for different aspects of the generated data. The object of each of these subfolders are as follows

- `precision` - How does the random seed affect the value of the test metric
- `mixed_sparisty` - How well does each of the methods capture global as well as local modes
- `noise_sensitivity` - How sensitive are each of the methods to the amount of noise
- `time_sensitivity` - How robust are each of the methods to the mtime length of observations
- `variance_sensitivity` - How robust are each of the methods to the presence of a linear dominant mode
- `nonlinvar_sensitivity` - How robust are each of the methods to the presence of a non-linear dominant mode
- `nonlinear_confounding` - How robust are each of the methods to the presence of a non-linear mode
- `nonlincyc_confounding` - How robust are each of the methods to the presence of a cyclic non-linear mode
- `wave_confounding` - How robust are each of the methods to the presence of a non-linear moving wave mode
- `mixed_modes` - How well does each method perform when data is a mixture of different mode types