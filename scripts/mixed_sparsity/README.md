# Mixed Sparsity 

In this test I create 30 different datasets with similar statistics. The spatial pattern and time series are drawn in the same ways the only difference is the random seed used. 

The objective of this testing framework is to see how each method performs on sparse and dense modes. 

In this framework there are 8 generative modes, 4 of which dense-linear type and 4 of which sparse-linear type. There are 4 defined time series draw functions which are used twice each. The time series are a mixture of ARMA and SARMA. Each mode has a variance of unity. There is also a noise mode which has a variances of unity as well.

The models are trained to find 12 modes each.