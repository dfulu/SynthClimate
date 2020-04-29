# Variance Sensitivity Test

In this test I create 30 different datasets with similar statistics except for the variance of the first generative mode. The spatial pattern and time series are drawn in the same ways the only difference is the random seed used. 

The objective of this testing framework is to see how sensitive each method is to having one dominant mode. This is important in climate analysis as we have a hierarchy of dominant modes including the yearly cycle and ENSO etc.

In this framework there are 8 generative modes which are all of the dense-linear type. There are 4 defined time series draw functions which are used twice each. The time series are a mixture of ARMA and SARMA. Each mode has a variance of unity except for the first generative mode which takes various values. There is also a noise mode which has a variances of unity as well.

The models are trained to find 12 modes each.