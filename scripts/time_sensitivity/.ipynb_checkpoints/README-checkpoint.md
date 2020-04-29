# Time Sensitivity Test

In this test I create datasets which have one of 6 sets of modes and one of 7 time lengths - from 20--2000 years

The objective of this testing framework is to see how sensitive each method is tothe length of the time series. This is an important consideration to see how the results of other tests might stand up to shorter observational datsasets.

In this framework there are 8 generative modes which are all of the dense-linear type. There are 4 defined time series draw functions which are used twice each. The time series are a mixture of ARMA and SARMA. Each mode has a variance of unity. There is also a noise mode which has a variances of unity as well.

The models are trained to find 12 modes each.