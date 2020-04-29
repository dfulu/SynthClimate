# Precision Test

In this test I create 30 different datasets with similar statistics. The spatial pattern and time series are drawn in the same ways the only difference is the random seed used. 

The objective of this testing framework is to see how much the particular draw affects the outcome of the test. So basically how is the metric affected by thre random seed. This will be an important consideration when considering further more complicated tests. 

In this framework there are 8 generative modes which are all of the dense-linear type. There are 4 defined time series draw functions which are used twice each. The time series are a mixture of ARMA and SARMA. Each mode has a variance of unity. There is also a noise mode which has a variances of unity as well.

The models are trained to find 12 modes each.