# Noise Sensitivity Test

In this test I create 30 different datasets with similar statistics but with different amount of additive noise. The spatial pattern and time series of the modes are drawn in the same ways the only difference between the modes in each of the 30 sets is the random seed used. 

The objective of this testing framework is to see how sensitive each of the pattern recovery tests are to the amount of noise.

In this framework there are 8 generative modes which are all of the dense-linear type. There are 4 defined time series draw functions which are used twice each. The time series are a mixture of ARMA and SARMA. Each mode has a variance of unity. There is also a noise mode which takes on differing amounts of variance.

The models are trained to find 12 modes each.