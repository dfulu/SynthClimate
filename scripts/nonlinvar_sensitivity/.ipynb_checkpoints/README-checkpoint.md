# Nonlinear Confounding Test

In this test I create 30 different datasets with similar statistics. The spatial patterns are drawn in the same ways the only difference is the random seed used. 

The objective of this testing framework is to see how robust each technique is to differnt variances of a non-linear injected mode.

In this framework there are 8 generative modes. there 7 are linear and 1 nonlinear modes. There are 4 defined linear time series draw functions and two nonlinear time series draw functions. The linear time series are a mixture of ARMA and SARMA in latent space. The two nonlinear time series are a sigmoid appplied to ARMA or SARMA generated time series. Each linear mode has a variance of unity and the variance of the non-linear modes takes on values `[2,4,8,16,32]`. There is also a noise mode which has a variance of unity as well.

We set up 5 trials , meaning 5 different draws of the spatial pattern. Under each trial we carry out the test procedure with each non-linear variance. this is so that we can test the influence of the variance of the non-linear mode alone holding everything else constant.

The models are trained to find 12 modes each.