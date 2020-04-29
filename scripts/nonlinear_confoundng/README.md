# Nonlinear Confounding Test

In this test I create 30 different datasets with similar statistics. The spatial pattern and time series are drawn in the same ways the only difference is the random seed used. 

The objective of this testing framework is to see how robust each technique is to injecting a non-linear mode.

In this framework there are 8 generative modes. In the first 15 instances 7 are linear and 1 nonlinear, in the remaining 15 instances 6 are linear and 2 nonlinear. There are 4 defined linear time series draw functions and two nonlinear time series draw functions. The linear time series are a mixture of ARMA and SARMA in latent space. The two nonlinear time series are a sigmoid appplied to ARMA or SARMA generated time series. Each mode has a variance of unity. There is also a noise mode which has a variances of unity as well.

The models are trained to find 12 modes each.