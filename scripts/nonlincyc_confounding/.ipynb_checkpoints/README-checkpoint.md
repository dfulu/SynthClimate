# Nonlinear Confounding Test

In this test I create 30 different datasets with similar statistics. The spatial pattern and time series are drawn in the same ways the only difference is the random seed used. 

The objective of this testing framework is to see how robust each technique is to injecting a non-linear cyclic mode.

In this framework there are 8 generative modes. In the first 15 instances 7 are linear and 1 nonlinear cyclic, in the remaining 15 instances 6 are linear and 2 nonlinear cyclic. There are 4 defined linear time series draw functions and two nonlinear cyclic time series draw functions. The linear time series are a mixture of ARMA and SARMA in latent space. One of the nonlinear cyclic time series functions has a linear smoothly increasing phase $\theta(t) \alpa t$. The other nonlinear cyclic time series function has a phase where the increase at each time step is generated by an ARMA process, $\theta_t - \theta_{t-1} ~ ARMA(p,q) + c$, where c is a constant such that $\theta_t - \theta_{t-1}$ is always positive. Each mode has a variance of unity. There is also a noise mode which has a variances of unity as well.

The models are trained to find 12 modes each.