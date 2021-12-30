# SSbayes Overview
This repository contains a minimally lightweight implementation of various Bayesian state-space or dynamic linear models. There are three core functions located within the ```src``` directory. To briefly elucidate the models implemented:

1. ```discountDLM.cpp``` implements a Bayesian dynamic linear model, with the time-varying dispersion parameters specified by way of discount factors (see [ch. 6](https://link.springer.com/book/10.1007/b98971))

2. ```localLevelDLM.cpp``` implements a static local-level model with conjugate priors on the dispersion parameters, i.e., $V$ and $W$ do not change over time and receive Inverse Wishart priors

$$
\begin{align}
y_t &= F_t\mu_t + v_t, \ v_t\sim\mathcal N(0, V)\\
\mu_t &= G_t\mu_{t-1} + w_t, \ w_t\sim\mathcal N(0, W)
\end{align}
$$

3. ```TVlocalLevelDLM.cpp``` implements a dynamic local-level model with conjugate priors on the time-varying dispersion parameters, i.e., $V_t$ and $W_t$ now change over time (like with the ```discountDLM.cpp``` implementation) and receive Inverse Wishart priors instead of using discount factors.

$$
\begin{align}
y_t &= F_t\mu_t + v_t, \ v_t\sim\mathcal N(0, V_t)\\
\mu_t &= G_t\mu_{t-1} + w_t, \ w_t\sim\mathcal N(0, W_t)
\end{align}
$$
