"""Inference sub-package

The inference sub-package contains three modules providing the actual functionality for Bayesian
inference. For more detailed information, please refer to the respective modules.

Modules:
========

1) model:
---------
Wrapper for a problem setup in the hIPPYlib library that is tailored to the inference of drift 
and/or diffusion functions of stochastic processes.

2) sampling:
------------
MCMC Sampler relying on MAP computation from hIPPYlib and MUQ routines.

3) transient:
-------------
Implementation of the misfit and optimization problem associated with MAP computation
of transient models.
"""
