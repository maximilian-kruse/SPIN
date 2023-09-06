"""Initialization file for the SP inference package

This is the top-level initialization of the SP inference package. The package comprises three 
sub-packages, as well as two stand-alone modules. For more detailed information, please refer to 
the respective sub-packages and modules.

Sub-Packages:
=============

1) inference:
-------------
Contains the actual Bayesian inference functionality. This is a hIPPYlib wrapper for stochastic
processes, an MCMC sampler for nonlinear problems, and the core implementation of the optimization
problem for transient problems.

2) pde_problems:
----------------
Contains the FEM capabilities necessary to solve the arising PDE problems. The variational forms of
the implemented generating equations are given alongside low-level functionalities for transient
assembly and solves. In addition, higher level solver objects facilitate the FEM problem setup and
transient data generation.

3) utilities:
-------------
Supplementary functions, including data type conversion, interpolation/projection routines and the
logger.

Modules:
========

1) postprocessing:
------------------
Result processing and visualization.

2) processes:
-------------
Example processes for artificial data generation.
"""

from .inference import model, sampling, transient
from .pde_problems import forms, functions, problems
from .utilities import general, interpolation, logging
from .postprocessing import visualization, data_types
from . import processes