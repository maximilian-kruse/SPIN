.. image:: doc/logo_uq.svg

============================================
Bayesian Inversion for Stochastic Processes
============================================

This Python package implements the functionalities for Bayesian inversion on stochastic processes.
In particular, from trajectory data it allows for the non-parametric inference of drift and 
diffusion functions of the underlying processes. The inversion procedure is governed by a generator 
PDE of the user's choice, such as the Fokker-Planck equation. The implemented routines constitutes an
all-in-one solution, from simple settings definition to the visualization of the results.

The stochastic inference package provides multiple options for customization and different user settings. It
allows for the separate inference of drift and diffusion functions, as well as the simultaneous
determination of both. It further implements an adjoint-based optimization procedure for determining
the maximum a-posteriori estimate (MAP), utilizing a Newton-CG solver. In combination with a low-rank
approximation of the MAP Hessian, a Gaussian approximation of the posterior can be constructed. In
addition, this approximation can be used to inform a subsequent MCMC procedure for nonlinear problems.

The governing PDE models can be stationary or transient. If necessary, artificial data can be generated
from a number of prototypical processes. Both processes and PDE models are easily extensible for
further use cases.

Most of the library's linearized inference procedures build upon the
`hIPPYlib <https://hippylib.github.io/>`_ project. PDE models are implemented in an FEM formulation
using the `legacy FEniCS (now FEniCSx) <https://fenicsproject.org/>`_. Furthermore, MCMC routines are
provided by `MUQ <https://mituq.bitbucket.io/source/_site/index.html>`_. These two model parts are 
combined through the `hippylib2muq <https://github.com/hippylib/hippylib2muq>`_ wrapper.

===========================
Structure of the Repository
===========================

``top directory:`` Readme, package setup, git info file

``sp_inference:`` Code of the inference library

``examples:`` Notebooks for exemplary use cases

``doc:`` API reference and explanations of theoretical background **(Under construction)**

``test:`` Unit and integration test library

=============================
Requirements and Installation
=============================

The most important, above mentioned dependencies have been tested on an Ubuntu 20.04 LTS operating
system with the following versions (Including the python environment):

+--------------+----------+
| Library      | Version  |
+==============+==========+
| FEniCS       | 2019.1.0 |
+--------------+----------+
| hIPPYlib     | 3.0.0    |
+--------------+----------+
| MUQ          | 0.3.4    |
+--------------+----------+
| hippylib2muq | 0.1.0    |
+--------------+----------+
| Conda        | 4.11.0   |
+--------------+----------+

These libraries, in turn, depend on numerous others. The easiest way is to set up a 
`Conda <https://docs.conda.io/en/latest/>`_ environment with the provided ``environment.yml`` file.
**With Conda installed**, follow these steps:

1) Clone the repository
-----------------------

For ssh:

.. code-block::

    git clone git@git.rwth-aachen.de:sp-bayesian/sp-bayesian-inference.git stochastic_process_inference

For https:

.. code-block::

    git clone https://git.rwth-aachen.de/sp-bayesian/sp-bayesian-inference.git stochastic_process_inference

Change to cloned directory:

.. code-block::

    cd stochastic_process_inference

*Feel free to use any other name for the cloning directory.*

1) Initialize Conda environment
-------------------------------

.. code-block::

    conda env create -f environment.yml
    conda activate sp_inference

3) Install package in editable mode
-----------------------------------

.. code-block::

    pip install -e .

*The package can be used without installation. However, in this case you have to make all
necessary paths available to the import system. In particular, the test library relies on the
package being installed via pip.*

===================
Important Resources
===================

Detailed resources on the theoretical background of the project are provided under ``docs``. Here we
only mention the most important publications.

Bayesian inference on Hilbert spaces:
    A. M. Stuart. “Inverse problems: A Bayesian perspective”. In: Acta Numerica 19 (May
    2013), pp. 451–559. doi: 10.1017/s0962492910000061.

Lagrangian formalism for linearized inference:
   N. Petra and G. Stadler. Model variational inverse problems governed by partial differential
   equations. Tech. rep. The Institute for Computational Engineering and Sciences,
   The University of Texas at Austin, 2011.

hIPPYlib:
   Umberto Villa, Noemi Petra, and Omar Ghattas. “hIPPYlib: An Extensible Software
   Framework for Large-Scale Inverse Problems Governed by PDEs; Part I: Deterministic
   Inversion and Linearized Bayesian Inference”. In: (Sept. 9, 2019). arXiv: 1909.03948
   [math.NA].

===================
Prospective Changes
===================

Extension of Functionality
--------------------------

- Allow for PDE models that are nonlinear in the forward and/or parameter variable
- Allow for more sophisticated transient solvers and stabilization
- Add more process examples
- Implement own prior with more versatile properties
- Combine mean exit time and Fokker-Planck models in vectorized formalism
- Implement PDE formulations with more versatile boundary conditions
- Write how-to guides for important steps (complementary to examples, more specific and advanced)
- Improve prior class: Different variance fields, evaluation of misfit on smaller domain

Technical Improvements
----------------------

- Implement better equality comparison with ``numpy isclose`` instead of ``==``
- Put settings into a global file for reuse
- Run longer MCMC chain by repeatedly restarting shorter simulations and saving the intermediate
  results
- Check for violation of the normalization condition for transient pdf solves
- For long chains, prune the qoi data for visualization
- Make option to pre-assemble structures for second variation of transient problems
- Improve ``setup.py``
- Extend test coverage and include more analytical solutions for comparison
- Allow tuples where constant lists are allowed
- Transient data generation: Provide option to switch between FEM and exact (if possible)