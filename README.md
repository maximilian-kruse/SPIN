![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FUQatKIT%2FSPIN%2Fmain%2Fpyproject.toml)
![License](https://img.shields.io/github/license/UQatKIT/SPIN)
![Beartype](https://github.com/beartype/beartype-assets/raw/main/badge/bear-ified.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

# SPIN: Stochastic Process INference

SPIN is a Python package for the non-parametric Bayesian inference of the parameter functions of autonomous diffusion processes. We consider processes $ \mathbf{X}_t $ on a domain $\Omega\in\mathbb{R}^d$, indexed over $ t\in\mathbb{R}_+ $. For a given drift vector $\mathbf{b}: \Omega\to\mathbb{R}^d$ and (quadratic) squared diffusion matrix $\mathbf{\Sigma}:\Omega\to\mathbb{R}^{d\times d}$, such processes can be described by an Ito SDE,

$$
    d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t) dt + \sqrt{\mathbf{\Sigma}(\mathbf{X}_t)} d\mathbf{W}_t,\quad \mathbf{X}(t=0)=\mathbf{X}_0\ a.s.,
$$

where $\mathbf{W}_t\in\mathbb{R}^d$ is a Wiener process.

SPIN infers $\mathbf{b}(\mathbf{x})$ and $\mathbf{\Sigma}(\mathbf{x})$ as functions of states-space from trajectory statistics data. These statistics are either the PDF of the process $p(\mathbf{x},t)$, or the moments $\tau_n(\mathbf{x})$ of the distribution of first exit times from a bounded domain $\mathcal{A}$, defined as

$$
    \tau(x) \coloneqq\inf \\{ t\geq 0: \mathbf{X}_t\notin\mathcal{A}|\mathbf{X}_0=\mathbf{x} \\} ,\quad \tau_n(\mathbf{x}) \coloneqq \mathbb{E}[\tau^n(\mathbf{x})].
$$

The inference is governed by PDE models defined through the Kolmogorov equations. Given the infinitesimal generator $\mathcal{L}$ of the above process,

$$
    \mathcal{L} = \mathbf{b}(\mathbf{x})\cdot\nabla + \frac{1}{2}\mathbf{\Sigma}(\mathbf{x})\colon \nabla\nabla,
$$

the PDE of the process is governed by the Kolmogorov backward or Fokker-Planck equation,

$$
    \frac{\partial p}{\partial t}(\mathbf{x},t) = \mathcal{L}^*p(\mathbf{x},t),\quad p(\mathbf{x},t) = p_0(\mathbf{x}).
$$

On the other hand, the Kolomogorov forward equation gives rise to a hierarchy of stationary PDEs for the first passage time moments,

$$
\begin{gather*}
    \mathcal{L}\tau_1 = -1,\ x\in\mathcal{A},\quad \tau_1 = 0,\ x\in\partial\mathcal{A}, \\
    \mathcal{L}\tau_n = -n\tau_{n-1},\ x\in\mathcal{A},\quad \tau_n = 0,\ x\in\partial\mathcal{A}.
\end{gather*}
$$

SPIN utilizes the solution of above PDEs at a discrete number of given points as a parameter-to-observable map $\mathcal{F}(\mathbf{m})$, which is then utilized in a function space Bayesian framework. The underlying discretization of the involved PDE problems is based on the finite element method.

Further details regarding the theoretical background can be found in the accompanying publication,

***[Non-parametric Inference for Diffusion Processes:
A Computational Approach via Bayesian Inversion
for PDEs](https://arxiv.org/abs/2411.02324)***

## Getting Started

SPIN depends on a mixture of pip and conda dependencies, which can be efficiently managed using [Pixi](https://pixi.sh/latest/). To get started, simply run in the root directory

```bash
pixi install -e all
```

The [documentation](https://uqatkit.github.io/SPIN/) provides further information regarding usage, theoretical background, technical setup and API. We further provide runnable notebooks under [`examples`](https://github.com/UQatKIT/SPIN/tree/main/examples).

## Acknowledgement and License

SPIN is being developed in the research group [Uncertainty Quantification](https://www.scc.kit.edu/forschung/uq.php) at KIT.
Large portions of SPIN are based on the [hIPPYlib](https://dl.acm.org/doi/10.1145/3428447) software library for large-scale (Bayesian) inverse problems. hIPPYlib, in turn, uses [FEniCS](https://fenicsproject.org/) for finite element computations.
It is distributed as free software under the [MIT License](https://choosealicense.com/licenses/mit/).