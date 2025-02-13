r"""Weak form in UFL syntax for the Kolmogorov PDEs.

This module implements the weak forms for the Kolmogorov equations, which describe the PDEs
governing stochastic process inference in SPIN.
For a domain $\Omega\subset\mathbb{R}^d$, drift vector $\mathbf{b}(\mathbf{x})$, and squared
diffusion $\mathbf{\Sigma}(\mathbf{x})$, the Kolmogorov equations are base on the
*infinitesimal Generator* of a process, defined as

$$
    \mathcal{L} = \mathbf{b}(\mathbf{x})\cdot\nabla
    + \frac{1}{2}\mathbf{\Sigma}(\mathbf{x})\colon\nabla\nabla
$$

In SPIN, we consider the Kolmogorov forward, as well as backward equation. The forward equation,
better known as Fokker-Planck equation, is defined in
[`weak_form_fokker_planck`][spin.core.weakforms.weak_form_fokker_planck]. It governs the evolution
of the law or distribution $p: \Omega\times\mathbb{R}_+ \to \mathbb{R}_+$ of a process over space
and time.
On the other hand, the backward equation gives rise to a hierarchy of PDEs that governes the moments
of the mean exit time or mean first passage time distribution of a process. The first passage time
of a process $\mathbf{X}_t$ is defined as

$$
    \tau(\mathbf{x}) = \inf\{t\geq 0: X_t\notin\Omega | \mathbf{X}_0 = \mathbf{x}\}.
$$
The respective weak forms are implemented in
[`weak_form_mean_exit_time`][spin.core.weakforms.weak_form_mean_exit_time] and
[`weak_form_mean_exit_time_moments`][spin.core.weakforms.weak_form_mean_exit_time_moments].

Functions:
    weak_form_mean_exit_time: UFL weak form for the mean exit time PDE.
    weak_form_mean_exit_time_moments: UFL weak form for the PDE system yielding the first two
        moments of the exit time.
    weak_form_fokker_planck: Weak form for the spatial contribution to the Fokker-Planck equation.
"""
import dolfin as dl
import ufl


# ==================================================================================================
def weak_form_mean_exit_time(
    forward_variable: ufl.Argument | dl.Function,
    adjoint_variable: ufl.Argument | dl.Function,
    drift: ufl.Coefficient | ufl.tensors.ListTensor,
    squared_diffusion: ufl.tensors.ListTensor,
) -> ufl.Form:
    r"""UFL weak form for the mean exit time PDE.

    Given the definitions above, the PDE reads

    $$
    \begin{gather}
        \mathcal{L}\tau + 1 = 0, \quad \mathbf{x}\in\Omega, \\
        \tau = 0, \quad \mathbf{x}\in\partial\Omega.
    \end{gather}
    $$

    Forward and adjoint variables need to be defined on scalar function spaces, drift and diffusion
    are a vector and matrix functions, respectively.

    Args:
        forward_variable (ufl.Argument | dl.Function): Forward or trial variable
        adjoint_variable (ufl.Argument | dl.Function): Adjoint or test variable
        drift (ufl.Coefficient | ufl.tensors.ListTensor): Drift vector function
        squared_diffusion (ufl.tensors.ListTensor): Diffusion tensor function

    Returns:
        ufl.Form: Resulting UFL form
    """
    weak_form = (
        dl.dot(drift * adjoint_variable, dl.grad(forward_variable)) * dl.dx
        - 0.5
        * dl.dot(dl.div(squared_diffusion * adjoint_variable), dl.grad(forward_variable))
        * dl.dx
        + dl.Constant(1) * adjoint_variable * dl.dx
    )
    return weak_form


# --------------------------------------------------------------------------------------------------
def weak_form_mean_exit_time_moments(
    forward_variable: ufl.Argument | dl.Function,
    adjoint_variable: ufl.Argument | dl.Function,
    drift: ufl.Coefficient | ufl.tensors.ListTensor,
    squared_diffusion: ufl.tensors.ListTensor,
) -> ufl.Form:
    r"""UFL weak form for the PDE system yielding the first two moments of the exit time.

    Given the definitions above, the PDE reads

    $$
    \begin{gather}
        \mathcal{L}\tau_1 + 1 = 0, \quad \mathbf{x}\in\Omega, \\
        \tau_1 = 0, \quad \mathbf{x}\in\partial\Omega,
    \end{gather}
    $$

    and

    $$
    \begin{gather}
        \mathcal{L}\tau_2 + 2\tau_1 = 0, \quad \mathbf{x}\in\Omega, \\
        \tau_2 = 0, \quad \mathbf{x}\in\partial\Omega.
    \end{gather}
    $$

    Forward and adjoint variables need to be defined on vector function spaces, drift and diffusion
    are a vector and matrix functions, respectively.

    Args:
        forward_variable (ufl.Argument | dl.Function): Forward or trial variable
        adjoint_variable (ufl.Argument | dl.Function): Adjoint or test variable
        drift (ufl.Coefficient | ufl.tensors.ListTensor): Drift vector function
        squared_diffusion (ufl.tensors.ListTensor): Diffusion tensor function

    Returns:
        ufl.Form: Resulting UFL form
    """
    weak_form_component_1 = (
        dl.dot(drift * adjoint_variable[0], dl.grad(forward_variable[0])) * dl.dx
        - 0.5
        * dl.dot(dl.div(squared_diffusion * adjoint_variable[0]), dl.grad(forward_variable[0]))
        * dl.dx
        + dl.Constant(1) * adjoint_variable[0] * dl.dx
    )
    weak_form_component_2 = (
        dl.dot(drift * adjoint_variable[1], dl.grad(forward_variable[1])) * dl.dx
        - 0.5
        * dl.dot(dl.div(squared_diffusion * adjoint_variable[1]), dl.grad(forward_variable[1]))
        * dl.dx
        + 2 * forward_variable[0] * adjoint_variable[1] * dl.dx
    )
    weak_form = weak_form_component_1 + weak_form_component_2
    return weak_form


# --------------------------------------------------------------------------------------------------
def weak_form_fokker_planck(
    forward_variable: ufl.Argument | dl.Function,
    adjoint_variable: ufl.Argument | dl.Function,
    drift: ufl.Coefficient | ufl.tensors.ListTensor,
    squared_diffusion: ufl.tensors.ListTensor,
) -> ufl.Form:
    r"""Weak form for the spatial contribution to the Fokker-Planck equation.

    Given the definitions above, the weak form defines the spatial contribution
    $\mathcal{L}^*p$ of the time-dependent PDE

    $$
        \frac{\partial p}{\partial t} = \mathcal{L}^*p, \quad, p(\mathbf{x},0) = p_0(\mathbf{x}),
    $$

    Args:
        forward_variable (ufl.Argument | dl.Function): Forward or trial variable
        adjoint_variable (ufl.Argument | dl.Function): Adjoint or test variable
        drift (ufl.Coefficient | ufl.tensors.ListTensor): Drift vector function
        squared_diffusion (ufl.tensors.ListTensor): Diffusion tensor function

    Returns:
        ufl.Form: Resulting UFL form
    """
    weak_form = (
        dl.div(drift * forward_variable) * adjoint_variable * dl.dx
        + 0.5
        * dl.dot(dl.div(squared_diffusion * forward_variable), dl.grad(adjoint_variable))
        * dl.dx
        + dl.Constant(0) * adjoint_variable * dl.dx
    )
    return weak_form
