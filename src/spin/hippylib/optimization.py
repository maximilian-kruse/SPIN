r"""This module provides a modern wrapper to Hippylib's Newton-CG solver.

Parametrization is done via data classes, all input and output vectors are numpy arrays.

The input Hippliyb inference model used for optimization is assumed to provide
a cost functional in form of a negative log-posterior. Thus, the optimal parameter value found by
the optimizer approximizes the maximum a-posteriori (MAP) estimate. In SPIN, we consider Gaussian
prior and noise models, resulting in an optimization problem of the form

$$
    \mathbf{m}_{\text{MAP}} = \underset{\mathbf{m}}{\text{argmin}}\
    \frac{1}{2}||\mathbf{F}(m)-\mathbf{d}_{obs}||_{R_{\text{noise}}}^2
    +\frac{1}{2}||\mathbf{m}-\mathbf{m}_{pr}||_{R_{\text{prior}}}^2.
$$

Classes:
    SolverSettings: Configuration of the Newton-CG solver.
    SolverResult: Data class for storage of solver results
    NewtonCGSolver: Wrapper class for the inexact Newton-CG solver implementation in Hippylib.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import hippylib as hl
import numpy as np
import numpy.typing as npt
from beartype.vale import Is

from spin.fenics import converter as fex_converter


# ==================================================================================================
@dataclass
class SolverSettings:
    """Configuration of the Newton-CG solver.

    All attributes have default values.

    Attributes:
        relative_tolerance (Real): Relative tolerance for the gradient norm
            (compared to initial guess).
        absolute_tolerance (Real): Absolute tolerance for the gradient norm.
        gradient_projection_tolerance (Real): Tolerance for the inner product (g,dm), where g is the
            current gradient and dm the search direction.
        max_num_newton_iterations (int): Maximum number of Newton iterations.
        num_gauss_newton_iterations (int): Number of Gauss-Newton iterations performed initially,
            before switching to full Newton.
        coarsest_tolerance_cg (Real): Termination tolerance for the conjugate gradient solver.
        max_num_cg_iterations (int): Maximum number of conjugate gradient iterations.
        armijo_line_search_constant (Real): Constant for the Armijo line search.
        max_num_line_search_iterations (int): Maximum number of line search iterations.
        verbose (bool): Whether to print the solver output.
    """

    relative_tolerance: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-6
    absolute_tolerance: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-12
    gradient_projection_tolerance: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-18
    max_num_newton_iterations: Annotated[int, Is[lambda x: x > 0]] = 20
    num_gauss_newton_iterations: Annotated[int, Is[lambda x: x > 0]] = 5
    coarsest_tolerance_cg: Annotated[Real, Is[lambda x: 0 < x < 1]] = 5e-1
    max_num_cg_iterations: Annotated[int, Is[lambda x: x > 0]] = 100
    armijo_line_search_constant: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-4
    max_num_line_search_iterations: Annotated[int, Is[lambda x: x > 0]] = 10
    verbose: bool = True


@dataclass
class SolverResult:
    """Data class for storage of solver results.

    Attributes:
        optimal_parameter (npt.NDArray[np.floating]): Optimal parameter, found by the optimizer.
        forward_solution (npt.NDArray[np.floating]): Solution of the PDE problem for the optimal
            parameter.
        adjoint_solution (npt.NDArray[np.floating]): Solution of the adjoint problem for the optimal
            parameter.
        converged (bool): Whether the solver has converged.
        num_iterations (int): Number of Newton iterations.
        termination_reason (str): Reason for termination.
        final_gradient_norm (Real): Final gradient norm.
    """

    optimal_parameter: npt.NDArray[np.floating]
    forward_solution: npt.NDArray[np.floating]
    adjoint_solution: npt.NDArray[np.floating]
    converged: bool
    num_iterations: Annotated[int, Is[lambda x: x >= 0]]
    termination_reason: str
    final_gradient_norm: Annotated[Real, Is[lambda x: x >= 0]]


# ==================================================================================================
class NewtonCGSolver:
    """Wrapper class for the inexact Newton-CG solver implementation in Hippylib.

    This class mainly exists to provide a more modern, convenient, and consistent interface to the
    underlying Hippylib functionality.
    The incomplite Newton-CG method constitutes a combination of algorithms that efficienly solve
    large-scale optimization problems, with good scalability in terms of the number of degrees of
    freedom. The outer Newton iterations are known to converge independently of the problem size for
    a wide range of applications. The linear system at each Newton step is solved inexactly using
    the conjugate gradient (CG) method. The CG solver is terminated early according to Steihaug and
    Eisenstat-Walker stopping criteria. This ensures termination independently of the problem size
    as well. For globalization, armijo line search is utilized.

    For more information on the implementation of the Solver in hippylib,
    check out the
    [`NewtonCG`](https://hippylib.readthedocs.io/en/latest/hippylib.algorithms.html#module-hippylib.algorithms.NewtonCG)
    documentation.

    Methods:
        run: Start the solver with an initial guess.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, solver_settings: SolverSettings, inference_model: hl.Model) -> None:
        """Constructor, initializing solver according to settings.

        Args:
            solver_settings (SolverSettings): Solver configuration.
            inference_model (hl.Model): Hippylib inference model defining the optimization problem.
        """
        hippylib_parameterlist = hl.ReducedSpaceNewtonCG_ParameterList()
        hippylib_parameterlist["rel_tolerance"] = solver_settings.relative_tolerance
        hippylib_parameterlist["abs_tolerance"] = solver_settings.absolute_tolerance
        hippylib_parameterlist["gdm_tolerance"] = solver_settings.gradient_projection_tolerance
        hippylib_parameterlist["max_iter"] = solver_settings.max_num_newton_iterations
        hippylib_parameterlist["GN_iter"] = solver_settings.num_gauss_newton_iterations
        hippylib_parameterlist["cg_coarse_tolerance"] = solver_settings.coarsest_tolerance_cg
        hippylib_parameterlist["cg_max_iter"] = solver_settings.max_num_cg_iterations
        if solver_settings.verbose:
            hippylib_parameterlist["print_level"] = 0
        else:
            hippylib_parameterlist["print_level"] = -1
        hippylib_parameterlist["LS"]["c_armijo"] = solver_settings.armijo_line_search_constant
        hippylib_parameterlist["LS"]["max_backtracking_iter"] = (
            solver_settings.max_num_line_search_iterations
        )
        self._hl_newtoncgsolver = hl.ReducedSpaceNewtonCG(inference_model, hippylib_parameterlist)
        self._inference_model = inference_model

    # ----------------------------------------------------------------------------------------------
    def solve(self, initial_guess: npt.NDArray[np.floating]) -> SolverResult:
        """Run the solver, given an initial guess.

        Args:
            initial_guess (npt.NDArray[np.floating]): Initial guess for the optimization problem.

        Raises:
            ValueError: Checks if the initial guess has the correct size.

        Returns:
            SolverResult: Optimal solution and metadata.
        """
        function_space_variables = self._inference_model.problem.Vh[hl.STATE]
        function_space_parameter = self._inference_model.problem.Vh[hl.PARAMETER]
        if not initial_guess.size == function_space_parameter.dim():
            raise ValueError(
                f"Initial guess has wrong size {initial_guess.size}, "
                f"expected {function_space_parameter.dim()}."
            )

        initial_function = fex_converter.convert_to_dolfin(
            initial_guess.copy(), function_space_parameter
        )
        initial_vector = initial_function.vector()
        forward_solution, optimal_parameter, adjoint_solution = self._hl_newtoncgsolver.solve(
            [None, initial_vector, None]
        )
        optimal_parameter = fex_converter.convert_to_numpy(
            optimal_parameter, function_space_parameter
        )
        forward_solution = fex_converter.convert_to_numpy(
            forward_solution, function_space_variables
        )
        adjoint_solution = fex_converter.convert_to_numpy(
            adjoint_solution, function_space_variables
        )
        solver_result = SolverResult(
            optimal_parameter=optimal_parameter,
            forward_solution=forward_solution,
            adjoint_solution=adjoint_solution,
            converged=self._hl_newtoncgsolver.converged,
            num_iterations=self._hl_newtoncgsolver.it,
            termination_reason=self._hl_newtoncgsolver.termination_reasons[
                self._hl_newtoncgsolver.reason
            ],
            final_gradient_norm=self._hl_newtoncgsolver.final_grad_norm,
        )
        return solver_result
