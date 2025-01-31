import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import dolfin as dl
import hippylib as hl
import numpy as np
import numpy.typing as npt
import ufl
from beartype.vale import Is

from spin.fenics import converter as fex_converter


# ==================================================================================================
class SqrtPrecisionPDEPrior(hl.prior._Prior):  # noqa: SLF001
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        function_space: dl.FunctionSpace,
        variational_form_handler: Callable[[ufl.Argument, ufl.Argument], ufl.Form],
        mean: dl.Function,
        cg_solver_relative_tolerance: Annotated[float, Is[lambda x: 0 < x < 1]] = 1e-12,
        cg_solver_max_iter: Annotated[int, Is[lambda x: x > 0]] = 1000,
    ) -> None:
        self._function_space = function_space
        self._mean = mean
        trial_function = dl.TrialFunction(function_space)
        test_function = dl.TestFunction(function_space)
        self._mass_matrix, self._matern_sdpe_matrix = self._assemble_system_matrices(
            trial_function, test_function, variational_form_handler
        )
        self._mass_matrix_solver, self._matern_spde_matrix_solver = self._initialize_solvers(
            cg_solver_max_iter, cg_solver_relative_tolerance
        )
        quadrature_degree = 2 * function_space.ufl_element().degree()
        representation_buffers = self._modify_quadrature_representation()
        quadrature_space, quadrature_trial_function, quadrature_test_function = (
            self._set_up_quadrature_space(quadrature_degree)
        )
        self._mass_matrix_cholesky_Factor = self._assemble_mass_matrix_cholesky_factor(
            quadrature_degree,
            quadrature_space,
            quadrature_trial_function,
            quadrature_test_function,
            test_function,
        )
        self._bilaplacian_precision_operator = hl.prior._BilaplacianR(  # noqa: SLF001
            self._matern_sdpe_matrix, self._mass_matrix_solver
        )
        self._bilaplacian_covariance_operator = hl.prior._BilaplacianRsolver(  # noqa: SLF001
            self._matern_spde_matrix_solver, self._mass_matrix
        )
        self._restore_quadrature_representation(representation_buffers)
        self._set_up_hippylib_interface()

    # ----------------------------------------------------------------------------------------------
    def _assemble_system_matrices(
        self,
        trial_function: ufl.Argument,
        test_function: ufl.Argument,
        variational_form_handler: Callable,
    ) -> tuple[dl.Matrix, dl.Matrix]:
        mass_matrix_term = ufl.inner(trial_function, test_function) * ufl.dx
        mass_matrix = dl.assemble(mass_matrix_term)
        matern_spde_term = variational_form_handler(trial_function, test_function)
        matern_spde_matrix = dl.assemble(matern_spde_term)
        return mass_matrix, matern_spde_matrix

    # ----------------------------------------------------------------------------------------------
    def _initialize_solvers(
        self, cg_solver_max_iter: int, cg_solver_relative_tolerance: Real
    ) -> tuple[dl.PETScKrylovSolver, dl.PETScKrylovSolver]:
        mass_matrix_solver = hl.algorithms.PETScKrylovSolver(
            self._function_space.mesh().mpi_comm(), "cg", "jacobi"
        )
        mass_matrix_solver.set_operator(self._mass_matrix)
        mass_matrix_solver.parameters["maximum_iterations"] = cg_solver_max_iter
        mass_matrix_solver.parameters["relative_tolerance"] = cg_solver_relative_tolerance
        mass_matrix_solver.parameters["error_on_nonconvergence"] = True
        mass_matrix_solver.parameters["nonzero_initial_guess"] = False

        matern_spde_matrix_solver = hl.algorithms.PETScKrylovSolver(
            self._function_space.mesh().mpi_comm(), "cg", hl.algorithms.amg_method()
        )
        matern_spde_matrix_solver.set_operator(self._matern_sdpe_matrix)
        matern_spde_matrix_solver.parameters["maximum_iterations"] = cg_solver_max_iter
        matern_spde_matrix_solver.parameters["relative_tolerance"] = cg_solver_relative_tolerance
        matern_spde_matrix_solver.parameters["error_on_nonconvergence"] = True
        matern_spde_matrix_solver.parameters["nonzero_initial_guess"] = False
        return mass_matrix_solver, matern_spde_matrix_solver

    # ----------------------------------------------------------------------------------------------
    def _modify_quadrature_representation(self) -> tuple[object, object]:
        quadrature_degree_buffer = dl.parameters["form_compiler"]["quadrature_degree"]
        representation_buffer = dl.parameters["form_compiler"]["representation"]
        dl.parameters["form_compiler"]["quadrature_degree"] = -1
        dl.parameters["form_compiler"]["representation"] = "quadrature"
        return quadrature_degree_buffer, representation_buffer

    # ----------------------------------------------------------------------------------------------
    def _restore_quadrature_representation(
        self, representation_buffers: tuple[object, object]
    ) -> None:
        quadrature_degree_buffer, representation_buffer = representation_buffers
        dl.parameters["form_compiler"]["quadrature_degree"] = quadrature_degree_buffer
        dl.parameters["form_compiler"]["representation"] = representation_buffer

    # ----------------------------------------------------------------------------------------------
    def _set_up_quadrature_space(
        self, quadrature_degree: int
    ) -> tuple[dl.FunctionSpace, ufl.Argument, ufl.Argument]:
        quadrature_element = ufl.VectorElement(
            "Quadrature",
            self._function_space.mesh().ufl_cell(),
            quadrature_degree,
            dim=self._function_space.num_sub_spaces(),
            quad_scheme="default",
        )
        quadrature_space = dl.FunctionSpace(self._function_space.mesh(), quadrature_element)
        quadrature_trial_function = dl.TrialFunction(quadrature_space)
        quadrature_test_function = dl.TestFunction(quadrature_space)
        return quadrature_space, quadrature_trial_function, quadrature_test_function

    # ----------------------------------------------------------------------------------------------
    def _assemble_mass_matrix_cholesky_factor(
        self,
        quadrature_degree: int,
        quadrature_space: dl.FunctionSpace,
        quadrature_trial_function: ufl.Argument,
        quadrature_test_function: ufl.Argument,
        test_function: ufl.Argument,
    ) -> dl.Matrix:
        quadrature_mass_matrix = dl.assemble(
            ufl.inner(quadrature_trial_function, quadrature_test_function)
            * ufl.dx(metadata={"quadrature_degree": quadrature_degree})
        )
        ones_constant = dl.Constant((1.0,) * self._function_space.num_sub_spaces())
        ones_vector = dl.interpolate(ones_constant, quadrature_space).vector()
        quadrature_mass_matrix_diagonal = quadrature_mass_matrix * ones_vector
        quadrature_mass_matrix.zero()
        quadrature_mass_matrix_diagonal.set_local(
            ones_vector.get_local() / np.sqrt(quadrature_mass_matrix_diagonal.get_local())
        )
        quadrature_mass_matrix.set_diagonal(quadrature_mass_matrix_diagonal)
        mixed_mass_matrix = dl.assemble(
            ufl.inner(quadrature_trial_function, test_function)
            * ufl.dx(metadata={"quadrature_degree": quadrature_degree})
        )
        mass_matrix_cholesky_factor = hl.algorithms.MatMatMult(
            mixed_mass_matrix, quadrature_mass_matrix
        )
        return mass_matrix_cholesky_factor

    # ----------------------------------------------------------------------------------------------
    def _set_up_hippylib_interface(self) -> None:
        self.mean = self._mean
        self.M = self._mass_matrix
        self.R = self._bilaplacian_precision_operator
        self.Rsolver = self._bilaplacian_covariance_operator

    # ----------------------------------------------------------------------------------------------
    def init_vector(self, vector_to_init: dl.Vector, matrix_dim: int | str) -> None:
        if matrix_dim == "noise":
            self._mass_matrix_cholesky_Factor.init_vector(vector_to_init, 1)
        else:
            self._matern_sdpe_matrix.init_vector(vector_to_init, matrix_dim)

    # ----------------------------------------------------------------------------------------------
    def sample(self, noise_vector: dl.Vector, matern_field_vector: dl.Vector, add_mean=True):
        rhs = self._mass_matrix_cholesky_Factor * noise_vector
        self._matern_spde_matrix_solver.solve(matern_field_vector, rhs)
        if add_mean:
            matern_field_vector.axpy(1.0, self.mean)


# ==================================================================================================
@dataclass
class PriorSettings:
    function_space: dl.FunctionSpace
    mean: Iterable[str]
    variance: Iterable[str]
    correlation_length: Iterable[str]
    anisotropy_tensor: Iterable[hl.ExpressionModule.AnisTensor2D] = None
    cg_solver_relative_tolerance: Annotated[float, Is[lambda x: 0 < x < 1]] = 1e-12
    cg_solver_max_iter: Annotated[int, Is[lambda x: x > 0]] = 1000
    robin_bc: bool = False
    robin_bc_const: Real = 1.42


@dataclass
class Prior:
    hippylib_prior: hl.prior._Prior
    mean_array: npt.NDArray[np.floating]
    variance_array: npt.NDArray[np.floating]
    correlation_length_array: npt.NDArray[np.floating]


# ==================================================================================================
class BilaplacianVectorPriorBuilder:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, prior_settings: PriorSettings):
        self._function_space = prior_settings.function_space
        self._num_components = self._function_space.num_sub_spaces()
        self._domain_dim = self._function_space.mesh().geometry().dim()
        self._mean = fex_converter.create_dolfin_function(prior_settings.mean, self._function_space)
        self._variance = fex_converter.create_dolfin_function(
            prior_settings.variance, self._function_space
        )
        self._correlation_length = fex_converter.create_dolfin_function(
            prior_settings.correlation_length, self._function_space
        )
        self._anisotropy_tensor = prior_settings.anisotropy_tensor
        self._cg_solver_relative_tolerance = prior_settings.cg_solver_relative_tolerance
        self._cg_solver_max_iter = prior_settings.cg_solver_max_iter
        self._robin_bc = prior_settings.robin_bc
        self._robin_bc_const = prior_settings.robin_bc_const
        self._gamma = None
        self._delta = None

    # ----------------------------------------------------------------------------------------------
    def build(self):
        self._gamma, self._delta = self._convert_prior_coefficients()

        def variational_form_handler(
            trial_function: ufl.Argument, test_function: ufl.Argument
        ) -> ufl.Form:
            component_forms = []
            for i in range(self._num_components):
                scalar_form = self._generate_scalar_form(trial_function, test_function, i)
                component_forms.append(scalar_form)
            vector_form = sum(component_forms)
            return vector_form

        prior_object = SqrtPrecisionPDEPrior(
            self._function_space,
            variational_form_handler,
            self._mean,
            self._cg_solver_relative_tolerance,
            self._cg_solver_max_iter,
        )
        mean_array = fex_converter.convert_to_numpy(self._mean.vector(), self._function_space)
        variance_array = fex_converter.convert_to_numpy(
            self._variance.vector(), self._function_space
        )
        correlation_length_array = fex_converter.convert_to_numpy(
            self._correlation_length.vector(), self._function_space
        )
        prior = Prior(
            hippylib_prior=prior_object,
            mean_array=mean_array,
            variance_array=variance_array,
            correlation_length_array=correlation_length_array,
        )
        return prior

    # ----------------------------------------------------------------------------------------------
    def _convert_prior_coefficients(self) -> tuple[dl.Function, dl.Function]:
        variance_array = self._variance.vector().get_local()
        correlation_length_array = self._correlation_length.vector().get_local()

        sobolev_exponent = 2 - 0.5 * self._domain_dim
        kappa_array = np.sqrt(8 * sobolev_exponent) / correlation_length_array
        gamma_array = 1 / (
            np.sqrt(variance_array)
            * np.power(kappa_array, sobolev_exponent)
            * np.sqrt(np.power(4 * np.pi, 0.5 * self._domain_dim) / math.gamma(sobolev_exponent))
        )
        delta_array = np.power(kappa_array, 2) * gamma_array
        gamma = fex_converter.convert_to_dolfin(gamma_array, self._function_space)
        delta = fex_converter.convert_to_dolfin(delta_array, self._function_space)
        return gamma, delta

    # ----------------------------------------------------------------------------------------------
    def _generate_scalar_form(
        self, trial_function: ufl.Argument, test_function: ufl.Argument, index: int
    ) -> ufl.Form:
        trial_component = trial_function[index]
        test_component = test_function[index]
        gamma_component = self._gamma[index]
        delta_component = self._delta[index]

        mass_matrix_term = delta_component * ufl.inner(trial_component, test_component) * ufl.dx
        if self._anisotropy_tensor is None:
            stiffness_matrix_term = (
                gamma_component
                * ufl.inner(ufl.grad(trial_component), ufl.grad(test_component))
                * ufl.dx
            )
        else:
            stiffness_matrix_term = (
                stiffness_matrix_term
                * ufl.inner(
                    self._anisotropy_tensor * ufl.grad(trial_component), ufl.grad(test_component)
                )
                * ufl.dx
            )

        if self._robin_bc:
            robin_coeff = (
                gamma_component
                * ufl.sqrt(delta_component / gamma_component)
                / dl.Constant(self._robin_bc_const)
            )
        else:
            robin_coeff = dl.Constant(0.0)
        boundary_term = robin_coeff * ufl.inner(trial_component, test_component) * ufl.ds

        scalar_form = mass_matrix_term + stiffness_matrix_term + boundary_term
        return scalar_form
