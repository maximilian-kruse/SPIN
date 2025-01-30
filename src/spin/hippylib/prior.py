import math
from collections.abc import Iterable
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

        def variational_form_handler(trial_function: ufl.Argument, test_function: ufl.Argument):
            component_forms = []
            for i in range(self._num_components):
                scalar_form = self._generate_scalar_form(trial_function, test_function, i)
                component_forms.append(scalar_form)
            vector_form = sum(component_forms)
            return vector_form

        prior_object = hl.SqrtPrecisionPDE_Prior(
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
