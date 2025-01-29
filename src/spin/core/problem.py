import functools
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import dolfin as dl
import ufl
from beartype.vale import Is

import spin.fenics.converter as fex_converter
import spin.hippylib as hlx
from spin.core import weakforms

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SyntaxWarning)
    import hippylib as hl


# ==================================================================================================
@dataclass
class SPINProblemSettings:
    mesh: dl.Mesh
    pde_type: Annotated[
        str, Is[lambda pde: pde in ["mean_exit_time", "mean_exit_time_moments", "fokker_planck"]]
    ]
    inference_type: Annotated[
        str,
        Is[lambda inference: inference in ["drift_only", "diffusion_only", "drift_and_diffusion"]],
    ]
    element_family_variables: Annotated[
        str, Is[lambda family: family in ufl.finiteelement.elementlist.ufl_elements]
    ] = "Lagrange"
    element_family_parameters: Annotated[
        str, Is[lambda family: family in ufl.finiteelement.elementlist.ufl_elements]
    ] = "Lagrange"
    element_degree_variables: Annotated[int, Is[lambda x: x > 0]] = 1
    element_degree_parameters: Annotated[int, Is[lambda x: x > 0]] = 1
    drift: str | Iterable[str] | None = None
    squared_diffusion: str | Iterable[Iterable[str]] | None = None
    start_time: Real | None = None
    end_time: Real | None = None
    num_steps: int | None = None
    initial_condition: str | Iterable[str] | None = None


# ==================================================================================================
class SPINProblemBuilder:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: SPINProblemSettings) -> None:
        self._mesh = settings.mesh
        self._pde_type = settings.pde_type
        self._inference_type = settings.inference_type
        self._element_family_variables = settings.element_family_variables
        self._element_family_parameters = settings.element_family_parameters
        self._element_degree_variables = settings.element_degree_variables
        self._element_degree_parameters = settings.element_degree_parameters
        self._drift = settings.drift
        self._squared_diffusion = settings.squared_diffusion
        self._start_time = settings.start_time
        self._end_time = settings.end_time
        self._num_steps = settings.num_steps
        self._initial_condition = settings.initial_condition

        self._domain_dim = self._mesh.geometry().dim()
        self._num_components = 2 if self._pde_type == "mean_exit_time_moments" else 1

        self._function_space_variables = None
        self._function_space_parameters = None
        self._function_space_drift = None
        self._function_space_diffusion = None
        self._function_space_composite = None
        self._boundary_condition = None
        self._weak_form = None
        self._weak_form_wrapper = None

    # ----------------------------------------------------------------------------------------------
    def build(self) -> hl.PDEProblem:
        (
            self._function_space_variables,
            self._function_space_drift,
            self._function_space_diffusion,
            self._function_space_composite,
        ) = self._create_function_spaces()
        self._function_space_parameters = self._assign_parameter_function_space()
        self._weak_form = self._assign_weak_form()
        self._boundary_condition = self._create_boundary_condition()
        self._weak_form_wrapper = self._create_weak_form_wrapper()
        spin_problem = self._create_variational_problem()
        return spin_problem

    # ----------------------------------------------------------------------------------------------
    def _create_function_spaces(
        self,
    ) -> tuple[dl.FunctionSpace, dl.FunctionSpace, dl.FunctionSpace, dl.FunctionSpace]:
        if self._num_components ==1:
            VariableElement = dl.FiniteElement  # noqa: N806
        else:
            VariableElement = functools.partial(dl.VectorElement, dim=self._num_components)  # noqa: N806

        elem_variables = VariableElement(
            family=self._element_family_variables,
            cell=self._mesh.ufl_cell(),
            degree=self._element_degree_variables,
        )
        elem_drift = dl.VectorElement(
            family=self._element_family_parameters,
            cell=self._mesh.ufl_cell(),
            degree=self._element_degree_parameters,
            dim=self._domain_dim,
        )
        elem_diffusion = dl.TensorElement(
            family=self._element_family_parameters,
            cell=self._mesh.ufl_cell(),
            degree=self._element_degree_parameters,
            shape=(self._domain_dim, self._domain_dim),
            symmetry=True,
        )
        elem_composite = dl.MixedElement([elem_drift, elem_diffusion])
        function_space_variables = dl.FunctionSpace(self._mesh, elem_variables)
        function_space_drift = dl.FunctionSpace(self._mesh, elem_drift)
        function_space_diffusion = dl.FunctionSpace(self._mesh, elem_diffusion)
        function_space_composite = dl.FunctionSpace(self._mesh, elem_composite)

        return (
            function_space_variables,
            function_space_drift,
            function_space_diffusion,
            function_space_composite,
        )

    # ----------------------------------------------------------------------------------------------
    def _assign_parameter_function_space(self) -> dl.FunctionSpace:
        if self._inference_type == "drift_only":
            parameter_function_space = self._function_space_drift
        elif self._inference_type == "diffusion_only":
            parameter_function_space = self._function_space_diffusion
        elif self._inference_type == "drift_and_diffusion":
            parameter_function_space = self._function_space_composite
        return parameter_function_space

    # ----------------------------------------------------------------------------------------------
    def _assign_weak_form(
        self,
    ) -> Callable[[dl.Function, dl.Function, dl.Function, dl.Function], dl.Form]:
        if self._pde_type == "mean_exit_time":
            weak_form = weakforms.weak_form_mean_exit_time
        elif self._pde_type == "mean_exit_time_moments":
            weak_form = weakforms.weak_form_mean_exit_time_moments
        elif self._pde_type == "fokker_planck":
            weak_form = weakforms.weak_form_fokker_planck
        return weak_form

    # ----------------------------------------------------------------------------------------------
    def _create_boundary_condition(self) -> dl.DirichletBC:
        bc_value = 0.0 if self._num_components == 1 else (0.0,) * self._num_components
        boundary_condition = dl.DirichletBC(
            self._function_space_variables,
            dl.Constant(bc_value),
            lambda _, on_boundary: on_boundary,
        )
        return boundary_condition

    # ----------------------------------------------------------------------------------------------
    def _create_weak_form_wrapper(
        self,
    ) -> Callable[[dl.Function, dl.Function, dl.Function], dl.Form]:
        if self._inference_type == "drift_only":
            if self._squared_diffusion is None:
                raise ValueError("Diffusion function is required for drift only inference.")
            diffusion_function = fex_converter.create_dolfin_function(
                self._squared_diffusion, self._function_space_diffusion
            )
            weak_form_wrapper = (  # noqa: E731
                lambda forward_variable, parameter_variable, adjoint_variable: self._weak_form(
                    forward_variable, adjoint_variable, parameter_variable, diffusion_function
                )
            )
        elif self._inference_type == "diffusion_only":
            if self._drift is None:
                raise ValueError("Drift function is required for diffusion only inference.")
            drift_function = fex_converter.create_dolfin_function(
                self._drift, self._function_space_drift
            )
            weak_form_wrapper = (  # noqa: E731
                lambda forward_variable, parameter_variable, adjoint_variable: self._weak_form(
                    forward_variable, adjoint_variable, drift_function, parameter_variable
                )
            )
        elif self._inference_type == "drift_and_diffusion":
            weak_form_wrapper = (  # noqa: E731
                lambda forward_variable, parameter_variable, adjoint_variable: self._weak_form(
                    forward_variable, adjoint_variable, parameter_variable[0], parameter_variable[1]
                )
            )
        return weak_form_wrapper

    # ----------------------------------------------------------------------------------------------
    def _create_variational_problem(self) -> hl.PDEProblem:
        function_space_list = (
            self._function_space_variables,
            self._function_space_parameters,
            self._function_space_variables,
        )
        if self._pde_type == "fokker_planck":
            if self._start_time is None or self._end_time is None or self._num_steps is None:
                raise ValueError("Time parameters are required for Fokker-Planck PDE inference.")
            if initial_condition is None:
                raise ValueError("Initial condition is required for Fokker-Planck PDE inference.")
            initial_condition = fex_converter.create_dolfin_function(
                self._initial_condition, self._function_space_variables
            )
            spin_problem = hlx.TDPDELinearVariationalProblem(
                function_space_list,
                self._weak_form_wrapper,
                self._boundary_condition,
                self._boundary_condition,
                initial_condition,
                self._start_time,
                self._end_time,
                self._num_steps,
            )
        else:
            spin_problem = hl.PDEVariationalProblem(
                function_space_list,
                self._weak_form_wrapper,
                self._boundary_condition,
                self._boundary_condition,
                is_fwd_linear=True,
            )
        return spin_problem
