import functools
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, Any, Final

import dolfin as dl
import numpy as np
import numpy.typing as npt
import ufl
import ufl.argument
import ufl.tensors
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
    log_squared_diffusion: str | Iterable[str] | None = None
    start_time: Real | None = None
    end_time: Real | None = None
    num_steps: int | None = None
    initial_condition: str | Iterable[str] | None = None


# --------------------------------------------------------------------------------------------------
@dataclass
class PDEType:
    weak_form: Callable[
        [ufl.Argument, ufl.Argument, ufl.Coefficient, ufl.tensors.ListTensor], ufl.Form
    ]
    num_components: Annotated[int, Is[lambda x: x > 0]]
    stationary: bool


# ==================================================================================================
@dataclass
class SPINProblem:
    hippylib_variational_problem: hl.PDEProblem
    num_variable_components: Annotated[int, Is[lambda x: x > 0]]
    domain_dim: Annotated[int, Is[lambda x: x > 0]]
    function_space_variables: dl.FunctionSpace
    function_space_parameters: dl.FunctionSpace
    function_space_drift: dl.FunctionSpace
    function_space_diffusion: dl.FunctionSpace
    coordinates_variables: npt.NDArray[np.floating]
    coordinates_parameters: npt.NDArray[np.floating]
    drift_array: npt.NDArray[np.floating] | None = None
    log_squared_diffusion_array: npt.NDArray[np.floating] | None = None
    initial_condition_array: npt.NDArray[np.floating] | None = None

    # ----------------------------------------------------------------------------------------------
    def solve_forward(self, parameter_array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        parameter_vector = fex_converter.convert_to_dolfin(
            parameter_array, self.function_space_parameters
        ).vector()
        solution_vector = self.hippylib_variational_problem.generate_state()
        self.hippylib_variational_problem.solveFwd(solution_vector, [None, parameter_vector, None])
        solution_array = fex_converter.convert_to_numpy(
            solution_vector, self.function_space_variables
        )
        return solution_array

    # ----------------------------------------------------------------------------------------------
    def solve_adjoint(
        self,
        forward_array: npt.NDArray[np.floating],
        parameter_array: npt.NDArray[np.floating],
        right_hand_side_array: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        forward_vector = fex_converter.convert_to_dolfin(
            forward_array, self.function_space_variables
        ).vector()
        parameter_vector = fex_converter.convert_to_dolfin(
            parameter_array, self.function_space_parameters
        ).vector()
        right_hand_side_vector = fex_converter.convert_to_dolfin(
            right_hand_side_array, self.function_space_variables
        ).vector()
        adjoint_vector = self.hippylib_variational_problem.generate_state()
        self.hippylib_variational_problem.solveAdj(
            adjoint_vector, [forward_vector, parameter_vector, None], right_hand_side_vector
        )
        adjoint_array = fex_converter.convert_to_numpy(
            adjoint_vector, self.function_space_variables
        )
        return adjoint_array


# ==================================================================================================
class SPINProblemBuilder:
    _registered_pde_types: Final[dict[str, PDEType]] = {
        "mean_exit_time": PDEType(
            weak_form=weakforms.weak_form_mean_exit_time,
            num_components=1,
            stationary=True,
        ),
        "mean_exit_time_moments": PDEType(
            weak_form=weakforms.weak_form_mean_exit_time_moments,
            num_components=2,
            stationary=True,
        ),
        "fokker_planck": PDEType(
            weak_form=weakforms.weak_form_fokker_planck,
            num_components=1,
            stationary=False,
        ),
    }

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: SPINProblemSettings) -> None:
        try:
            self._pde_type = self._registered_pde_types[settings.pde_type]
        except KeyError:
            raise ValueError(
                f"Unknown PDE type: {settings.pde_type}, "
                f"must be one of {self._registered_pde_types.keys()}"
            ) from KeyError
        self._mesh = settings.mesh
        self._inference_type = settings.inference_type
        self._element_family_variables = settings.element_family_variables
        self._element_family_parameters = settings.element_family_parameters
        self._element_degree_variables = settings.element_degree_variables
        self._element_degree_parameters = settings.element_degree_parameters
        self._drift = settings.drift
        self._log_squared_diffusion = settings.log_squared_diffusion
        self._start_time = settings.start_time
        self._end_time = settings.end_time
        self._num_steps = settings.num_steps
        self._initial_condition = settings.initial_condition
        self._domain_dim = self._mesh.geometry().dim()

        self._function_space_variables = None
        self._function_space_drift = None
        self._function_space_diffusion = None
        self._function_space_parameters = None
        self._function_space_composite = None
        self._drift_function = None
        self._log_squared_diffusion_function = None
        self._initial_condition_function = None
        self._boundary_condition = None
        self._weak_form_wrapper = None

    # ----------------------------------------------------------------------------------------------
    def build(self) -> SPINProblem:
        (
            self._function_space_variables,
            self._function_space_drift,
            self._function_space_diffusion,
            self._function_space_composite,
        ) = self._create_function_spaces()
        (
            self._drift_function,
            self._log_squared_diffusion_function,
            self._initial_condition_function,
        ) = self._compile_expressions()
        drift_array, log_squared_diffusion_array, initial_condition_array = (
            self._get_parameter_arrays()
        )
        self._function_space_parameters = self._assign_parameter_function_space()
        self._boundary_condition = self._create_boundary_condition()
        self._weak_form_wrapper = self._create_weak_form_wrapper()
        variational_problem = self._create_variational_problem()

        coordinates_variables = fex_converter.get_coordinates(self._function_space_variables)
        coordinates_parameters = fex_converter.get_coordinates(self._function_space_drift)

        spin_problem = SPINProblem(
            hippylib_variational_problem=variational_problem,
            num_variable_components=self._pde_type.num_components,
            domain_dim=self._domain_dim,
            function_space_variables=self._function_space_variables,
            function_space_parameters=self._function_space_parameters,
            function_space_drift=self._function_space_drift,
            function_space_diffusion=self._function_space_diffusion,
            coordinates_variables=coordinates_variables,
            coordinates_parameters=coordinates_parameters,
            drift_array=drift_array,
            log_squared_diffusion_array=log_squared_diffusion_array,
            initial_condition_array=initial_condition_array,
        )
        return spin_problem

    # ----------------------------------------------------------------------------------------------
    def _create_function_spaces(
        self,
    ) -> tuple[dl.FunctionSpace, dl.FunctionSpace, dl.FunctionSpace, dl.FunctionSpace]:
        if self._pde_type.num_components == 1:
            VariableElement = dl.FiniteElement  # noqa: N806
        else:
            VariableElement = functools.partial(dl.VectorElement, dim=self._pde_type.num_components)  # noqa: N806

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
        elem_diffusion = dl.VectorElement(
            family=self._element_family_parameters,
            cell=self._mesh.ufl_cell(),
            degree=self._element_degree_parameters,
            dim=self._domain_dim,
        )
        elem_composite = dl.VectorElement(
            family=self._element_family_parameters,
            cell=self._mesh.ufl_cell(),
            degree=self._element_degree_parameters,
            dim=2 * self._domain_dim,
        )
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
    def _compile_expressions(
        self,
    ) -> tuple[dl.Function | None, dl.Function | None, dl.Function | None]:
        if self._drift is not None:
            drift_function = fex_converter.create_dolfin_function(
                self._drift, self._function_space_drift
            )
        else:
            drift_function = None
        if self._log_squared_diffusion is not None:
            log_squared_diffusion_function = fex_converter.create_dolfin_function(
                self._log_squared_diffusion, self._function_space_diffusion
            )
        else:
            log_squared_diffusion_function = None
        if self._initial_condition is not None:
            initial_condition_function = fex_converter.create_dolfin_function(
                self._initial_condition, self._function_space_variables
            )
        else:
            initial_condition_function = None
        return drift_function, log_squared_diffusion_function, initial_condition_function

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
    def _get_parameter_arrays(
        self,
    ) -> tuple[
        npt.NDArray[np.floating] | None,
        npt.NDArray[np.floating] | None,
        tuple[npt.NDArray[np.floating]] | None,
    ]:
        if self._drift_function is not None:
            drift_array = fex_converter.convert_to_numpy(
                self._drift_function.vector(), self._function_space_drift
            )
        else:
            drift_array = None
        if self._log_squared_diffusion_function is not None:
            log_squared_diffusion_array = fex_converter.convert_to_numpy(
                self._log_squared_diffusion_function.vector(), self._function_space_diffusion
            )
        else:
            log_squared_diffusion_array = None
        if self._initial_condition_function is not None:
            initial_condition_array = fex_converter.convert_to_numpy(
                self._initial_condition_function.vector(), self._function_space_variables
            )
        else:
            initial_condition_array = None
        return drift_array, log_squared_diffusion_array, initial_condition_array

    # ----------------------------------------------------------------------------------------------
    def _create_boundary_condition(self) -> dl.DirichletBC:
        bc_value = (
            0.0 if self._pde_type.num_components == 1 else (0.0,) * self._pde_type.num_components
        )
        boundary_condition = dl.DirichletBC(
            self._function_space_variables,
            dl.Constant(bc_value),
            lambda _, on_boundary: on_boundary,
        )
        return boundary_condition

    # ----------------------------------------------------------------------------------------------
    def _create_weak_form_wrapper(
        self,
    ) -> Callable[[Any, Any, Any], ufl.Form]:
        if self._inference_type == "drift_only":
            if self._log_squared_diffusion_function is None:
                raise ValueError("Diffusion function is required for drift only inference.")

            def weak_form_wrapper(
                forward_variable: ufl.Argument | dl.Function,
                parameter_variable: ufl.Coefficient | dl.Function,
                adjoint_variable: ufl.Argument | dl.Function,
            ) -> ufl.Form:
                return self._pde_type.weak_form(
                    forward_variable,
                    adjoint_variable,
                    parameter_variable,
                    self._compute_matrix_exponential(self._log_squared_diffusion_function),
                )
        elif self._inference_type == "diffusion_only":
            if self._drift is None:
                raise ValueError("Drift function is required for diffusion only inference.")

            def weak_form_wrapper(
                forward_variable: dl.Argument,
                parameter_variable: dl.Coefficient,
                adjoint_variable: dl.Argument,
            ) -> ufl.Form:
                return self._pde_type.weak_form(
                    forward_variable,
                    adjoint_variable,
                    self._drift_function,
                    self._compute_matrix_exponential(parameter_variable),
                )
        elif self._inference_type == "drift_and_diffusion":

            def weak_form_wrapper(
                forward_variable: ufl.Argument | dl.Function,
                parameter_variable: ufl.Coefficient | dl.Function,
                adjoint_variable: ufl.Argument | dl.Function,
            ) -> ufl.Form:
                drift_variable = [parameter_variable[i] for i in range(self._domain_dim)]
                log_squared_diffusion_variable = [
                    parameter_variable[i] for i in range(self._domain_dim, 2 * self._domain_dim)
                ]
                drift_variable = ufl.as_vector(drift_variable)
                log_squared_diffusion_variable = ufl.as_vector(log_squared_diffusion_variable)
                return self._pde_type.weak_form(
                    forward_variable,
                    adjoint_variable,
                    drift_variable,
                    self._compute_matrix_exponential(log_squared_diffusion_variable),
                )

        return weak_form_wrapper

    # ----------------------------------------------------------------------------------------------
    def _compute_matrix_exponential(
        self, matrix_diagonal: dl.Function | ufl.tensors.ListTensor
    ) -> ufl.tensors.ListTensor:
        diagonal_components = [ufl.exp(component) for component in matrix_diagonal]
        diagonal_components = ufl.as_vector(diagonal_components)
        matrix_exponential = ufl.diag(diagonal_components)
        return matrix_exponential

    # ----------------------------------------------------------------------------------------------
    def _create_variational_problem(self) -> hl.PDEProblem:
        function_space_list = (
            self._function_space_variables,
            self._function_space_parameters,
            self._function_space_variables,
        )
        if self._pde_type.stationary:
            spin_problem = hl.PDEVariationalProblem(
                function_space_list,
                self._weak_form_wrapper,
                self._boundary_condition,
                self._boundary_condition,
                is_fwd_linear=True,
            )
        else:
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
        return spin_problem
