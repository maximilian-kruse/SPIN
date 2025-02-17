"""Setup of the PDE variational problem for stochastic process inference.

This module implements the functionality of SPIN that is specific to stochastic processes. It sets
up the Kolmogorov equations for PDE-based inference with Hippylib, as defined in the
[`weakforms`][spin.core.weakforms] module. The builder pattern is employed to return a
Hippylib-conformant object with additional data and methods for convenience. Different inference
modes are available to infer drift, diffusion, or both.

!!! info
    We do not consider the actual diffusion matrix, but the logarithm of its square. Inferring the
    square avoids disambiguity issues, whereas infereing the log enforces positivity of the
    diffusion matrix. SPIN can only infer diagonal diffusion matrices, meaning that enforcing
    positivity of the diagonal entries ensures that the diffusion matrix is s.p.d.

Classes:
    SPINProblemSettings: Configuration and data for SPIN problem setup.
    PDEType: Registration of PDEs, including weak form and metadata.
    SPINProblem: Wrapper for Hippylib PDE problem with additional data and functionality.
    SPINProblemBuilder: Builder for the SPINProblem object.
"""

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
    """Configuration and data for SPIN problem setup.

    Attributes:
        mesh (dl.Mesh): Dolfin mesh as discretization of the PDE domain.
        pde_type (str): Identifier of the PDE to use, needs to be registered in the
            `_registered_pde_types` dictionary of the `SPINProblemBuilder` class. Available options
            are "mean_exit_time", "mean_exit_time_moments", and "fokker_planck".
        inference_type (str): Type of inference, meaning which parameter(s) to infer. Available
            options are "drift_only", "diffusion_only", and "drift_and_diffusion".
        element_family_variables (str): FE Family for the forward and adjoint variable, according
            to options in UFL.
        element_family_parameters (str): FE Family for the parameter variable(s), according
            to options in UFL.
        element_degree_variables (int): FE degree for the forward and adjoint variable, according
            to options in UFL.
        element_degree_parameters (int): FE degree for the parameter variable(s), according
            to options in UFL.
        drift (str | Iterable[str] | None): String in dolfin syntax defining drift vector. Needs to
            be provided as list of length corresponding to problem dimension. Only required for
            "diffusion_only" inference mode.
        log_squared_diffusion (str | Iterable[str] | None): String in dolfin syntax defining
            diagonal of the log squared diffusion function. Needs to be provided as list of length
            corresponding to problem dimension. Only required for "drift_only" inference mode.
        start_time (Real | None): Start time for PDE solver, only required for time-dependent PDE.
        end_time (Real | None): End time for PDE solver, only required for time-dependent PDE.
        num_steps (int | None): Number of time steps for PDE solver, only required for
            time-dependent PDE.
        initial_condition (str | Iterable[str] | None): Initial condition for PDE solver, only
            required for time-dependent PDE..
    """

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
    """Registration of PDEs, including weak form and metadata.

    This class is used by the builder internally, and does not require interaction by the user.
    Only for development purposes, when a new PDE is implemented.

    Attributes:
        weak_form (Callable): Weak form in UFL syntax, defined in the
            [`weakforms`][spin.core.weakforms] module.
        num_components (int): Number of components of the solution/adjoint variable.
        linear (bool): If the PDE is linear.
        stationary (bool): If the PDE is stationary.
    """

    weak_form: Callable[
        [ufl.Argument, ufl.Argument, ufl.Coefficient, ufl.tensors.ListTensor], ufl.Form
    ]
    num_components: Annotated[int, Is[lambda x: x > 0]]
    linear: bool
    stationary: bool


# ==================================================================================================
@dataclass
class SPINProblem:
    """Wrapper for Hippylib PDE problem with additional data and functionaliry.

    A `SPINProblem` object is returned as the output of the builder. It wraps a Hippylib PDE problem
    to conduct inference with. It further provides data like function spaces, coordinates, etc. for
    more transparency. Moreoever, it implements methods for forward, adjoint and gradient solves
    with a Numpy interface.

    Attributes:
        hippylib_variational_problem (hl.PDEProblem): Hippylib PDE problem object.
        num_variable_components (int): Number of components in forard/adjoint variable.
        domain_dim (int): Dimension of the computational domain.
        function_space_variables (dl.FunctionSpace): Function space for forward/adjoint variable.
        function_space_parameters (dl.FunctionSpace): Function space for parameter.
        function_space_drift (dl.FunctionSpace): Function space for the drift vector.
        function_space_diffusion (dl.FunctionSpace): Function space for the diffusion matrix.
        coordinates_variables (npt.NDArray[np.floating]): Coordinates for the forward/adjoint
            variable degrees of freedom.
        coordinates_parameters (npt.NDArray[np.floating]): Coordinates for the parameter degrees of
            freedom.
        drift_array (npt.NDArray[np.floating] | None): Drift function converted to numpy array, if
            provided by the user.
        log_squared_diffusion_array (npt.NDArray[np.floating] | None): Log squared diffusion
            function converted to numpy array, if provided by the user.
        initial_condition_array (npt.NDArray[np.floating] | None): Initial condition converted to
            numpy array, if provided by the user.

    Methods:
        solve_forward: Solve the PDE, given a parameter.
        solve_adjoint: Solve the adjoint equation, given parameter and forward solution.
        evaluate_gradient: Evaluate parametric gradient, given parameter, forward, and adjoint
            solution.
    """

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
        r"""Solve the defined PDE, given a parameter function.

        The parameter can be drift, difusion, or both, depending on the inference mode. It has to be
        provided according to the convention defined in the fenics
        [`converter`][spin.fenics.converter] module. This means that the array has shape
        $K\times N$ for $k$ components and $N$ degrees of freedom on the computational domain.

        Args:
            parameter_array (npt.NDArray[np.floating]): Parameter function to solve PDE for.

        Raises:
            ValueError: Checks the parameter array has correct size.

        Returns:
            npt.NDArray[np.floating]: Forward solution of the PDE.
        """
        if not parameter_array.size == self.function_space_parameters.dim():
            raise ValueError(
                f"Parameter array has wrong size {parameter_array.size}, "
                f"expected {self.function_space_parameters.dim()}"
            )
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
        r"""Solve the adjoint equation for the define PDE, given parameter and forward solution.

        The parameter can be drift, difusion, or both, depending on the inference mode. It has to be
        provided according to the convention defined in the fenics
        [`converter`][spin.fenics.converter] module. This means that the array has shape
        $K\times N$ for $k$ components and $N$ degrees of freedom on the computational domain.
        The forward solution can be obtained by calling the `solve_forward` method of this class.
        Latly, the adjoint equation is solved with a given right hand side, which is typically
        provided as the gradient of some loss functional governed by the PDE model

        Args:
            forward_array (npt.NDArray[np.floating]): Forward solution of the PDE.
            parameter_array (npt.NDArray[np.floating]): Parameter function.
            right_hand_side_array (npt.NDArray[np.floating]): Right-hand-side for the adjoint.

        Raises:
            ValueError: Checks that the array sizes of forward, parameter, and RHS are correct.

        Returns:
            npt.NDArray[np.floating]: Adjoint solution of the PDE problem.
        """
        if not forward_array.size == self.function_space_variables.dim():
            raise ValueError(
                f"Forward array has wrong size {forward_array.size}, "
                f"expected {self.function_space_variables.dim()}"
            )
        if not parameter_array.size == self.function_space_parameters.dim():
            raise ValueError(
                f"Parameter array has wrong size {parameter_array.size}, "
                f"expected {self.function_space_parameters.dim()}"
            )
        if not right_hand_side_array.size == self.function_space_variables.dim():
            raise ValueError(
                f"Right-hand-side array has wrong size {right_hand_side_array.size}, "
                f"expected {self.function_space_variables.dim()}"
            )
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

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self,
        forward_array: npt.NDArray[np.floating],
        parameter_array: npt.NDArray[np.floating],
        adjoint_array: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        r"""Evaluate the parametric gradient for parameter, forward, and adjoint solution.

        The parameter can be drift, difusion, or both, depending on the inference mode. It has to be
        provided according to the convention defined in the fenics
        [`converter`][spin.fenics.converter] module. This means that the array has shape
        $K\times N$ for $k$ components and $N$ degrees of freedom on the computational domain.
        The forward solution can be obtained by calling the `solve_forward` method of this class,
        the adjoint solution by calling the `solve_adjoint` method.

        Args:
            forward_array (npt.NDArray[np.floating]): Forward solution.
            parameter_array (npt.NDArray[np.floating]): Parameter function.
            adjoint_array (npt.NDArray[np.floating]): Adjoint solution.

        Raises:
            ValueError: Checks that the array sizes of forward, parameter, and adjoint are correct.

        Returns:
            npt.NDArray[np.floating]: Parametric gradient.
        """
        if not forward_array.size == self.function_space_variables.dim():
            raise ValueError(
                f"Forward array has wrong size {forward_array.size}, "
                f"expected {self.function_space_variables.dim()}"
            )
        if not parameter_array.size == self.function_space_parameters.dim():
            raise ValueError(
                f"Parameter array has wrong size {parameter_array.size}, "
                f"expected {self.function_space_parameters.dim()}"
            )
        if not adjoint_array.size == self.function_space_variables.dim():
            raise ValueError(
                f"Adjoint array has wrong size {adjoint_array.size}, "
                f"expected {self.function_space_variables.dim()}"
            )
        forward_vector = fex_converter.convert_to_dolfin(
            forward_array, self.function_space_variables
        ).vector()
        parameter_vector = fex_converter.convert_to_dolfin(
            parameter_array, self.function_space_parameters
        ).vector()
        adjoint_vector = fex_converter.convert_to_dolfin(
            adjoint_array, self.function_space_variables
        ).vector()
        gradient_vector = self.hippylib_variational_problem.generate_parameter()
        self.hippylib_variational_problem.evalGradientParameter(
            [forward_vector, parameter_vector, adjoint_vector], gradient_vector
        )
        gradient_array = fex_converter.convert_to_numpy(
            gradient_vector, self.function_space_parameters
        )
        return gradient_array


# ==================================================================================================
class SPINProblemBuilder:
    """_summary_."""

    # Registered PDE types with weak form and metadata
    # Add newly implemented forms here
    _registered_pde_types: Final[dict[str, PDEType]] = {
        "mean_exit_time": PDEType(
            weak_form=weakforms.weak_form_mean_exit_time,
            num_components=1,
            linear=True,
            stationary=True,
        ),
        "mean_exit_time_moments": PDEType(
            weak_form=weakforms.weak_form_mean_exit_time_moments,
            num_components=2,
            linear=True,
            stationary=True,
        ),
        "fokker_planck": PDEType(
            weak_form=weakforms.weak_form_fokker_planck,
            num_components=1,
            linear=True,
            stationary=False,
        ),
    }

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: SPINProblemSettings) -> None:
        """Constructor, set all data structures internally for usage in `build` method.

        Args:
            settings (SPINProblemSettings): Configuration and data for the PDE variational problem.

        Raises:
            ValueError: Checks that given PDE type is registered with the builder in
                `_registered_pde_types`.
        """
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
        """Main interface of the builder, returning a SPINProblem object.

        The builder internally cals a sequence of methods that result in a Hippylib `PDEProblem`
        object to be used for inference. The methods are implemented in a semi-explicit manner:
        Function arguments are implicit, as they are set as class attributes in the constructor.
        Output of the methods is explicit however. This is a compromise between clarity and
        verbosity of the OOP design in this class.

        Returns:
            SPINProblem: Object wrapping the Hippylib `PDEProblem` with additional methods and
                metadata.
        """
        # Set up function spaces
        (
            self._function_space_variables,
            self._function_space_drift,
            self._function_space_diffusion,
            self._function_space_composite,
        ) = self._create_function_spaces()

        # Compile available dolfin expressions
        (
            self._drift_function,
            self._log_squared_diffusion_function,
            self._initial_condition_function,
        ) = self._compile_expressions()

        # Convert given dolfin functions to arrays
        drift_array, log_squared_diffusion_array, initial_condition_array = (
            self._get_parameter_arrays()
        )

        # Get mesh coordinates
        coordinates_variables = fex_converter.get_coordinates(self._function_space_variables)
        coordinates_parameters = fex_converter.get_coordinates(self._function_space_drift)

        # Assemble weak form and boundary condition, depending on PDE type and inference mode
        self._function_space_parameters = self._assign_parameter_function_space()
        self._boundary_condition = self._create_boundary_condition()
        self._weak_form_wrapper = self._create_weak_form_wrapper()

        # Create a hippylip PDEProblem object
        variational_problem = self._create_variational_problem()

        # Return output as SPINProblem object
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
        """Create function spaces for variables, drift, diffusion, and composite parameters.

        The precise form of the function spaces depends on the PDE typ and inference mode.
        For scalar PDEs, the solution and adjoint variable space is scalae, otherwise it is vector-
        valued. The drift and diffusion spaces are always vector-valued, while the composite space
        is a  vector space comprising both the drift and diffusion components.

        Returns:
            tuple[dl.FunctionSpace, dl.FunctionSpace, dl.FunctionSpace, dl.FunctionSpace]:
                Tuple of function spaces for variables, drift, diffusion, and composite parameters.
        """
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
        """Generate dolfin expressions from strings, if they are provided.

        Depending on the inference mode, different expressions need to be provided. Their
        existence is checked upon assemble of the PDE problem.

        Returns:
            tuple[dl.Function | None, dl.Function | None, dl.Function | None]: Created dolfin
                functions for drift, diffusion, and initial condition, if their string
                representation has been provided.
        """
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
        """Decide which function space is the parameter space, depending on inference type.

        Returns:
            dl.FunctionSpace: Space to use for the parameter variable
        """
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
        """Convert dolfin functions to numpy arrays, if they are provided by the user.

        Returns:
            tuple[npt.NDArray[np.floating] | None, npt.NDArray[np.floating] | None,
                tuple[npt.NDArray[np.floating]] | None]: Drift, diffusion, and initial condition
                as numpy arrays, if provided by the user.
        """
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
        """Create homogeneous Dirichlet Boundary conditions on the given mesh.

        Returns:
            dl.DirichletBC: Dolfin boundary conditions.
        """
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
        """Generate generic weak form taking forward, parameter, and adjoint variable.

        The implemented PDE forms in the [`weakforms`][spin.core.weakforms] explicitly take
        drift and diffusion as coefficient functions. This method provides a wrapper that dispatches
        to either drift, diffusion, or both as the parameter, depending on the inference mode.
        The resultin form wrapper has the generic argument signature (forward, parameter, adjoint)
        that is required for computations in Hippylib.

        Raises:
            ValueError: Checks that drift has been provided for "diffusion_only" inference.
            ValueError: Checks that diffusion has been provided for "drift_only" inference.

        Returns:
            Callable[[Any, Any, Any], ufl.Form]: UFL weak form wrapper with generic signature.
        """
        # Drift only: Drift is parameter, diffusion needs to be given as coefficient function
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

        # Diffusion only: Diffusion is parameter, drift needs to be given as coefficient function
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

        # Drift and diffusion: Both drift and diffusion are parameters, on composite space
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
        """Create matrix exponential for diagonal matrix in UFL syntax.

        Args:
            matrix_diagonal (dl.Function | ufl.tensors.ListTensor): Diagonal entries

        Returns:
            ufl.tensors.ListTensor: Diagonal matrix exponential.
        """
        diagonal_components = [ufl.exp(component) for component in matrix_diagonal]
        diagonal_components = ufl.as_vector(diagonal_components)
        matrix_exponential = ufl.diag(diagonal_components)
        return matrix_exponential

    # ----------------------------------------------------------------------------------------------
    def _create_variational_problem(self) -> hl.PDEProblem:
        """Create the Hippylib-conformant `PDEProblem` object.

        This method utilizes the previously assembled dolfin objects to create an object conforming
        to the interface of the hippylib `PDEProblem` class through nominal subtyping. For
        stationary problems. this is the `PDEVariationalProblem` class, for time-dependent problems,
        we utilize the `TDPDELinearVariationalProblem` class in SPIN.

        !!! warning
            Time-dependent PDE inference is not yet implemented.

        Raises:
            ValueError: Checks that time-stepping parameters are provided for time-dependent PDEs.
            ValueError: Checks that initial condition is provided for time-dependent PDEs.
            NotImplementedError: Indicates that time-dependent problems are not yet implemented.

        Returns:
            hl.PDEProblem: Hippylib object for inference.
        """
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
                is_fwd_linear=self._pde_type.linear,
            )
        else:
            if self._start_time is None or self._end_time is None or self._num_steps is None:
                raise ValueError("Time parameters are required for Fokker-Planck PDE inference.")
            if initial_condition is None:
                raise ValueError("Initial condition is required for Fokker-Planck PDE inference.")
            initial_condition = fex_converter.create_dolfin_function(
                self._initial_condition, self._function_space_variables
            )
            raise NotImplementedError("Time-dependent PDE inference is not yet implemented.")
            spin_problem = hlx.TDPDELinearVariationalProblem(
                function_space_list,
                self._weak_form_wrapper,
                self._boundary_condition,
                self._boundary_condition,
                initial_condition,
                self._start_time,
                self._end_time,
                self._num_steps,
                is_fwd_linear=self._pde_type.linear,
            )
        return spin_problem
