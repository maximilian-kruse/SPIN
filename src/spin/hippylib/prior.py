r"""Hippylib Bilaplacian prior fields for vector-valued functions.

This module implements Gaussian prior fields for non-parametric Bayesian inference with Hippylib.
It extends the Hippylib `SqrtPrecisionPDEPrior` class to vector-valued fields, where the individual
component fields are non-stationary, but statistically independent. For optimization-based
inference, we define the negative log distribution of the prior as a cost functional. With a given
precision matrix $\mathbf{R}$ and mean vector \bar{\mathbf{m}}, the discretized cost functional
reads

$$
    J_{\text{prior}}(\mathbf{m}) = \frac{1}{2} ||(\mathbf{m} - \bar{\mathbf{m}})||_\mathbf{R}^2.
$$

The prior object provides methods for evaluating this functional, as well as its gradient and
Hessian-vector products. In addition, we can draw samples from the distribution.
We focus on a special class of priors with Matérn or Matérn-like covariance structure. We exploit
the correspondence between such fields and the solution of SPDEs with specific left-hand-side
left-hand-side operators, as proposed [here](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2011.00777.x).
This allows for the definition of a sparsite-promoting precision operator, in this case one
reminiscent of a bilaplacian operator. For a domain $\Omega\subset\mathbb{R}^d$, we write

$$
    \mathcal{R} = \left(\delta - \gamma \Delta\right)^2, x\in\Omega,
$$

The parameters $\gamma$ and $\delta$ can be spatially varying, and have direct correspondence
(neglecting boundary effects) to the variance and correlation length of the prior field,

$$
    \sigma^2 = \frac{\Gamma(\nu)}{(4\pi^{d/2})}\frac{1}{\delta^\nu\gamma^{d/2}},\quad
    \rho = \sqrt{\frac{8\nu\gamma}{\delta}},\quad \nu = 2 - \frac{d}{2}.
$$

Lastly, to mitigate boundary effects, we can apply Robin boundary conditions to the prior field,

$$
    \mathcal{R} = \gamma \nabla m \cdot \mathbf{n} + \frac{\sqrt{\delta\gamma}}{c} m,
    x\in\partial\Omega,
$$

with an empirically optimized constant $c=1.42$.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import dolfin as dl
import hippylib as hl
import numpy as np
import numpy.typing as npt
import scipy as sp
import ufl
from beartype.vale import Is

from spin.fenics import converter as fex_converter


# ==================================================================================================
class SqrtPrecisionPDEPrior(hl.prior._Prior):  # noqa: SLF001
    r"""Re-implementation of Hippylib's `SqrtPrecisionPDEPrior` prior.

    This class is an almost one-to-one re-implementation of the `SqrtPrecisionPDEPrior` class from
    Hippylib. It introduces some minor improvements to the original design, and, more importantly,
    has better support for vector-valued prior fields.

    On the component level, this class implements Gaussian prior fields, whose precision operator
    is given as a bilaplacian-like operator, reading for a domain $\Omega\subset\mathbb{R}^d$:

    $$
        R_i= \big(\delta_i(x) - \gamma_I(x) \Delta\big)^2. \mathbf{x}\in\Omega,
    $$
    where the parameters $\delta_i(x)$ and $\gamma_i(x)$ can be spatially varying. Vector-valued
    prior treat components as statistically independent, meaning that the precision is block-
    structured.

    According to the Hippylib `_Prior` nominal subtyping, this class has methods for evaluation of
    the prior cost functional (the negative log distribution), its parametric gradient,
    and the action of the Hessian (which is the precision operator). It further allows for sampling
    from the prior field via a specialized, sparse Cholesky decomposition.

    Attributes:
        mean (dl.Vector | dl.PETScVector): Mean vector of the prior field (Hippylib interface)
        M (dl.Matrix): Mass matrix of the prior field (Hippylib interface)
        Msolver (dl.PETScKrylovSolver): Solver for the mass matrix (Hippylib interface)
        R (hl.prior._BilaplacianR): Precision operator action of the prior field
            (Hippylib interface)
        Rsolver (hl.prior._BilaplacianRsolver): Covariance operator action of the prior field
            (Hippylib interface)

    Methods:
        init_vector: Initialize a vector for prior computations
        sample: Draw a sample from the prior field
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        function_space: dl.FunctionSpace,
        variational_form_handler: Callable[[ufl.Argument, ufl.Argument], ufl.Form],
        mean: dl.Vector | dl.PETScVector,
        cg_solver_relative_tolerance: Annotated[float, Is[lambda x: 0 < x < 1]] = 1e-12,
        cg_solver_max_iter: Annotated[int, Is[lambda x: x > 0]] = 1000,
    ) -> None:
        """Constructor, building the prior internally.

        The Prior works on vector function spaces, with the corresponding build-up in the
        [`BiLaplacianVectorPriorBuilder`][spin.hippylib.prior.BiLaplacianVectorPriorBuilder] class.

        Strictly speaking, the size of the constructor is a violation of good design principle,
        in that this class is its own builder. However, we stick to this pattern according to the
        Hippylib interface, and sub-divide the constructor into a sequence of smaller methods for
        clarity.

        Args:
            function_space (dl.FunctionSpace): Scalar function space.
            variational_form_handler (Callable[[ufl.Argument, ufl.Argument], ufl.Form]): variational
                form describing the underlying prior field as an SPDE.
            mean (dl.Vector | dl.PETScVector): Mean vector.
            cg_solver_relative_tolerance (int, optional): Relative tolerance for CG solver for
                matrix-free inversion. Defaults to 1e-12.
            cg_solver_max_iter (int, optional): Maximum number of iterations for CG solver for
                matrix-free inversion. Defaults to 1000.

        Raises:
            ValueError: Checks if the mean vector has the same dimension as the function space.
        """
        if not mean.size() == function_space.dim():
            raise ValueError(
                f"The mean vector must have the same dimension ({self.mean.size()}) "
                f"as the function space ({function_space.dim()})."
            )
        # Set persistent variables
        self._function_space = function_space
        self._mean = mean

        # Assemble mass matrix and bilaplace operator matrix
        trial_function = dl.TrialFunction(function_space)
        test_function = dl.TestFunction(function_space)
        self._mass_matrix, self._matern_sdpe_matrix = self._assemble_system_matrices(
            trial_function, test_function, variational_form_handler
        )

        # Initialize solvers for inversion of mass matrix and bilaplace operator matrix
        self._mass_matrix_solver, self._matern_spde_matrix_solver = self._initialize_solvers(
            cg_solver_max_iter, cg_solver_relative_tolerance
        )

        # Set up mass_matrix cholesky factor in quadrature space
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

        # Set up wrappers for prior precision and covariance operators
        self._bilaplacian_precision_operator = hl.prior._BilaplacianR(  # noqa: SLF001
            self._matern_sdpe_matrix, self._mass_matrix_solver
        )
        self._bilaplacian_covariance_operator = hl.prior._BilaplacianRsolver(  # noqa: SLF001
            self._matern_spde_matrix_solver, self._mass_matrix
        )

        # Restore setting and initialize hippylib interface
        self._restore_quadrature_representation(representation_buffers)
        self._set_up_hippylib_interface()

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _assemble_system_matrices(
        trial_function: ufl.Argument,
        test_function: ufl.Argument,
        variational_form_handler: Callable,
    ) -> tuple[dl.Matrix, dl.Matrix]:
        """Assemble the mass and bilaplacian operator matrix for the FEM system.

        Args:
            trial_function (ufl.Argument): FEM trial function.
            test_function (ufl.Argument): FEM test function.
            variational_form_handler (Callable): UFL form callable for the bilaplacian operator.

        Returns:
            tuple[dl.Matrix, dl.Matrix]: Mass and operator matrices.
        """
        mass_matrix_term = ufl.inner(trial_function, test_function) * ufl.dx
        mass_matrix = dl.assemble(mass_matrix_term)
        matern_spde_term = variational_form_handler(trial_function, test_function)
        matern_spde_matrix = dl.assemble(matern_spde_term)
        return mass_matrix, matern_spde_matrix

    # ----------------------------------------------------------------------------------------------
    def _initialize_solvers(
        self, cg_solver_max_iter: int, cg_solver_relative_tolerance: Real
    ) -> tuple[dl.PETScKrylovSolver, dl.PETScKrylovSolver]:
        """Initialize solvers for matrix-free inversion of the mass and bilalacian operator matrix.

        Both matrices are inverted matrix-free using the CG method. For the mass matrix, we employ
        a simple Jacobi preconditioner, while for the bilaplacian operator matrix, we use the
        algebraic multigrid method.

        Args:
            cg_solver_max_iter (int): Maximum number of CG iterations.
            cg_solver_relative_tolerance (Real): Relative tolerance for CG termination.

        Returns:
            tuple[dl.PETScKrylovSolver, dl.PETScKrylovSolver]: Initialized solver objects,
                acting as application of the inverse matrices to vectors.
        """
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
    @staticmethod
    def _modify_quadrature_representation() -> tuple[object, object]:
        """Change UFL form representation for quadrature space.

        The change in settings is utlized for the assembly of the mass matrix cholesky factor.

        Returns:
            tuple[object, object]: Buffers holding the default representation settings.
        """
        quadrature_degree_buffer = dl.parameters["form_compiler"]["quadrature_degree"]
        representation_buffer = dl.parameters["form_compiler"]["representation"]
        dl.parameters["form_compiler"]["quadrature_degree"] = -1
        dl.parameters["form_compiler"]["representation"] = "quadrature"
        return quadrature_degree_buffer, representation_buffer

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _restore_quadrature_representation(representation_buffers: tuple[object, object]) -> None:
        """Change UFL representation back to default.

        Reverts effects of `_modify_quadrature_representation`.

        Args:
            representation_buffers (tuple[object, object]): Buffers holding the default
                representation settings.
        """
        quadrature_degree_buffer, representation_buffer = representation_buffers
        dl.parameters["form_compiler"]["quadrature_degree"] = quadrature_degree_buffer
        dl.parameters["form_compiler"]["representation"] = representation_buffer

    # ----------------------------------------------------------------------------------------------
    def _set_up_quadrature_space(
        self, quadrature_degree: int
    ) -> tuple[dl.FunctionSpace, ufl.Argument, ufl.Argument]:
        """Set up quadrature space, trial and test function.

        Used for the assembly of the mass matrix cholseky factor.

        Args:
            quadrature_degree (int): Degree of the quadrature representation.

        Returns:
            tuple[dl.FunctionSpace, ufl.Argument, ufl.Argument]: Quadrature space, trial and test
                functions.
        """
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
        """Assemble mass matrix Cholesky factor.

        As in Hippylib, we provide a special form of the mass matrix decomposition for sampling
        from the prior field. The decomposition is rectangular and sparse, based on the
        quadrature space functionality in Fenics. For more details, we refer to the appendix of the
        Hippylib [paper](https://dl.acm.org/doi/10.1145/3428447).

        Args:
            quadrature_degree (int): Degree of the quadrature representation.
            quadrature_space (dl.FunctionSpace): Quadrature function space.
            quadrature_trial_function (ufl.Argument): Quadrature space trial function.
            quadrature_test_function (ufl.Argument): Quadrature space test function.
            test_function (ufl.Argument): Test function on original space.

        Returns:
            dl.Matrix: Sparse Cholesky factor of the mass matrix
        """
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
        """Assign internal attributes to public properties, to adhere to Hippylib interface."""
        self.mean = self._mean
        self.M = self._mass_matrix
        self.Msolver = self._mass_matrix_solver
        self.R = self._bilaplacian_precision_operator
        self.Rsolver = self._bilaplacian_covariance_operator

    # ----------------------------------------------------------------------------------------------
    def init_vector(self, vector_to_init: dl.Vector, matrix_dim: int | str) -> None:
        """Initialize a vector to be used for prior computations.

        Args:
            vector_to_init (dl.Vector): Dolfin vector to initialize.
            matrix_dim (int | str): Matrix "dimension" to initialize the vector for.
        """
        if matrix_dim == "noise":
            self._mass_matrix_cholesky_Factor.init_vector(vector_to_init, 1)
        else:
            self._matern_sdpe_matrix.init_vector(vector_to_init, matrix_dim)

    # ----------------------------------------------------------------------------------------------
    def sample(
        self, noise_vector: dl.Vector, matern_field_vector: dl.Vector, add_mean: bool = True
    ) -> None:
        """Draw a sample from the prior field.

        Args:
            noise_vector (dl.Vector): Noise vector for the sample.
            matern_field_vector (dl.Vector): Resulting sample vector.
            add_mean (bool, optional): Whether to add prior mean vector to sample. Defaults to True.
        """
        rhs = self._mass_matrix_cholesky_Factor * noise_vector
        self._matern_spde_matrix_solver.solve(matern_field_vector, rhs)
        if add_mean:
            matern_field_vector.axpy(1.0, self.mean)


# ==================================================================================================
@dataclass
class PriorSettings:
    """Settings for the setup of a bilaplacian vector prior.

    Attributes:
        function_space (dl.FunctionSpace): Function space the prior is defined on.
        mean (Iterable[str]): Mean functions, given as a sequence of strings in dolfin syntax, will
            be compiled to dolfin expressions.
        variance (Iterable[str]): Variance functions, given as a sequence of strings in dolfin
            syntax, will be compiled to dolfin expressions.
        correlation_length (Iterable[str]): Correlation length functions, given as a sequence of
            strings in dolfin syntax, will be compiled to dolfin expressions.
        anisotropy_tensor (Iterable[hl.ExpressionModule.AnisTensor2D], optional): Anisitropy tensor
            for anisotropic covariance structure. Defaults to None.
        cg_solver_relative_tolerance (Annotated[float, Is[lambda x: 0 < x < 1]]): Tolerance of CG
            solver for matrix-free application of inverse operators. Defaults to 1e-12.
        cg_solver_max_iter (Annotated[int, Is[lambda x: x > 0]]): Maximum iteration number of CG
            solver for matrix-free application of inverse operators. Defaults to 1000.
        robin_bc (bool): Whether to apply Robin boundary conditions. Defaults to False.
        robin_bc_const (Real): Constant for the Robin boundary condition. Defaults to 1.42.
    """

    function_space: dl.FunctionSpace
    mean: Iterable[str]
    variance: Iterable[str]
    correlation_length: Iterable[str]
    anisotropy_tensor: Iterable[hl.ExpressionModule.AnisTensor2D] = None  # type: ignore # noqa: PGH003
    cg_solver_relative_tolerance: Annotated[float, Is[lambda x: 0 < x < 1]] = 1e-12
    cg_solver_max_iter: Annotated[int, Is[lambda x: x > 0]] = 1000
    robin_bc: bool = False
    robin_bc_const: Real = 1.42


# ==================================================================================================
@dataclass
class Prior:
    """SPIN prior object, wrapping the Hippylib object with addiaional data and functionality.

    Attributes:
        hippylib_prior (hl.prior._Prior): Hippylib prior object
        function_space (dl.FunctionSpace): Underlying function space
        mean_array (npt.NDArray[np.floating]): Mean as numpy array
        variance_array (npt.NDArray[np.floating]): Ideal pointwise variance (without boundary
            effects) as numpy array
        correlation_length_array (npt.NDArray[np.floating]): Ideal correlation length (without
            boundary effects) as numpy array
        spde_matern_matrix (sp.sparse.coo_array): Matern SPDE matrix in COO format
        mass_matrix (sp.sparse.coo_array): Mass matrix in COO format

    Methods:
        compute_variance_with_boundaries: Approximate the actual variance field with boundary
            effects.
        compute_precision_with_boundaries: Compute the actual precision matrix with boundary
            effects (only for testing and development).
        evaluate_gradient: Evaluate the parametric gradient of the prior cost functional.
    """

    hippylib_prior: hl.prior._Prior
    function_space: dl.FunctionSpace
    mean_array: npt.NDArray[np.floating]
    variance_array: npt.NDArray[np.floating]
    correlation_length_array: npt.NDArray[np.floating]
    spde_matern_matrix: sp.sparse.coo_array
    mass_matrix: sp.sparse.coo_array

    # ----------------------------------------------------------------------------------------------
    def compute_variance_with_boundaries(
        self,
        method: Annotated[str, Is[lambda x: x in ("Exact", "Randomized")]],
        num_eigenvalues_randomized: Annotated[int, Is[lambda x: x > 0]] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute the actual variance of the prior field.

        Opposed to the prescribed, theoretical prior variance, this method evaluates the actual
        variance for the discretized prior field on a bounded domain. The variance can be computed
        exactly or approximated by a randomized method. The exact method should only be used for
        small test problems. The randomized method is based on a truncated eigendecomposition of
        the covariance matrix, and can therefore be used for larger problems.

        Args:
            method (str): Method for variance computation, either "Exact" or "Randomized".
            num_eigenvalues_randomized (int, optional): Number of eigenvalues to use for
                randomized diagonal estimation. Only necessary for option "Randomized".
                Defaults to None.

        Raises:
            ValueError: Checks that a number of eigenvalues is provided for the "Randomized" method.

        Returns:
            npt.NDArray[np.floating]: Pointwise variance of the prior field..
        """
        if method == "Exact":
            variance = self.hippylib_prior.pointwise_variance()
        if method == "Randomized":
            if num_eigenvalues_randomized is None:
                raise ValueError(
                    "num_eigenvalues_randomized must be provided for 'Randomized' method."
                )
            variance = self.hippylib_prior.pointwise_variance(
                method=method, r=num_eigenvalues_randomized
            )
        pointwise_variance = fex_converter.convert_to_numpy(variance, self.function_space)
        return pointwise_variance

    # ----------------------------------------------------------------------------------------------
    def compute_precision_with_boundaries(self) -> npt.NDArray[np.floating]:
        """Compute the entire precision matrix of the prior field.

        Should only be used for small test problems.

        Returns:
            npt.NDArray[np.floating]: Prior field precision matrix.
        """
        matrix_rows = []
        for i in range(self.function_space.dim()):
            input_vector = dl.Vector()
            output_vector = dl.Vector()
            self.hippylib_prior.init_vector(input_vector, 0)
            self.hippylib_prior.init_vector(output_vector, 0)
            input_vector[i] = 1.0
            self.hippylib_prior.R.mult(input_vector, output_vector)
            output_array = fex_converter.convert_to_numpy(output_vector, self.function_space)
            matrix_rows.append(output_array)

        precision_matrix = np.stack(matrix_rows, axis=0)
        return precision_matrix

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, parameter_array: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Evaluate parametric gradient of the prior cost functional (negative log distribution).

        Args:
            parameter_array (npt.NDArray[np.floating]): Parameter value for which to compute the
                gradient.

        Returns:
            npt.NDArray[np.floating]: Gradient array
        """
        if not parameter_array.size == self.function_space.dim():
            raise ValueError(
                f"Parameter array must have the same dimension ({parameter_array.size}) "
                f"as the function space ({self.function_space.dim()})."
            )
        parameter_vector = fex_converter.convert_to_dolfin(
            parameter_array, self.function_space
        ).vector()
        grad_vector = dl.Vector(parameter_vector)
        self.hippylib_prior.grad(parameter_vector, grad_vector)
        grad_array = fex_converter.convert_to_numpy(grad_vector, self.function_space)
        return grad_array


# ==================================================================================================
class BilaplacianVectorPriorBuilder:
    """Builder for vector-valued Bilaplacian priors.

    This builder assembles a vector-valued prior field, with bilaplacian-like precision operator.
    It constructs component-wise variational forms and initializes an
    [`SQRTPrecisionPDEPrior`][spin.hippylib.prior.SQRTPrecisionPDEPrior] object with the resulting
    data. Different variance and correlation structures can be supplied for each component, but
    the components themselves are statistically independent of each other.

    Methods:
        build: Main interface of the builder.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, prior_settings: PriorSettings) -> None:
        """Constructor, internally initializes data structures.

        Args:
            prior_settings (PriorSettings): Configuration and data for the prior field.
        """
        if (
            not len(prior_settings.mean)
            == len(prior_settings.variance)
            == len(prior_settings.correlation_length)
            == prior_settings.function_space.num_sub_spaces()
        ):
            raise ValueError(
                f"Number of mean functions ({len(prior_settings.mean)}), "
                f"variance functions ({len(prior_settings.variance)}), and "
                f"correlation length functions ({len(prior_settings.correlation_length)}) "
                f"must be equal to the number of function space components "
                f"({prior_settings.function_space.num_sub_spaces()})."
            )
        self._function_space = prior_settings.function_space
        self._num_components = self._function_space.num_sub_spaces()
        self._domain_dim = self._function_space.mesh().geometry().dim()
        self._mean = fex_converter.create_dolfin_function(
            prior_settings.mean, self._function_space
        ).vector()
        self._variance = fex_converter.create_dolfin_function(
            prior_settings.variance, self._function_space
        ).vector()
        self._correlation_length = fex_converter.create_dolfin_function(
            prior_settings.correlation_length, self._function_space
        ).vector()
        self._anisotropy_tensor = prior_settings.anisotropy_tensor
        self._cg_solver_relative_tolerance = prior_settings.cg_solver_relative_tolerance
        self._cg_solver_max_iter = prior_settings.cg_solver_max_iter
        self._robin_bc = prior_settings.robin_bc
        self._robin_bc_const = prior_settings.robin_bc_const
        self._gamma = None
        self._delta = None

    # ----------------------------------------------------------------------------------------------
    def build(self) -> Prior:
        """Main interface of the builder.

        Returns a Prior wrapper object, containing the Hippylib prior object alongside additional
        data and functionalities.

        Returns:
            Prior: SPIN Prior object.
        """
        self._gamma, self._delta = self._convert_prior_coefficients()

        # Define vctor variational form handler as sum over component forms
        def variational_form_handler(
            trial_function: ufl.Argument, test_function: ufl.Argument
        ) -> ufl.Form:
            component_forms = []
            for i in range(self._num_components):
                scalar_form = self._generate_scalar_form(trial_function, test_function, i)
                component_forms.append(scalar_form)
            vector_form = sum(component_forms)
            return vector_form

        # Initialize prior object
        prior_object = SqrtPrecisionPDEPrior(
            self._function_space,
            variational_form_handler,
            self._mean,
            self._cg_solver_relative_tolerance,
            self._cg_solver_max_iter,
        )

        # Get other prior data in readable format
        mean_array = fex_converter.convert_to_numpy(self._mean, self._function_space)
        variance_array = fex_converter.convert_to_numpy(self._variance, self._function_space)
        correlation_length_array = fex_converter.convert_to_numpy(
            self._correlation_length, self._function_space
        )
        spde_matern_matrix = fex_converter.convert_matrix_to_scipy(prior_object._matern_sdpe_matrix)  # noqa: SLF001
        mass_matrix = fex_converter.convert_matrix_to_scipy(prior_object._mass_matrix)  # noqa: SLF001

        # Return prior wrapper object
        prior = Prior(
            hippylib_prior=prior_object,
            function_space=self._function_space,
            mean_array=mean_array,
            variance_array=variance_array,
            correlation_length_array=correlation_length_array,
            spde_matern_matrix=spde_matern_matrix,
            mass_matrix=mass_matrix,
        )
        return prior

    # ----------------------------------------------------------------------------------------------
    def _convert_prior_coefficients(self) -> tuple[dl.Function, dl.Function]:
        r"""Convert variance and correlation length fields to prior coefficients.

        For the Bilaplacian prior, the variance $\sigma^2$ and correlation length $\rho$ can be
        converted to the prior SPDE parameters $\gamma$ and $\delta$ as

        $$
        \begin{gather*}
            \nu = 2- \frac{d}{2}, \quad \kappa = \frac{\sqrt{8\nu}}{\rho}, \\
            s = \sigma \kappa^\nu \sqrt{\frac{4\pi^{d/2}}{\Gamma(\nu)}}, \\
            \gamma = \frac{1}{s}, \quad \delta = \frac{\kappa^2}{s}.
        \end{gather*}
        $$

        Returns:
            tuple[dl.Function, dl.Function]: Coefficients $\gamma$ and $\delta$ for the prior field.
        """
        variance_array = fex_converter.convert_to_numpy(self._variance, self._function_space)
        correlation_length_array = fex_converter.convert_to_numpy(
            self._correlation_length, self._function_space
        )
        sobolev_exponent = 2 - 0.5 * self._domain_dim
        kappa_array = np.sqrt(8 * sobolev_exponent) / correlation_length_array
        s_array = (
            np.sqrt(variance_array)
            * np.power(kappa_array, sobolev_exponent)
            * np.sqrt(np.power(4 * np.pi, 0.5 * self._domain_dim) / math.gamma(sobolev_exponent))
        )
        gamma_array = 1 / s_array
        delta_array = np.power(kappa_array, 2) / s_array
        gamma = fex_converter.convert_to_dolfin(gamma_array, self._function_space)
        delta = fex_converter.convert_to_dolfin(delta_array, self._function_space)
        return gamma, delta

    # ----------------------------------------------------------------------------------------------
    def _generate_scalar_form(
        self, trial_function: ufl.Argument, test_function: ufl.Argument, index: int
    ) -> ufl.Form:
        r"""Generate the compomnent-wise UFL form for the Bilaplacian prior.

        The component form has three different components. Given a trial function $u$ and test
        function $v$, this components are:

        1. The mass matrix term, simply the discretization of $\int_{\Omega}\delta u v d\mathbf{x}$
        2. The stiffness matrix term, discretizing
            $\int_{\Omega}\gamma \nabla u \cdot \nabla v d\mathbf{x}$
        3. The Robin boundary term, discretizing
            $\int_{\partial\Omega} \frac{\sqrt{\delta\gamma}}{c} u v dx

        The boundary term is only applied if Robin boundary conditions are set to true with a
        boundary constant $c$. The stiffness matrix term can additionally contain an anisotropy
        tensor, if the user provides one.

        Args:
            trial_function (ufl.Argument): FE Trial function.
            test_function (ufl.Argument): FE Test function.
            index (int): Index of the component being discretized in a vector context.

        Returns:
            ufl.Form: UFL form for the component-wise discretization.
        """
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
                gamma_component
                * ufl.inner(
                    self._anisotropy_tensor * ufl.grad(trial_component), ufl.grad(test_component)
                )
                * ufl.dx
            )

        if self._robin_bc:
            robin_coeff = ufl.sqrt(delta_component * gamma_component) / dl.Constant(
                self._robin_bc_const
            )
        else:
            robin_coeff = dl.Constant(0.0)
        boundary_term = robin_coeff * ufl.inner(trial_component, test_component) * ufl.ds

        scalar_form = mass_matrix_term + stiffness_matrix_term + boundary_term
        return scalar_form
