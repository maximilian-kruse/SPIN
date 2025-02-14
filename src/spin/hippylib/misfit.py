r"""Misfit/Likelihood for usage in Hippylib.

This module implements the likelihood for Bayesian inverse problems in the Hippylib framework.
In the optimization context, it can be interpreted as a misfit functional. We focus on additive,
zero-centered Gaussian noise models. Given a PDE solution operator
$\mathcal{G}(\mathbf{m}) = \mathbf{u}$, we define for each output component $u$ the projection to
points of observation $u_d = \mathcal{B}u in \mathbb{R}^q$ with observation operator
$\mathcal{B}: \mathcal{U} \to \mathbb{R}^q$. We then write for the data $d$ that

$$
    d = u_d + \eta, \quad \eta \sim \mathcal{N}(0, \Gamma).
$$

Therefore, we can define the likelihood density for a single variable component u as

$$
    \pi_{\text{like}} \propto \exp\left(-\frac{1}{2} \| \mathcal{B}u - d \|_{\Gamma^{-1}}^2\right).
$$

The below methods implements discrete versions of the projection operator and diagonal noise
precision matrices with pointwise varying coefficients. The misfit functional is then given as
the negative log-likelihood, along with gradients and Hessian-vector products of the misfit w.r.t.
$u$.

Classes:
    MisfitSettings: Dataclass to store settings for misfit construction.
    DiscreteMisfit: Misfit class for scalar misfit problems.
    VectorMisfit: Misfit class for vector-valued misfit problems.
    TDMisfit: Misfit class for time-dependent misfit problems (not implemented yet).
    Misfit: Dataclass to store assembled misfit objects.
    MisfitBuilder: Builder class to assemble misfit objects from settings.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import dolfin as dl
import hippylib as hl
import numpy as np
import numpy.typing as npt
import scipy as sp
from beartype.vale import IsEqual
from petsc4py import PETSc

from spin.fenics import converter as fex_converter


# ==================================================================================================
def assemble_pointwise_observation_operator(
    function_space: dl.FunctionSpace, observation_points: npt.NDArray[np.floating]
) -> dl.Matrix:
    r"""Assemble pointwise observation matrix.

    This is a simple PEP8-conformant wrapper to Hippylibs `assemblePointwiseObservation` function.
    The function takes an input function space and points of observation, to which a function
    shalle be projected. Supports multidimensional domains. In a $d$-dimensional domain with $q_d$
    observation points per dimension, the `observation_points` array should have shape
    $q_d\timed d$. The resulting projection matrix $\mathbf{B}$ has shape $q_d d \times N$,
    where $N$ is the number of degrees of freedom in the function space.

    !!! warning
        The observation operator works only component-wise, on scalar function spaces.

    Args:
        function_space (dl.FunctionSpace): Function space of vector to project.
        observation_points (npt.NDArray[np.floating]): Points of observation to project to.

    Returns:
        dl.Matrix: Projection operator/matrix.
    """
    observation_matrix = hl.assemblePointwiseObservation(function_space, observation_points)
    return observation_matrix


# --------------------------------------------------------------------------------------------------
def assemble_noise_precision_matrix(noise_variance: npt.NDArray[np.floating]) -> dl.PETScMatrix:
    r"""Assemble the precision matrix for a Gaussian noise model.

    This function only supports diagonal precision matrices, but allows for pointwise varying
    values. For $q$ overall observation points, the `noise_variance` array should have shape $q$,
    and the resulting precision matrix $\Gamma^{-1}$ will have shape $q \times q$.

    !!! warning
        The precision matric works only component-wise, on scalar function spaces.

    Args:
        noise_variance (npt.NDArray[np.floating]): Diagonal elements of the precision matrix.

    Returns:
        dl.PETScMatrix: Sparse precision matrix in dolfin format.
    """
    noise_precision = 1.0 / noise_variance
    petsc_matrix = PETSc.Mat().createAIJ(
        size=(noise_precision.size, noise_precision.size), comm=dl.MPI.comm_world
    )
    petsc_matrix.setUp()
    for i, value in enumerate(noise_precision):
        petsc_matrix.setValues(i, i, value)
    petsc_matrix.assemble()
    precision_matrix = dl.PETScMatrix(petsc_matrix)
    return precision_matrix


# ==================================================================================================
class DiscreteMisfit(hl.Misfit):
    """summary."""

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        data: npt.NDArray[np.floating],
        observation_matrix: dl.Matrix,
        noise_precision_matrix: dl.PETScMatrix,
    ) -> None:
        """Constructor, initializes internal data structures and buffers.

        Args:
            data (npt.NDArray[np.floating]): Observed data.
            observation_matrix (dl.Matrix): Projection matrix from function to observation space.
            noise_precision_matrix (dl.PETScMatrix): Precision of the noise model.
        """
        self._observation_matrix = observation_matrix
        self._noise_precision_matrix = noise_precision_matrix
        self._data = dl.Vector()
        self._vector_buffer_one = dl.Vector()
        self._vector_buffer_two = dl.Vector()
        self._observation_matrix.init_vector(self._data, 0)
        self._data.set_local(data)
        self._data.apply("insert")
        self._observation_matrix.init_vector(self._vector_buffer_two, 0)
        self._observation_matrix.init_vector(self._vector_buffer_two, 0)

    # ----------------------------------------------------------------------------------------------
    def cost(
        self,
        state_list: Iterable[
            dl.Vector | dl.PETScVector,
            dl.Vector | dl.PETScVector | None,
            dl.Vector | dl.PETScVector | None,
        ],
    ) -> Real:
        r"""Evaluate the cost of the misfit functional.

        For a given state $u$, data, $d$, projection operator $B$ and noise precision $\Gamma^{-1}$,
        the cost is given as

        $$
            J_{\text{misfit}}(u) = \frac{1}{2} \| B u - d \|_{\Gamma^{-1}}^2.
        $$

        Args:
            state_list (Iterable): List of forward, parameter and adjoint variables $(u,m,p)$,
                only $u$ is required.

        Returns:
            Real: Cost value for given state.
        """
        forward_vector = state_list[hl.STATE]
        self._observation_matrix.mult(forward_vector, self._vector_buffer_one)
        self._vector_buffer_one.axpy(-1.0, self._data)
        self._noise_precision_matrix.mult(self._vector_buffer_one, self._vector_buffer_two)
        cost = 0.5 * self._vector_buffer_one.inner(self._vector_buffer_two)
        return cost

    # ----------------------------------------------------------------------------------------------
    def grad(
        self,
        derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        state_list: Iterable[
            dl.Vector | dl.PETScVector,
            dl.Vector | dl.PETScVector | None,
            dl.Vector | dl.PETScVector | None,
        ],
        output_vector: dl.Vector | dl.PETScVector,
    ) -> None:
        """Gradient of the cost functional given in the `cost` method.

        Args:
            derivative_type (Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]]): Variable
                with respect to which the gradient is computed. Only `STATE` yields a non-zero
                result.
            state_list (Iterable): List of forward, parameter and adjoint variables $(u,m,p)$,
                only $u$ is required.
            output_vector (dl.Vector | dl.PETScVector): COmputed gradient.
        """
        if derivative_type == hl.STATE:
            forward_vector = state_list[hl.STATE]
            self._observation_matrix.mult(forward_vector, self._vector_buffer_one)
            self._vector_buffer_one.axpy(-1.0, self._data)
            self._noise_precision_matrix.mult(self._vector_buffer_one, self._vector_buffer_two)
            self._observation_matrix.transpmult(self._vector_buffer_two, output_vector)
        elif derivative_type == hl.PARAMETER:
            output_vector.zero()

    # ----------------------------------------------------------------------------------------------
    def setLinearizationPoint(  # noqa: N802
        self,
        _state_list: Iterable[
            dl.Vector | dl.PETScVector | None,
            dl.Vector | dl.PETScVector | None,
            dl.Vector | dl.PETScVector | None,
        ],
        _gauss_newton_approx: bool,
    ) -> None:
        """Set point for Hessian evaluation.

        This method does nothing, as the misfit is a quadratic form and its second variation is
        thus constant. Only required for interface compatibility.
        """
        pass

    # ----------------------------------------------------------------------------------------------
    def apply_ij(
        self,
        first_derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        second_derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        hvp_direction: dl.Vector | dl.PETScVector,
        output_vector: dl.Vector | dl.PETScVector,
    ) -> None:
        """Apply Hessian vector-product.

        Second derivatives are only computed with respect to the forward variable $u$.

        Args:
            first_derivative_type (Annotated[int, IsEqual[hl.STATE]  |  IsEqual[hl.PARAMETER]]):
                Variable to compute first derivative with respect to. Only `STATE` yields a
                non-zero result.
            second_derivative_type (Annotated[int, IsEqual[hl.STATE]  |  IsEqual[hl.PARAMETER]]):
                Variable to compute second derivative with respect to. Only `STATE` yields a
                non-zero result.
            hvp_direction (dl.Vector | dl.PETScVector): Direction vector for which to compute
                Hessian-vector product.
            output_vector (dl.Vector | dl.PETScVector): Resulting vector.
        """
        if first_derivative_type == hl.STATE and second_derivative_type == hl.STATE:
            self._observation_matrix.mult(hvp_direction, self._vector_buffer_one)
            self._noise_precision_matrix.mult(self._vector_buffer_one, self._vector_buffer_two)
            self._observation_matrix.transpmult(self._vector_buffer_two, output_vector)
        else:
            output_vector.zero()

    # ----------------------------------------------------------------------------------------------
    @property
    def observation_matrix(self) -> dl.Matrix:
        """Return internally stored observation matrix."""
        return self._observation_matrix


# ==================================================================================================
class VectorMisfit(hl.Misfit):
    """Summary."""

    # ----------------------------------------------------------------------------------------------
    def __init__(self, misfit_list: Iterable[hl.Misfit], function_space: dl.FunctionSpace) -> None:
        """Constructor, initialize list of misfits and buffers.

        Args:
            misfit_list (Iterable[hl.Misfit]): List of misfit objects for each component.
            function_space (dl.FunctionSpace): Vector function space of all components.

        Raises:
            ValueError: Checks that number of misfits matches number of components in function
                space.
        """
        if not len(misfit_list) == function_space.num_sub_spaces():
            raise ValueError(
                f"Number of misfits ({len(misfit_list)}) must match "
                f"number of components ({function_space.num_sub_spaces()}) in the function space."
            )
        self._misfit_list = misfit_list
        self._function_space = function_space
        self._input_component_buffer, self._output_component_buffer = self._create_buffers(
            function_space
        )

    # ----------------------------------------------------------------------------------------------
    def cost(
        self,
        state_list: Iterable[
            dl.Vector | dl.PETScVector,
            dl.Vector | dl.PETScVector | None,
            dl.Vector | dl.PETScVector | None,
        ],
    ) -> float:
        """_summary_.

        Args:
            state_list (Iterable): _description_

        Returns:
            float: _description_
        """
        forward_vector, _, _ = state_list
        self._input_component_buffer = fex_converter.extract_components(
            forward_vector, self._input_component_buffer, self._function_space
        )
        cost = 0.0
        for i, misfit in enumerate(self._misfit_list):
            cost += misfit.cost([self._input_component_buffer[i], None, None])
        return cost

    # ----------------------------------------------------------------------------------------------
    def grad(
        self,
        derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        state_list: Iterable[
            dl.Vector | dl.PETScVector,
            dl.Vector | dl.PETScVector | None,
            dl.Vector | dl.PETScVector | None,
        ],
        output_vector: dl.Vector | dl.PETScVector,
    ) -> None:
        """_summary_.

        Args:
            derivative_type (Annotated[int, IsEqual[hl.STATE]  |  IsEqual[hl.PARAMETER]]):
                _description_
            state_list (Iterable): _description_
            output_vector (dl.Vector | dl.PETScVector): _description_
        """
        forward_vector, _, _ = state_list
        self._input_component_buffer = fex_converter.extract_components(
            forward_vector, self._input_component_buffer, self._function_space
        )
        for i, misfit in enumerate(self._misfit_list):
            misfit.grad(
                derivative_type,
                [self._input_component_buffer[i], None, None],
                self._output_component_buffer[i],
            )
        output_vector = fex_converter.combine_components(
            self._output_component_buffer, output_vector, self._function_space
        )

    # ----------------------------------------------------------------------------------------------
    def setLinearizationPoint(  # noqa: N802
        self,
        _state_list: Iterable[
            dl.Vector | dl.PETScVector | None,
            dl.Vector | dl.PETScVector | None,
            dl.Vector | dl.PETScVector | None,
        ],
        _gauss_newton_approx: bool,
    ) -> None:
        """Summary."""
        pass

    # ----------------------------------------------------------------------------------------------
    def apply_ij(
        self,
        first_derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        second_derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        hvp_direction: dl.Vector | dl.PETScVector,
        output_vector: dl.Vector | dl.PETScVector,
    ) -> None:
        """_summary_.

        Args:
            first_derivative_type (Annotated[int, IsEqual[hl.STATE]  |  IsEqual[hl.PARAMETER]]):
                _description_
            second_derivative_type (Annotated[int, IsEqual[hl.STATE]  |  IsEqual[hl.PARAMETER]]):
                _description_
            hvp_direction (dl.Vector | dl.PETScVector): _description_
            output_vector (dl.Vector | dl.PETScVector): _description_
        """
        if first_derivative_type == hl.STATE and second_derivative_type == hl.STATE:
            self._input_component_buffer = fex_converter.extract_components(
                hvp_direction, self._input_component_buffer, self._function_space
            )
            for i, misfit in enumerate(self._misfit_list):
                misfit.apply_ij(
                    hl.STATE,
                    hl.STATE,
                    self._input_component_buffer[i],
                    self._output_component_buffer[i],
                )
            output_vector = fex_converter.combine_components(
                self._output_component_buffer, output_vector, self._function_space
            )
        else:
            output_vector.zero()

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _create_buffers(
        function_space: dl.FunctionSpace,
    ) -> tuple[Iterable[dl.Vector], Iterable[dl.Vector]]:
        input_component_buffer = []
        output_component_buffer = []
        num_sub_spaces = function_space.num_sub_spaces()
        for i in range(num_sub_spaces):
            input_component = dl.Vector()
            output_component = dl.Vector()
            input_component.init(function_space.sub(i).dim())
            output_component.init(function_space.sub(i).dim())
            input_component_buffer.append(input_component)
            output_component_buffer.append(output_component)

        return input_component_buffer, output_component_buffer


# ==================================================================================================
class TDMisfit(hl.Misfit):
    """Time-dependent misfit.

    !!! warning
        This class is not implemented yet.
    """

    def __init__(
        self, stationary_misfits: Iterable[hl.Misfit], observation_times: Iterable[Real]
    ) -> None:
        """Constructor (raises error)."""
        raise NotImplementedError("Time-dependent problems are not implemented yet.")


# ==================================================================================================
@dataclass
class MisfitSettings:
    """_summary_.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
    """
    function_space: dl.FunctionSpace
    observation_points: npt.NDArray[np.floating] | Iterable[npt.NDArray[np.floating]]
    observation_values: (
        npt.NDArray[np.floating]
        | Iterable[npt.NDArray[np.floating]]
        | Iterable[Iterable[npt.NDArray[np.floating]]]
    )
    noise_variance: npt.NDArray[np.floating] | Iterable[npt.NDArray[np.floating]]
    stationary: bool = True
    observation_times: npt.NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Post init."""
        if self.function_space.num_sub_spaces() == 0 and not (
            isinstance(self.observation_points, np.ndarray)
            and isinstance(self.observation_values, np.ndarray)
            and isinstance(self.noise_variance, np.ndarray)
        ):
            raise ValueError(
                "The observation points, values, and noise variance must be "
                "single arrays if the function space is scalar."
            )
        if self.function_space.num_sub_spaces() > 0 and not (
            isinstance(self.observation_points, Iterable)
            and isinstance(self.observation_values, Iterable)
            and isinstance(self.noise_variance, Iterable)
            and len(self.observation_points)
            == len(self.observation_values)
            == len(self.noise_variance)
        ):
            raise ValueError(
                "The observation points, values, and noise variance must be "
                "iterables if the function space has multiple components."
            )
        if not self.stationary and self.observation_times is None:
            raise ValueError("Observation times must be provided for a time-dependent misfit.")


# --------------------------------------------------------------------------------------------------
@dataclass
class Misfit:
    """_summary_.

    bla
    """
    hippylib_misfit: hl.Misfit
    noise_precision_matrix: sp.sparse.coo_array | Iterable[sp.sparse.coo_array]
    observation_matrix: sp.sparse.coo_array | Iterable[sp.sparse.coo_array]


# ==================================================================================================
class MisfitBuilder:
    """_summary_.

    Returns:
        _type_: _description_
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: MisfitSettings) -> None:
        """_summary_.

        Args:
            settings (MisfitSettings): _description_
        """
        self._function_space = settings.function_space
        self._observation_points = settings.observation_points
        self._observation_values = settings.observation_values
        self._noise_variance = settings.noise_variance
        self._stationary = settings.stationary
        self._observation_times = settings.observation_times
        self._num_components = self._function_space.num_sub_spaces()

    # ----------------------------------------------------------------------------------------------
    def build(self) -> Misfit:
        """_summary_.

        Returns:
            Misfit: _description_
        """
        self._observation_matrices = self._assemble_observation_matrices()
        self._noise_precision_matrices = self._assemble_noise_precision_matrices()
        misfit = self._build_misfit()
        return misfit

    # ----------------------------------------------------------------------------------------------
    def _assemble_observation_matrices(self) -> dl.Matrix | list[dl.Matrix]:
        """_summary_.

        Returns:
            dl.Matrix | list[dl.Matrix]: _description_
        """
        if self._num_components == 0:
            observation_matrices = assemble_pointwise_observation_operator(
                self._function_space, self._observation_points
            )
        else:
            observation_matrices = []
            for i in range(self._num_components):
                component_observation_matrix = assemble_pointwise_observation_operator(
                    self._function_space.sub(i).collapse(), self._observation_points[i]
                )
                observation_matrices.append(component_observation_matrix)
        return observation_matrices

    def _convert_matrices_to_scipy(
        self,
        matrices: dl.Matrix | dl.PETScMatrix | Iterable[dl.Matrix | dl.PETScMatrix],
    ) -> sp.sparse.coo_array | Iterable[sp.sparse.coo_array]:
        """_summary_.

        Args:
            matrices (dl.Matrix | dl.PETScMatrix | Iterable[dl.Matrix  |  dl.PETScMatrix]):
                _description_

        Returns:
            sp.sparse.coo_array | Iterable[sp.sparse.coo_array]: _description_
        """
        if self._num_components == 0:
            scipy_matrices = fex_converter.convert_matrix_to_scipy(matrices)
        else:
            scipy_matrices = []
            for matrix in matrices:
                scipy_matrix = fex_converter.convert_matrix_to_scipy(matrix)
                scipy_matrices.append(scipy_matrix)
        return scipy_matrices

    # ----------------------------------------------------------------------------------------------
    def _assemble_noise_precision_matrices(self) -> dl.PETScMatrix | list[dl.PETScMatrix]:
        """_summary_.

        Returns:
            dl.PETScMatrix | list[dl.PETScMatrix]: _description_
        """
        if self._num_components == 0:
            noise_precision_matrices = assemble_noise_precision_matrix(self._noise_variance)
        else:
            noise_precision_matrices = []
            for component_noise_variance in self._noise_variance:
                component_noise_precision_matrix = assemble_noise_precision_matrix(
                    component_noise_variance
                )
                noise_precision_matrices.append(component_noise_precision_matrix)
        return noise_precision_matrices

    # ----------------------------------------------------------------------------------------------
    def _build_misfit(self) -> Misfit:
        """_summary_.

        Returns:
            Misfit: _description_
        """
        # Single component, stationary
        if self._num_components == 0 and self._stationary:
            hl_misfit = DiscreteMisfit(
                self._observation_values, self._observation_matrices, self._noise_precision_matrices
            )
        # Single component, time-dependent
        if self._num_components == 0 and not self._stationary:
            misfit_list = []
            for time_observation_values in self._observation_values:
                misfit = DiscreteMisfit(
                    time_observation_values,
                    self._observation_matrices,
                    self._noise_precision_matrices,
                )
                misfit_list.append(misfit)
            hl_misfit = TDMisfit(misfit_list, self._observation_times)
        # Multiple components, stationary
        if self._num_components > 0 and self._stationary:
            misfit_list = []
            for component_observation_values, observation_matrix, noise_precision_matrix in zip(
                self._observation_values,
                self._observation_matrices,
                self._noise_precision_matrices,
                strict=True,
            ):
                misfit = DiscreteMisfit(
                    component_observation_values, observation_matrix, noise_precision_matrix
                )
                misfit_list.append(misfit)
            hl_misfit = VectorMisfit(misfit_list, self._function_space)
        # Multiple components, time-dependent
        if self._num_components > 0 and not self._stationary:
            misfit_list = []
            for time_observation_values in self._observation_values:
                time_misfit_list = []
                for component_observation_values, observation_matrix, noise_precision_matrix in zip(
                    time_observation_values,
                    self._observation_matrices,
                    self._noise_precision_matrices,
                    strict=True,
                ):
                    misfit = DiscreteMisfit(
                        component_observation_values, observation_matrix, noise_precision_matrix
                    )
                    time_misfit_list.append(misfit)
                time_misfit = VectorMisfit(time_misfit_list, self._function_space)
                misfit_list.append(time_misfit)
            hl_misfit = TDMisfit(misfit_list, self._observation_times)

        noise_precision_sparse = self._convert_matrices_to_scipy(self._noise_precision_matrices)
        observation_sparse = self._convert_matrices_to_scipy(self._observation_matrices)
        misfit = Misfit(hl_misfit, noise_precision_sparse, observation_sparse)
        return misfit
