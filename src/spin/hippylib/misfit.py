from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import dolfin as dl
import hippylib as hl
import numpy as np
import numpy.typing as npt
from beartype.vale import IsEqual
from petsc4py import PETSc

from spin.fenics import converter as fex_converter


# ==================================================================================================
def assemble_pointwise_observation_operator(
    function_space: dl.FunctionSpace, observation_points: npt.NDArray[np.floating]
) -> dl.Matrix:
    observation_matrix = hl.assemblePointwiseObservation(function_space, observation_points)
    return observation_matrix


# --------------------------------------------------------------------------------------------------
def assemble_noise_precision_matrix(noise_variance: npt.NDArray[np.floating]) -> dl.PETScMatrix:
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
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        data: npt.NDArray[np.floating],
        observation_matrix: dl.Matrix,
        noise_precision_matrix: dl.PETScMatrix,
    ):
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
        forward_vector = state_list[hl.STATE]
        self._observation_matrix.mult(forward_vector, self._vector_buffer_one)
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
        pass

    # ----------------------------------------------------------------------------------------------
    def apply_ij(
        self,
        first_derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        second_derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        hvp_direction: dl.Vector | dl.PETScVector,
        output_vector: dl.Vector | dl.PETScVector,
    ) -> None:
        if first_derivative_type == hl.STATE and second_derivative_type == hl.STATE:
            self._observation_matrix.mult(hvp_direction, self._vector_buffer_one)
            self._noise_precision_matrix.mult(self._vector_buffer_one, self._vector_buffer_two)
            self._observation_matrix.transpmult(self._vector_buffer_two, output_vector)
        else:
            output_vector.zero()

    # ----------------------------------------------------------------------------------------------
    @property
    def observation_matrix(self) -> dl.Matrix:
        return self._observation_matrix


# ==================================================================================================
class VectorMisfit(hl.Misfit):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, misfit_list: Iterable[hl.Misfit], function_space: dl.FunctionSpace) -> None:
        self._misfit_list = misfit_list
        self._function_space = function_space
        self._input_component_buffer, self._output_component_buffer = (
            self._create_buffers(function_space)
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
        pass

    # ----------------------------------------------------------------------------------------------
    def apply_ij(
        self,
        first_derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        second_derivative_type: Annotated[int, IsEqual[hl.STATE] | IsEqual[hl.PARAMETER]],
        hvp_direction: dl.Vector | dl.PETScVector,
        output_vector: dl.Vector | dl.PETScVector,
    ) -> None:
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
    def __init__(
        self, stationary_misfits: Iterable[hl.Misfit], observation_times: Iterable[Real]
    ) -> None:
        pass


# ==================================================================================================
@dataclass
class MisfitSettings:
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


# ==================================================================================================
class MisfitBuilder:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: MisfitSettings) -> None:
        self._function_space = settings.function_space
        self._observation_points = settings.observation_points
        self._observation_values = settings.observation_values
        self._noise_variance = settings.noise_variance
        self._stationary = settings.stationary
        self._observation_times = settings.observation_times
        self._num_components = self._function_space.num_sub_spaces()

    # ----------------------------------------------------------------------------------------------
    def build(self) -> hl.Misfit:
        self._observation_matrices = self._assemble_observation_matrices()
        self._noise_precision_matrices = self._assemble_noise_precision_matrices()
        misfit = self._build_misfit()
        return misfit

    # ----------------------------------------------------------------------------------------------
    def _assemble_observation_matrices(self) -> dl.Matrix | list[dl.Matrix]:
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

    # ----------------------------------------------------------------------------------------------
    def _assemble_noise_precision_matrices(self) -> dl.PETScMatrix | list[dl.PETScMatrix]:
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
    def _build_misfit(self) -> hl.Misfit:
        # Single component, stationary
        if self._num_components == 0 and self._stationary:
            misfit = DiscreteMisfit(
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
            misfit = TDMisfit(misfit_list, self._observation_times)
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
            misfit = VectorMisfit(misfit_list, self._function_space)
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
            misfit = TDMisfit(misfit_list, self._observation_times)

        return misfit
