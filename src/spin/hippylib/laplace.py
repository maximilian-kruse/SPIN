from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated

import hippylib as hl
import numpy as np
import numpy.typing as npt
from beartype.vale import Is

from spin.fenics import converter as fex_converter


# ==================================================================================================
@dataclass
class LowRankLaplaceApproximationSettings:
    inference_model: hl.Model
    mean: np.ndarray
    low_rank_hessian_eigenvalues: npt.NDArray[np.floating]
    low_rank_hessian_eigenvectors: Iterable[npt.NDArray[np.floating]]


# ==================================================================================================
class LowRankLaplaceApproximation:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: LowRankLaplaceApproximationSettings):
        self._function_space = settings.inference_model.problem.Vh[1]
        self._mean = settings.mean
        low_rank_hessian_eigenvectors = fex_converter.convert_to_multivector(
            settings.low_rank_hessian_eigenvectors, self._function_space
        )
        mean_vector = fex_converter.convert_to_dolfin(self._mean, self._function_space)
        self._laplace_approximation = hl.GaussianLRPosterior(
            settings.inference_model.prior,
            settings.low_rank_hessian_eigenvalues,
            low_rank_hessian_eigenvectors,
            mean_vector,
        )

    # ----------------------------------------------------------------------------------------------
    def compute_pointwise_variance(
        self,
        method: Annotated[str, Is[lambda x: x in ("Exact", "Estimator", "Randomized")]],
        num_expansion_values_estimator: Annotated[int, Is[lambda x: x > 0]] | None = None,
        num_eigenvalues_randomized: Annotated[int, Is[lambda x: x > 0]] | None = None,
    ) -> npt.NDArray[np.floating]:
        if method == "Exact":
            variance = self._laplace_approximation.pointwise_variance()
        if method == "Estimator":
            if num_expansion_values_estimator is None:
                raise ValueError(
                    "num_expansion_values_estimator must be provided for 'Estimator' method."
                )
            variance = self._laplace_approximation.pointwise_variance(
                method=method, k=num_expansion_values_estimator
            )
        if method == "Randomized":
            if num_eigenvalues_randomized is None:
                raise ValueError(
                    "num_eigenvalues_randomized must be provided for 'Randomized' method."
                )
            variance, *_ = self._laplace_approximation.pointwise_variance(
                method=method, r=num_eigenvalues_randomized
            )
        pointwise_variance = fex_converter.convert_to_numpy(variance, self._function_space)
        return pointwise_variance

    # ----------------------------------------------------------------------------------------------
    @property
    def hippylib_gaussian_posterior(self) -> hl.GaussianLRPosterior:
        return self._laplace_approximation
