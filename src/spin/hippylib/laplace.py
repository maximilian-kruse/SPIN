from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated

import dolfin as dl
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


class LowRankLaplaceApproximation:
    def __init__(self, settings: LowRankLaplaceApproximationSettings):
        self._mean = settings.mean
        low_rank_hessian_eigenvectors = fex_converter.convert_to_multivector(
            settings.low_rank_hessian_eigenvectors, settings.inference_model.problem.Vh[1]
        )
        self._laplace_approximation = hl.GaussianLRPosterior(
            settings.inference_model.prior,
            settings.low_rank_hessian_eigenvalues,
            low_rank_hessian_eigenvectors,
        )

    def compute_pointwise_variance(self) -> npt.NDArray[np.floating]:
        pass
