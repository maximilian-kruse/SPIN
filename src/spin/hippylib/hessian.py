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
class LowRankHessianSettings:
    num_eigenvalues: Annotated[int, Is[lambda x: x > 0]]
    num_oversampling: Annotated[int, Is[lambda x: x > 0]]
    inference_model: hl.Model
    evaluation_point: Iterable[npt.NDArray[np.floating]]
    gauss_newton_approximation: bool = False


# --------------------------------------------------------------------------------------------------
def compute_low_rank_hessian(
    settings: LowRankHessianSettings,
) -> tuple[npt.NDArray[np.floating], Iterable[npt.NDArray[np.floating]]]:
    function_spaces = settings.inference_model.problem.Vh
    evaluation_point_vectors = []
    for array, function_space in zip(settings.evaluation_point, function_spaces, strict=True):
        evaluation_point_vectors.append(
            fex_converter.convert_to_dolfin(array, function_space).vector()
        )
    settings.inference_model.setPointForHessianEvaluations(
        evaluation_point_vectors, settings.gauss_newton_approximation
    )
    misfit_hessian = hl.ReducedHessian(settings.inference_model, misfit_only=True)
    random_multivector = hl.MultiVector(
        evaluation_point_vectors[1], settings.num_eigenvalues + settings.num_oversampling
    )
    hl.parRandom.normal(1.0, random_multivector)
    eigenvalues, eigenvectors = hl.doublePassG(
        misfit_hessian,
        settings.inference_model.prior.R,
        settings.inference_model.prior.Rsolver,
        random_multivector,
        settings.num_eigenvalues,
    )
    eigenvectors = fex_converter.convert_multivector_to_numpy(eigenvectors, function_spaces[1])
    return eigenvalues, eigenvectors
