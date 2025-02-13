"""Wrapper for low-rank Hessian approximation in Hippylib.

This moduleprovides a wrappe for the corresponding functionality in Hippylib. Configuration is done
via dataclasses, all input and output vectors are numpy arrays.

Internally, the module uses the double-pass algorithm in Hippylib, to compute a randomized truncated
SVD of the Hessian operator of the provided Hippylib inference model.

Classes:
    LowRankHessianSettings: Configuration for the low-rank computation

Functions:
    compute_low_rank_hessian: Compute the low-rank Hessian approximation of the inference model
        posterior at a given evaluation point.
"""

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
    """Configuration for computation of the low-rank Hessian approximation.

    Attributes:
        num_eigenvalues (int): Number of eigenvalues to compute.
        num_oversampling (int): Number of oversampling eigenvalues (for numerical stability).
        inference_model (hl.Model): Hippylib inference model to use for computations.
        evaluation_point
            (tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]):
                The evaluation point for the Hessian.
        gauss_newton_approximation: Whether to use the Gauss-Newton approximation.
    """

    num_eigenvalues: Annotated[int, Is[lambda x: x > 0]]
    num_oversampling: Annotated[int, Is[lambda x: x > 0]]
    inference_model: hl.Model
    evaluation_point: tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]
    gauss_newton_approximation: bool = False


# --------------------------------------------------------------------------------------------------
def compute_low_rank_hessian(
    settings: LowRankHessianSettings,
) -> tuple[npt.NDArray[np.floating], Iterable[npt.NDArray[np.floating]]]:
    r"""Compute a low-rank Hessian approximation of the inference model posterior.

    The Hessian is computed about the given evaluation point, based on randomized truncated SVD.
    More precisely, we consider the case where the Hessian is the second parametric derivative of a
    negative posterior functional, defined through a Gaussian prior and noise model,

    $$
        \pi_{post}(\mathbf{m}) \propto \exp\Big(
        -\frac{1}{2}||\mathbf{F}(m)-\mathbf{d}_{obs}||_{R_{\text{noise}}}^2
        -\frac{1}{2}||\mathbf{m}-\mathbf{m}_{pr}||_{R_{\text{prior}}}^2\Big).
    $$
    Conesquently, the Hessian can be divided into a misfit and a prior term,

    $$
        \mathbf{H}(\mathbf{m}) = \mathbf{H}_{\text{misfit}}(\mathbf{m}) + \mathbf{R}_{\text{prior}}
        \in \mathbb{R}^{n\times n}.
    $$

    To obtain an approximation of $\mathbf{H}$ at the point $\mathbf{m}$, we solve a generalized
    eigenvalue problem for the first $r$ eigenvalues and eigenvectors of the Hessian,

    $$
        \mathbf{H}_{\text{misfit}}\mathbf{v}_i = \lambda_i\mathbf{R}_{\text{prior}}\mathbf{v}_i,
        \quad \lamba_1m \geq \ldots \geq \lambda_r.
    $$

    The low-rank approximation of the Hessian can then be constructed as

    $$
    \begin{gather*}
        \mathbf{H} = \mathbf{R}_{\text{prior}}
        + \mathbf{R}_{\text{prior}}\mathbf{V}_r\mathbf{D}_r\mathbf{V}_r^T\mathbf{R}_{\text{prior}}
        + \mathcal{O}\big(\sum_{i=r+1}^n\lambda_i\big).
    \end{gather*}
    $$

    The inverse can then be appoximated with the Sherman-Morrison-Woodbury formula.
    Apparently, the approximation is good if the trailing eigenvalues $\lambda_i, i=r+1,\ldots,n$
    are small, corresponding to compactness of the Hessian operator.

    Args:
        settings (LowRankHessianSettings): Configuration for the low-rank computation

    Returns:
        tuple[npt.NDArray[np.floating], Iterable[npt.NDArray[np.floating]]]: Arrays for the
            computed eigenvalues and eigenvectors
    """
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
