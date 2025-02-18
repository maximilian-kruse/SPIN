"""Wrapper for the Gaussian low-rank posterior object in hippylib.

This module provides the functionality for the so-called Laplace approximation of the posterior,
a method of variational inference. The Laplace approximation is based on the linearization of the
forward map that governs the inverse problem, typically about the MAP point. In combination with
a Gaussian prior and likelihood, the Laplace approximation results in a Gaussian posterior, defined
by a mean function (the MAP estimate), and the local Hessian at that point.

The Laplace-approximation relies on the `GaussianLRPosterior` object in Hippylib, which takes a
low-rank approximation of the Hessian as input. This low-rank approximation is computed in SPIN
using the [`compute_low_rank_hessian`][spin.hippylib.hessian.compute_low_rank_hessian] function.
The MAP can be found using the Newton-CG solver in Hippylib, which is wrapped in the
[`NewtonCGSolver`][spin.hippylib.optimization.NewtonCGSolver] class for SPIN applications.

Classes:
    LowRankLaplaceApproximationSettings: Input for the low-rank Laplace approximation object.
    LowRankLaplaceApproximation: Wrapper for the low-rank Laplace approximation object.
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
class LowRankLaplaceApproximationSettings:
    """Input for the low-rank Laplace approximation object.

    Attributes:
        inference_model (hl.Model): Hippylib inference model object.
        mean (npt.NDArray[np.floating]): Mean function (usually the MAP).
        low_rank_hessian_eigenvalues (npt.NDArray[np.floating]): Eigenvalues for low-rank
            approximation of the negative log-posterior Hessian (at the MAP).
        low_rank_hessian_eigenvectors (Iterable[npt.NDArray[np.floating]]): Eigenvectors for
            low-rank approximation of the negative log-posterior Hessian (at the MAP).
    """

    inference_model: hl.Model
    mean: np.ndarray
    low_rank_hessian_eigenvalues: npt.NDArray[np.floating]
    low_rank_hessian_eigenvectors: Iterable[npt.NDArray[np.floating]]


# ==================================================================================================
class LowRankLaplaceApproximation:
    """Low-rank Laplace approximation of the posterior distribution.

    This class provides a wrapper for the `GaussianLRPosterior` object in Hippylib with a more
    modern and user-friendly interface.

    Attributes:
        hippylib_gaussian_posterior (hl.GaussianLRPosterior): Underlying Hippylib object

    Methods:
        compute_pointwise_variance: Compute the pointwise variance for the Laplace approximation.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: LowRankLaplaceApproximationSettings) -> None:
        """Initialize the underlying hippylib object.

        Args:
            settings (LowRankLaplaceApproximationSettings): Input data for the approximation.
        """
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
        method: Annotated[str, Is[lambda x: x in ("Exact", "Randomized")]],
        num_eigenvalues_randomized: Annotated[int, Is[lambda x: x > 0]] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute the pointwise variance field of the laplace approximation.

        For a detailed description of the different methods, we refer to the documentation of the
        [`GaussianLRPosterior`](https://hippylib.readthedocs.io/en/latest/hippylib.modeling.html?highlight=posterior#hippylib.modeling.posterior.GaussianLRPosterior)
        object in Hippylib.

        Args:
            method (str): Algorithm for computation of the variance, options are 'Exact',
                and 'Randomized'.
            num_eigenvalues_randomized (int, optional): Number of dominant eigenvalues for the
                randomized algorithm. Defaults to None.

        Raises:
            ValueError: Checks that `num_eigenvalues_randomized` is provided for the 'Randomized'
                algorithm

        Returns:
            npt.NDArray[np.floating]: Pointwise variance field.
        """
        if method == "Exact":
            variance = self._laplace_approximation.pointwise_variance(method=method)
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
        """Return the underlying `GaussianLRPosterior` Hippylib object."""
        return self._laplace_approximation
