"""Interpolation module

This module interpolates/projects FEM data in space and time. Its capabilities are used for data
generation, in that they convert the results of FEM simulations to arbitrary points & times.

Classes:
--------
InterpolationHandle: Converter in space and time
"""

# ====================================== Preliminary Commands =======================================
import warnings
import numpy as np
from typing import Tuple, Union

from . import general as utils

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl


# ======================================== Interpolation Class ======================================
class InterpolationHandle:
    """Converts data points between different locations in space and time
    Conversion in time is a simple linear interpolation. The projection in space is slightly
    different, in the sense that it takes into account the FEM nature of the data.

    Methods:
    --------
    interpolate_in_time: Interpolates data to alternative time points
    project_in_space: Projects FEM data to alternative locations in space
    """

    # -----------------------------------------------------------------------------------------------
    def __init__(
        self,
        simTimes: np.ndarray,
        obsTimes: Union[int, float, np.ndarray],
        obsPoints: Union[int, float, np.ndarray],
        funcSpace: fe.FunctionSpace,
    ) -> None:
        """Constructor

        Sets up interpolation and projection mapping.

        Args:
            simTimes (np.ndarray): Simulation time array
            obsTimes (np.ndarray): Observation time array
            obsPoints (np.ndarray): Observation point location array
            funcSpace (fe.FunctionSpace): Function space of the FEM variable (gives FEM locations)

        Raises:
            TypeError: Checks type of simulation and observation time array
            ValueError: Checks that simulation times define an actual range
            ValueError: Checks that simulation time array is sorted
            ValueError: Checks that observation times lie within simulation times
            TypeError: Checks type of the provided function space
        """

        if not all(
            isinstance(timeStruct, (int, float, np.ndarray))
            for timeStruct in [obsTimes, simTimes]
        ):
            raise TypeError(
                "Time structures need to be given as numbers or numpy arrays."
            )
        if not simTimes.ndim > 0:
            raise ValueError("Simulation time array must contain more than one value")
        if not np.all(np.diff(simTimes) > 0):
            raise ValueError("Time arrays need to be sorted.")
        if not np.all(
            (obsTimes >= np.amin(simTimes)) & (obsTimes <= np.amax(simTimes))
        ):
            raise ValueError("Observation times need to lie within simulation times.")
        if not isinstance(funcSpace, fe.FunctionSpace):
            raise TypeError("Function Space needs to be valid FEniCS object.")

        obsTimes = utils.process_input_data(obsTimes)
        obsPoints = utils.process_input_data(obsPoints)

        self._simTimes = simTimes
        self._obsTimes = obsTimes
        self._obsTimeInds = np.indices((obsTimes.size,)).flatten()
        self._timeInterpData = self._assemble_interpolation_data()
        self._projMat = hl.pointwiseObservation.assemblePointwiseObservation(
            funcSpace, obsPoints
        )

    # -----------------------------------------------------------------------------------------------
    def interpolate_in_time(
        self, simVec: hl.TimeDependentVector
    ) -> hl.TimeDependentVector:
        """Interpolates linearly in time

        For every new data point, its values are determined via the previously assembled
        interpolation mapping

        Args:
            simVec (hl.TimeDependentVector): Result vector from FEM simulation

        Raises:
            TypeError: Checks input type
            ValueError: Checks that input is defined over simulation times

        Returns:
            hl.TimeDependentVector: Interpolated solution at observation times
        """

        if not isinstance(simVec, hl.TimeDependentVector):
            raise TypeError("Input needs to be hIPPYlib tdv.")
        if not np.array_equal(self._simTimes, simVec.times):
            raise ValueError(
                "Simulation times and times for construction of time interpolation "
                "need to be equal."
            )
        interpVec = hl.TimeDependentVector(self._obsTimes)
        interpVec.initialize(self._projMat, 1)

        for i, interpPoint in enumerate(self._timeInterpData):
            (obsInd, interpCoeff) = interpPoint
            lowerBoundVec = simVec.data[obsInd - 1]
            upperBoundVec = simVec.data[obsInd]
            projPoint = (1 - interpCoeff) * lowerBoundVec + interpCoeff * upperBoundVec
            interpVec.data[i].axpy(1, projPoint)

        return interpVec

    # -----------------------------------------------------------------------------------------------
    def project_in_space(
        self, interpVec: hl.TimeDependentVector
    ) -> hl.TimeDependentVector:
        """Projects FEM points to alternative locations in space

        The projection takes into account the definition of the FEM data w.r.t. to shape functions.
        In that respect, the result might differ from simple interpolation.

        Args:
            interpVec (hl.TimeDependentVector): Vector that is already interpolated in time

        Raises:
            ValueError: Checks that input is defined over observation times

        Returns:
            hl.TimeDependentVector: Data interpolated in time and projected to observation points
        """

        if not np.array_equal(self._obsTimes, interpVec.times):
            raise ValueError(
                "Observation times and times for construction of space projection "
                "need to be equal."
            )
        projVec = hl.TimeDependentVector(self._obsTimes)
        projVec.initialize(self._projMat, 0)

        for i in self._obsTimeInds:
            self._projMat.mult(interpVec.data[i], projVec.data[i])

        return projVec

    # -----------------------------------------------------------------------------------------------
    def interpolate_and_project(
        self, simVec: hl.TimeDependentVector
    ) -> hl.TimeDependentVector:
        """Wrapper method for interpolation and projection

        Args:
            simVec (hl.TimeDependentVector): Simulation result

        Returns:
            hl.TimeDependentVector: Interpolated and projected vector in space and time
        """

        interpVec = self.interpolate_in_time(simVec)
        projVec = self.project_in_space(interpVec)

        return projVec

    # -----------------------------------------------------------------------------------------------
    def _assemble_interpolation_data(self) -> list[Tuple[int, float]]:
        """Construct mapping for linear interpolation in time"""

        interpData = []
        for t in self._obsTimes:
            if t == self._simTimes[-1]:
                obsInd = -1
                interpCoeff = 1
            else:
                obsInd = np.where(self._simTimes >= t)[0][0]
                interpCoeff = (t - self._simTimes[obsInd - 1]) / (
                    self._simTimes[obsInd] - self._simTimes[obsInd - 1]
                )
            interpData.append((obsInd, interpCoeff))

        return interpData
