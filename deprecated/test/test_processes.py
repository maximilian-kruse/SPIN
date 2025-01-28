"""Tests of the stochastic processes module"""

# ====================================== Preliminary Commands =======================================
import pytest
import numpy as np
from typing import Union

import sp_inference.processes as sp


# ============================================== Data ===============================================


class TestData:
    feSettings = {
        "num_mesh_points": 100,
        "boundary_locations": [-5, 5],
        "boundary_values": [0, 0],
        "element_degrees": [1, 1],
    }

    transientSettings = {
        "start_time": 0.1,
        "end_time": 5.1,
        "time_step_size": 0.01,
        "initial_condition": lambda x: np.exp(-np.square(x)),
    }

    processTypes = sp.get_option_list()
    points = [0.5, np.array([-1, 0, 1])]
    times = [2, np.array([1, 2.5, 4])]

    metDomainBounds = [-2, 2]
    dataStd = 0.1
    dataSeed = 0


# ============================================ Fixtures =============================================


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=TestData.processTypes)
def process_setup(request):
    processType = request.param
    process = sp.get_process(processType)()
    return process


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=TestData.points)
def point_setup(request):
    return request.param


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=TestData.times)
def time_setup(request):
    return request.param


# =================================== Stochastic Processes Tests ====================================


# ---------------------------------------------------------------------------------------------------
def test_process_generation(process_setup: sp.BaseProcess) -> None:
    assert isinstance(process_setup, sp.BaseProcess)


# ---------------------------------------------------------------------------------------------------
def test_drift_diffusion(
    process_setup: sp.BaseProcess, point_setup: Union[int, float, np.ndarray]
) -> None:
    testProcess = process_setup
    testPoint = point_setup
    driftValue = testProcess.compute_drift(testPoint)
    diffusionValue = testProcess.compute_squared_diffusion(testPoint)

    _test_stationary_shape(testPoint, driftValue)
    _test_stationary_shape(testPoint, diffusionValue)


# ---------------------------------------------------------------------------------------------------
def test_stationary_distribution(
    process_setup: sp.BaseProcess, point_setup: Union[int, float, np.ndarray]
) -> None:
    testProcess = process_setup
    testPoint = point_setup
    statDistr = testProcess.compute_stationary_distribution(testPoint)

    _test_stationary_shape(testPoint, statDistr)


# ---------------------------------------------------------------------------------------------------
def test_mean_exit_time(
    process_setup: sp.BaseProcess, point_setup: Union[int, float, np.ndarray]
) -> None:
    testProcess = process_setup
    testPoint = point_setup
    met = testProcess.compute_mean_exit_time(testPoint, TestData.metDomainBounds)

    _test_stationary_shape(testPoint, met)


# ---------------------------------------------------------------------------------------------------
def test_transient_distribution(process_setup: sp.BaseProcess) -> None:
    testProcess = process_setup
    timePoints, pdfValues, funcSpace = testProcess.compute_transient_distribution_fem(
        TestData.feSettings, TestData.transientSettings
    )
    spacePoints = funcSpace.tabulate_dof_coordinates().flatten()
    _test_transient_shape(spacePoints, timePoints, pdfValues)


# ---------------------------------------------------------------------------------------------------
def test_data_stationary_fpe(
    process_setup: sp.BaseProcess, point_setup: Union[int, float, np.ndarray]
) -> None:
    testProcess = process_setup
    testPoint = point_setup
    perturbedValues, _ = testProcess.generate_data_stationary_fpe(
        testPoint, TestData.dataStd, TestData.dataSeed
    )

    _test_stationary_shape(testPoint, perturbedValues)


# ---------------------------------------------------------------------------------------------------
def test_data_mean_exit_time(
    process_setup: sp.BaseProcess, point_setup: Union[int, float, np.ndarray]
) -> None:
    testProcess = process_setup
    testPoint = point_setup
    perturbedValues, _ = testProcess.generate_data_mean_exit_time(
        testPoint, TestData.dataStd, TestData.dataSeed, TestData.metDomainBounds
    )

    _test_stationary_shape(testPoint, perturbedValues)


# ---------------------------------------------------------------------------------------------------
def test_data_transient_fpe(
    process_setup: sp.BaseProcess,
    point_setup: Union[int, float, np.ndarray],
    time_setup: Union[int, float, np.ndarray],
) -> None:
    testProcess = process_setup
    testPoint = point_setup
    testTime = time_setup
    perturbedValues, _ = testProcess.generate_data_transient_fpe(
        testPoint,
        testTime,
        TestData.dataStd,
        TestData.dataSeed,
        TestData.feSettings,
        TestData.transientSettings,
    )
    _test_transient_shape(testPoint, testTime, perturbedValues)


# ======================================== Utility Functions ========================================


# ---------------------------------------------------------------------------------------------------
def _test_stationary_shape(
    domainPoints: Union[int, float, np.ndarray], output: Union[int, float, np.ndarray]
) -> None:
    if isinstance(domainPoints, (int, float)):
        assert isinstance(output, float)
    if isinstance(domainPoints, np.ndarray):
        assert isinstance(output, np.ndarray) and output.shape == domainPoints.shape


# ---------------------------------------------------------------------------------------------------
def _test_transient_shape(
    domainPoints: Union[int, float, np.ndarray],
    timePoints: Union[int, float, np.ndarray],
    output: Union[int, float, np.ndarray],
) -> None:
    if isinstance(domainPoints, (int, float)) and isinstance(timePoints, (int, float)):
        assert isinstance(output, float)
    elif isinstance(domainPoints, np.ndarray) and isinstance(timePoints, (int, float)):
        assert isinstance(output, np.ndarray) and output.shape == domainPoints.shape
    elif isinstance(domainPoints, (int, float)) and isinstance(timePoints, np.ndarray):
        assert isinstance(output, np.ndarray) and output.shape == timePoints.shape
    else:
        assert (
            isinstance(output, np.ndarray)
            and output.shape == timePoints.shape + domainPoints.shape
        )
