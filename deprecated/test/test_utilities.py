"""Tests of the utilities sub-package"""

# ====================================== Preliminary Commands =======================================
import warnings
import pytest
import numpy as np
from typing import Tuple

import sp_inference.utilities.general as utils
import sp_inference.utilities.interpolation as interpolation

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl


# ============================================== Data ===============================================


class TestData:
    lowerDomainBound = 0
    upperDomainBound = 2
    numMeshPoints = 10
    femElemDegree = 1

    simTimes = np.array([1, 2, 3])
    obsTimes = np.array([1.5, 2.5])
    obsPoints = np.array([0.5, 1.5])
    interpValues = np.array(([0.75, 2.25], [1.25, 3.75]))


# ============================================ Fixtures =============================================


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data_arr_vec() -> Tuple:
    mesh = fe.IntervalMesh(
        TestData.numMeshPoints, TestData.lowerDomainBound, TestData.upperDomainBound
    )
    funcSpace = fe.FunctionSpace(mesh, "Lagrange", TestData.femElemDegree)
    feVec = fe.Function(funcSpace).vector()
    npArray = np.linspace(
        TestData.lowerDomainBound, TestData.upperDomainBound, TestData.numMeshPoints + 1
    )
    feVec.set_local(npArray)

    return npArray, feVec, funcSpace


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data_arr_func(data_arr_vec: Tuple) -> Tuple:
    npArray, feVec, funcSpace = data_arr_vec
    feFunc = hl.vector2Function(feVec, funcSpace)

    return npArray, feFunc, funcSpace


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data_arr_tdv(data_arr_vec: Tuple) -> Tuple:
    singleArray, feVec, _ = data_arr_vec
    timeSeriesArray = np.ndarray((len(TestData.simTimes), len(singleArray)))
    tdVec = hl.TimeDependentVector(TestData.simTimes)

    for i, _ in enumerate(TestData.simTimes):
        timeSeriesArray[i, :] = singleArray
        tdVec.data[i] = fe.Vector(feVec)

    return TestData.simTimes, timeSeriesArray, tdVec


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data_pyfunc_fefunc() -> Tuple:
    mesh = fe.IntervalMesh(
        TestData.numMeshPoints, TestData.lowerDomainBound, TestData.upperDomainBound
    )

    funcSpaceScalar = fe.FunctionSpace(mesh, "Lagrange", TestData.femElemDegree)
    funcSpaceVec1D = fe.VectorFunctionSpace(mesh, "Lagrange", TestData.femElemDegree, dim=1)
    funcSpaceVec2D = fe.VectorFunctionSpace(mesh, "Lagrange", TestData.femElemDegree, dim=2)
    funcSpaceTensor1D = fe.TensorFunctionSpace(
        mesh, "Lagrange", TestData.femElemDegree, shape=(1, 1), symmetry=True
    )
    funcSpaceTensor2D = fe.TensorFunctionSpace(
        mesh, "Lagrange", TestData.femElemDegree, shape=(2, 2), symmetry=True
    )

    dummyFuncScalar = lambda x: np.square(x)
    dummyFuncVec2D = [dummyFuncScalar, dummyFuncScalar]
    dummyFuncTensor2D = [dummyFuncScalar, dummyFuncScalar, dummyFuncScalar]

    gridPointsScalar = funcSpaceScalar.tabulate_dof_coordinates().flatten()
    gridPointsVec2D = funcSpaceVec2D.tabulate_dof_coordinates().flatten()
    gridPointsTensor2D = funcSpaceTensor2D.tabulate_dof_coordinates().flatten()

    pyFuncData = [
        (dummyFuncScalar, funcSpaceScalar, gridPointsScalar),
        (dummyFuncScalar, funcSpaceVec1D, gridPointsScalar),
        (dummyFuncVec2D, funcSpaceVec2D, gridPointsVec2D),
        (dummyFuncScalar, funcSpaceTensor1D, gridPointsScalar),
        (dummyFuncTensor2D, funcSpaceTensor2D, gridPointsTensor2D),
    ]

    return pyFuncData


# ========================================= Utility Tests ===========================================


# ---------------------------------------------------------------------------------------------------
def test_nparray_to_fevec(data_arr_vec: Tuple) -> None:
    npArray, feVec, _ = data_arr_vec
    convertedVec = utils.nparray_to_fevec(npArray)

    assert isinstance(feVec, fe.GenericVector)
    assert np.all(feVec == convertedVec)


# ---------------------------------------------------------------------------------------------------
def test_nparray_to_fefunc(data_arr_func: Tuple) -> None:
    npArray, feFunc, funcSpace = data_arr_func
    convertedFunc = utils.nparray_to_fefunc(npArray, funcSpace)

    assert isinstance(feFunc, fe.Function)
    assert np.all(convertedFunc.vector().get_local() == feFunc.vector().get_local())


# ---------------------------------------------------------------------------------------------------
def test_nparray_to_tdv(data_arr_tdv: Tuple) -> None:
    timePoints, timeSeriesArray, tdVec = data_arr_tdv
    convertedTdv = utils.nparray_to_tdv(timePoints, timeSeriesArray)

    assert isinstance(convertedTdv, hl.TimeDependentVector)
    for i, data in enumerate(convertedTdv.data):
        assert np.all(data == tdVec.data[i])


# ---------------------------------------------------------------------------------------------------
def test_tdv_to_nparray(data_arr_tdv: Tuple) -> None:
    _, timeSeriesArray, tdVec = data_arr_tdv
    tdArray = utils.tdv_to_nparray(tdVec)

    assert isinstance(tdArray, np.ndarray)
    assert np.array_equal(tdArray, timeSeriesArray)


# ---------------------------------------------------------------------------------------------------
def test_pyfunc_to_feFunc(data_pyfunc_fefunc: Tuple) -> None:
    for data in data_pyfunc_fefunc:
        dummyFunc, funcSpace, gridPoints = data
        feFunc = utils.pyfunc_to_fefunc(dummyFunc, funcSpace)
        if callable(dummyFunc):
            funcValues = dummyFunc(gridPoints)
        else:
            funcValues = dummyFunc[0](gridPoints)

        assert isinstance(feFunc, fe.Function)
        assert np.array_equal(feFunc.vector().get_local(), funcValues)


# ---------------------------------------------------------------------------------------------------
def test_interpolation(data_arr_vec: Tuple) -> None:
    _, _, funcSpace = data_arr_vec
    massMatrix = fe.assemble(fe.TrialFunction(funcSpace) * fe.TestFunction(funcSpace) * fe.dx)

    simVec = hl.TimeDependentVector(TestData.simTimes)
    simVec.initialize(massMatrix, 1)
    for i, t in enumerate(TestData.simTimes):
        funcHandle = lambda x: t * x
        currVec = utils.pyfunc_to_fevec(funcHandle, funcSpace)
        simVec.data[i].axpy(1, currVec)

    interpHandle = interpolation.InterpolationHandle(
        TestData.simTimes, TestData.obsTimes, TestData.obsPoints, funcSpace
    )
    projVec = interpHandle.interpolate_and_project(simVec)

    assert isinstance(projVec, hl.TimeDependentVector)
    assert np.array_equal(projVec.times, TestData.obsTimes)
    for i, _ in enumerate(TestData.obsTimes):
        assert np.array_equal(projVec.data[i].get_local(), TestData.interpValues[i, :])
