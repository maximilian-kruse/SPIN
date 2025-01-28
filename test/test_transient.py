"""Tests of the transient inference problem module"""

# ====================================== Preliminary Commands =======================================
import warnings
import pytest
import numpy as np
from typing import Callable, Tuple

import sp_inference.processes as sp
from sp_inference.inference import transient
from sp_inference.pde_problems import forms
from sp_inference.pde_problems import problems as fem

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl


# ============================================== Data ===============================================


class TestData:
    domainDim = 1
    solutionDim = 1
    lowerDomainBound = -5
    upperDomainBound = 5
    boundaryValue = 0
    numMeshPoints = 10
    femElemDegree = 1

    startTime = 0.1
    endTime = 5.1
    timeStepSize = 0.1

    testProcess = sp.OUProcess
    driftCoeff = 1
    diffusionCoeff = 1

    dataStd = 0.1
    dataSeed = 0

    simTimes = np.arange(startTime, endTime + timeStepSize, timeStepSize)
    obsTimes = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    obsPoints = np.array([-4.5, -3.0, -1.5, 0, 1.5, 3.0, 4.5])
    initSol = (
        lambda x: 1
        / (0.25 * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * np.square(x) / 0.25**2)
    )

    logSettings = {
        "params_to_infer": "all",
        "is_stationary": False,
        "verbose": False,
        "output_directory": None,
        "print_interval": 1,
    }

    feSettings = {
        "num_mesh_points": numMeshPoints,
        "boundary_locations": [lowerDomainBound, upperDomainBound],
        "boundary_values": [boundaryValue, boundaryValue],
        "element_degrees": [femElemDegree, femElemDegree],
    }

    transientSettings = {
        "start_time": startTime,
        "end_time": endTime,
        "time_step_size": timeStepSize,
        "initial_condition": initSol,
    }


# ============================================ Fixtures =============================================


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fem_space_setup() -> Tuple:
    mesh = fe.IntervalMesh(
        TestData.numMeshPoints, TestData.lowerDomainBound, TestData.upperDomainBound
    )
    funcSpaceVar = fe.FunctionSpace(mesh, "Lagrange", TestData.femElemDegree)
    funcSpaceDrift = fe.VectorFunctionSpace(
        mesh, "Lagrange", TestData.femElemDegree, dim=TestData.domainDim
    )
    funcSpaceDiffusion = fe.TensorFunctionSpace(
        mesh,
        "Lagrange",
        TestData.femElemDegree,
        shape=2 * (TestData.domainDim,),
        symmetry=True,
    )

    return mesh, funcSpaceVar, funcSpaceDrift, funcSpaceDiffusion


# ---------------------------------------------------------------------------------------------------
@pytest.fixture()
def exact_data_setup() -> fe.GenericVector:
    process = TestData.testProcess(TestData.driftCoeff, TestData.diffusionCoeff)
    _, exactValues, _ = process.compute_transient_distribution_fem(
        TestData.feSettings, TestData.transientSettings, convert=False
    )

    return exactValues


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def perturbed_data_setup() -> np.ndarray:
    process = TestData.testProcess(TestData.driftCoeff, TestData.diffusionCoeff)
    perturbedValues, _ = process.generate_data_transient_fpe(
        TestData.obsPoints,
        TestData.obsTimes,
        TestData.dataStd,
        TestData.dataSeed,
        TestData.feSettings,
        TestData.transientSettings,
    )

    return perturbedValues


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def misfit_setup(
    fem_space_setup: Tuple, perturbed_data_setup: np.ndarray
) -> transient.TransientPointwiseStateObservation:
    _, funcSpace, *_ = fem_space_setup
    data = perturbed_data_setup
    print(type(data))
    misfit = transient.TransientPointwiseStateObservation(
        funcSpace,
        TestData.obsPoints,
        TestData.obsTimes,
        TestData.simTimes,
        data,
        TestData.dataStd**2,
    )
    return misfit


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fem_model_setup():
    femModel = fem.FEMProblem(
        TestData.domainDim, TestData.solutionDim, TestData.feSettings
    )
    return femModel


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def param_var_setup(fem_model_setup: fem.FEMProblem) -> Tuple:
    femProblem = fem_model_setup
    driftFunc = fe.interpolate(
        fe.Expression(("-x[0]",), degree=TestData.femElemDegree),
        femProblem.funcSpaceDrift,
    )
    diffusionFunc = fe.interpolate(
        fe.Expression((("1.0",),), degree=TestData.femElemDegree),
        femProblem.funcSpaceDiffusion,
    )

    return driftFunc, diffusionFunc


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def form_handle_setup(param_var_setup: Tuple) -> Callable:
    _, diffusionFunc = param_var_setup
    formCallable, _ = forms.get_form("fokker_planck")

    def varform_handle(fwdVar, paramVar, adjVar):
        return formCallable(fwdVar, paramVar, diffusionFunc, adjVar)

    return varform_handle


# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pde_problem_setup(
    form_handle_setup: Callable, fem_model_setup: fem.FEMProblem
) -> transient.TransientPDEVariationalProblem:
    varform_handle = form_handle_setup
    femProblem = fem_model_setup
    funcSpaces = [
        femProblem.funcSpaceVar,
        femProblem.funcSpaceDrift,
        femProblem.funcSpaceVar,
    ]
    pdeProblem = transient.TransientPDEVariationalProblem(
        funcSpaces,
        varform_handle,
        femProblem.boundCondsForward,
        femProblem.boundCondAdjoint,
        TestData.initSol,
        TestData.simTimes,
    )

    return pdeProblem


# ---------------------------------------------------------------------------------------------------
@pytest.fixture()
def linearization_point_setup(
    param_var_setup: Tuple, pde_problem_setup: Tuple
) -> transient.TransientPDEVariationalProblem:
    paramFunc, _ = param_var_setup
    pdeProblem = pde_problem_setup
    paramVec = paramFunc.vector()
    forwardVec = pdeProblem.generate_state()
    adjointVec = pdeProblem.generate_state()
    pdeProblem.setLinearizationPoint([forwardVec, paramVec, adjointVec])

    return pdeProblem


# ========================================== Misfit Tests ===========================================


# ---------------------------------------------------------------------------------------------------
def test_misfit_generation(
    misfit_setup: transient.TransientPointwiseStateObservation,
) -> None:
    transientMisfit = misfit_setup
    assert isinstance(transientMisfit, transient.TransientPointwiseStateObservation)


# ---------------------------------------------------------------------------------------------------
def test_cost(
    misfit_setup: transient.TransientPointwiseStateObservation,
    exact_data_setup: fe.GenericVector,
) -> None:
    forwardVar = exact_data_setup
    stateList = [forwardVar, None, None]
    misfit = misfit_setup
    cost = misfit.cost(stateList)

    assert isinstance(cost, float)


# ---------------------------------------------------------------------------------------------------
def test_grad(
    misfit_setup: transient.TransientPointwiseStateObservation,
    exact_data_setup: fe.GenericVector,
) -> None:
    forwardVar = exact_data_setup
    stateList = [forwardVar, None, None]
    gradVec = forwardVar.copy()
    gradVec.zero()
    misfit = misfit_setup
    misfit.grad(hl.STATE, stateList, gradVec)

    assert True


# ---------------------------------------------------------------------------------------------------
def test_apply_ij(
    misfit_setup: transient.TransientPointwiseStateObservation,
    exact_data_setup: fe.GenericVector,
) -> None:
    direction = exact_data_setup
    hessVec = direction.copy()
    hessVec.zero()
    misfit = misfit_setup
    misfit.apply_ij(hl.STATE, hl.STATE, direction, hessVec)

    assert True


# ======================================== PDE Problem Tests ========================================


# ---------------------------------------------------------------------------------------------------
def test_pde_problem_generation(
    pde_problem_setup: transient.TransientPDEVariationalProblem,
) -> None:
    pdeProblem = pde_problem_setup
    assert isinstance(pdeProblem, transient.TransientPDEVariationalProblem)


# ---------------------------------------------------------------------------------------------------
def test_first_variation(param_var_setup: Tuple, pde_problem_setup: Tuple) -> None:
    paramFunc, _ = param_var_setup
    pdeProblem = pde_problem_setup
    paramVec = paramFunc.vector()
    forwardVec = pdeProblem.generate_state()
    adjointVec = pdeProblem.generate_state()
    rhsVec = forwardVec.copy()
    gradVec = paramVec.copy()

    pdeProblem.solveFwd(forwardVec, [None, paramVec, None])
    pdeProblem.solveAdj(adjointVec, [forwardVec, paramVec, None], rhsVec)
    pdeProblem.evalGradientParameter([forwardVec, paramVec, adjointVec], gradVec)
    assert True


# ---------------------------------------------------------------------------------------------------
def test_linearization_point(
    linearization_point_setup: transient.TransientPDEVariationalProblem,
) -> None:
    pdeProblem = linearization_point_setup
    assert True


# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("isAdj", [True, False])
def test_incremental_solve(
    linearization_point_setup: transient.TransientPDEVariationalProblem, isAdj: bool
) -> None:
    pdeProblem = linearization_point_setup
    forwardVec = pdeProblem.generate_state()
    rhsVec = forwardVec.copy()

    pdeProblem.solveIncremental(forwardVec, rhsVec, isAdj=isAdj)
    assert True


# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "i, j, direction, out",
    [
        (hl.PARAMETER, hl.STATE, "forward", "parameter"),
        (hl.PARAMETER, hl.ADJOINT, "forward", "parameter"),
        (hl.STATE, hl.PARAMETER, "parameter", "forward"),
        (hl.ADJOINT, hl.PARAMETER, "parameter", "forward"),
    ],
)
def test_apply_ij(
    linearization_point_setup: transient.TransientPDEVariationalProblem,
    i: int,
    j: int,
    direction: str,
    out: str,
) -> None:
    pdeProblem = linearization_point_setup
    forwardVec = pdeProblem.generate_state()
    paramVec = pdeProblem.generate_parameter()
    varList = {"forward": forwardVec, "parameter": paramVec}

    pdeProblem.apply_ij(i, j, varList[direction], varList[out])
    assert True
