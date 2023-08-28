"""Tests of the FEM sub-package"""

#====================================== Preliminary Commands =======================================
import warnings
import pytest
import numpy as np
from typing import Any, Tuple

import sp_inference.utilities.general as utils
import sp_inference.pde_problems.forms as femForms
import sp_inference.pde_problems.functions as femFunctions
import sp_inference.pde_problems.problems as femProblems

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl

#============================================== Data ===============================================

class TestData:
    domainDim = 1
    solutionDim = 1
    lowerDomainBound = -5
    upperDomainBound = 5
    boundaryValue = 0
    numMeshPoints = 10
    femElemDegree = 1

    startTime = 0.1
    endTime = 1.1
    timeStepSize = 0.1
    def initSol(x):
        return np.exp(-np.square(x))
    simTimes = np.arange(startTime, endTime+timeStepSize, timeStepSize)

    feSettings = {"num_mesh_points": numMeshPoints,
                  "boundary_locations": [lowerDomainBound, upperDomainBound],
                  "boundary_values": [boundaryValue, boundaryValue],
                  "element_degrees": [femElemDegree, femElemDegree]}


#============================================ Fixtures =============================================

#---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fem_space_setup() -> Tuple:
    mesh = fe.IntervalMesh(TestData.numMeshPoints, 
                           TestData.lowerDomainBound, 
                           TestData.upperDomainBound)
    funcSpace = fe.FunctionSpace(mesh, 'Lagrange', TestData.femElemDegree)
    funcSpaceVec = fe.VectorFunctionSpace(mesh, 
                                          'Lagrange',
                                          TestData.femElemDegree,
                                          dim=TestData.domainDim)
    funcSpaceTensor = fe.TensorFunctionSpace(mesh,
                                             'Lagrange',
                                             TestData.femElemDegree,
                                             shape=2*(TestData.domainDim,))

    return mesh, funcSpace, funcSpaceVec, funcSpaceTensor

#---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fem_var_setup(fem_space_setup: Tuple) -> Tuple:
    _, funcSpace, funcSpaceVec, funcSpaceTensor = fem_space_setup
    forwardFunc = fe.TrialFunction(funcSpace)
    adjointFunc = fe.TestFunction(funcSpace)
    driftFunc = fe.interpolate(fe.Expression(('-x[0]',), degree=TestData.femElemDegree),
                               funcSpaceVec)
    diffusionFunc = fe.interpolate(fe.Expression((('1.0',),), degree=TestData.femElemDegree),
                                   funcSpaceTensor)

    return forwardFunc, adjointFunc, driftFunc, diffusionFunc

#---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def transient_problem_setup(fem_space_setup: Tuple, fem_var_setup: Tuple) -> Tuple:

    def on_boundary(x: Any, on_boundary: bool) -> bool:
        return on_boundary

    _, funcSpace, *_ = fem_space_setup
    forwardFunc, adjointFunc, driftFunc, diffusionFunc = fem_var_setup
    varFormFunc, _ = femForms.get_form('fokker_planck')
    varForm = varFormFunc(forwardFunc, driftFunc, diffusionFunc, adjointFunc)
    bcList = [fe.DirichletBC(funcSpace, fe.Constant(TestData.boundaryValue), on_boundary)]

    return varForm, bcList

#---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def transient_matrix_setup(fem_var_setup: Tuple, transient_problem_setup: Tuple) -> Tuple:
    forwardFunc, adjointFunc, *_ = fem_var_setup
    varForm, bcList = transient_problem_setup

    massMatrix = fe.assemble(forwardFunc * adjointFunc * fe.dx)
    lhsForm = fe.lhs(varForm)
    rhsForm = fe.rhs(varForm)
    lhsMatrix = massMatrix + TestData.timeStepSize * fe.assemble(lhsForm)
    rhsVector = -fe.assemble(rhsForm)
    for bc in bcList:
        bc.apply(lhsMatrix)

    return massMatrix, lhsMatrix, rhsVector

#======================================= FEM Function Tests ========================================

#---------------------------------------------------------------------------------------------------
def test_transient_assembly(fem_space_setup: Tuple, 
                            transient_problem_setup: Tuple, 
                            transient_matrix_setup: Tuple) -> None:
    _, funcSpace, *_ = fem_space_setup
    varForm, bcList = transient_problem_setup
    _, lhsMatrix, rhsVector = transient_matrix_setup

    testLhsMatrix, testRhsVector = femFunctions.assemble_transient(funcSpace, 
                                                                   TestData.timeStepSize, 
                                                                   varForm, 
                                                                   bcList)

    assert isinstance(testLhsMatrix, fe.Matrix)
    assert isinstance(testRhsVector, fe.GenericVector)
    assert np.array_equal(testLhsMatrix.array(), lhsMatrix.array())
    assert np.all(testRhsVector == rhsVector)

#---------------------------------------------------------------------------------------------------
def test_transient_solve(fem_space_setup: Tuple,
                         transient_problem_setup: Tuple,
                         transient_matrix_setup: Tuple) -> None:
    _, funcSpace, *_ = fem_space_setup
    _, bcList = transient_problem_setup
    massMatrix, lhsMatrix, rhsConst = transient_matrix_setup

    initSol = utils.pyfunc_to_fefunc(TestData.initSol, funcSpace).vector()
    simTimeInds = np.indices((TestData.simTimes.size,)).flatten()
    rhsVector = hl.TimeDependentVector(TestData.simTimes)
    rhsVector.initialize(massMatrix, 0)
    for i in simTimeInds:
        rhsVector.data[i].axpy(1, rhsConst)

    solveSettings = {"time_step_size": TestData.timeStepSize,
                     "init_sol": initSol,
                     "mass_matrix": massMatrix,
                     "rhs": rhsVector,
                     "bcs": bcList}
    solver = fe.LUSolver(lhsMatrix)
    resultVec = hl.TimeDependentVector(simTimeInds)
    resultVec.initialize(massMatrix, 1)
    femFunctions.solve_transient(simTimeInds, solver, solveSettings, resultVec)
    assert True


#======================================= FEM Problem Tests =========================================

#---------------------------------------------------------------------------------------------------
def test_stationary_fem_problem() -> None:
    domainDim = TestData.domainDim
    solutionDim = TestData.solutionDim
    problem = femProblems.FEMProblem(domainDim, solutionDim, TestData.feSettings)

    assert isinstance(problem.mesh, fe.Mesh)
    assert (isinstance(problem.funcSpaceVar, fe.FunctionSpace)
            and problem.funcSpaceVar.num_sub_spaces() == 0)
    assert (isinstance(problem.funcSpaceDrift, fe.FunctionSpace)
            and problem.funcSpaceDrift.num_sub_spaces() == domainDim)
    assert (isinstance(problem.funcSpaceDiffusion, fe.FunctionSpace)
            and problem.funcSpaceDiffusion.num_sub_spaces() == domainDim)
    assert (isinstance(problem.funcSpaceAll, fe.FunctionSpace)
            and problem.funcSpaceAll.num_sub_spaces() \
                == int(0.5 * domainDim * (domainDim + 1)) + domainDim)

    assert (isinstance(problem.boundCondsForward, list)
            and all(isinstance(bc, fe.DirichletBC) for bc in problem.boundCondsForward))
    assert (isinstance(problem.boundCondAdjoint, list)
            and all(isinstance(bc, fe.DirichletBC) for bc in problem.boundCondAdjoint))

#---------------------------------------------------------------------------------------------------
def test_transient_fem_problem() -> None:
    problem = femProblems.TransientFEMProblem(TestData.domainDim,
                                              TestData.solutionDim,
                                              TestData.feSettings, 
                                              TestData.simTimes)
    def driftFunc(x):
        return -x
    def diffusionFunc(x):
        return np.ones(x.size)
    varFormFunc, _ = femForms.get_form('fokker_planck')

    problem.assemble(varFormFunc, driftFunc, diffusionFunc)
    transientSol = problem.solve(TestData.initSol)

    assert isinstance(transientSol, np.ndarray)
    assert transientSol.shape == (TestData.simTimes.size, problem.funcSpaceVar.dim())
