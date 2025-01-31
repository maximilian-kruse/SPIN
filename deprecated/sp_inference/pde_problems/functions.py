"""Transient FEM Functions Module

This module provides low-level routines for the assembly and solving of transient FEM problems.
The solver is rather flexible, in that in can be used to solve problems forwards and backwards in
time. The routines are used for a high level wrapper to generate data, as well as directly in the
inference procedure.
Importantly, the current implementation assumes PDEs of the form
:math:`\frac{\partial u }{\partial t} + \mathcal{L}(u) = 0`, where the spatial operator :math:`L`
is linear in the solution variable and also includes a potential rhs contribution.
It further prescribes a fixed time step size :math:`\delta t`.

Functions:
----------
assemble_transient: Assemble lhs and rhs structures for transient solve
solve_transient
"""

# ====================================== Preliminary Commands =======================================
import warnings
import numpy as np
from typing import Any, Tuple

from ..utilities import general as utils

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ufl
    import fenics as fe
    import hippylib as hl


# ============================== Generic Transient Assembly and Solve ===============================

_checkDictProblem = {
    "init_sol": (fe.GenericVector, None, False),
    "time_step_size": ((int, float), [0, 1e10], False),
    "mass_matrix": (fe.Matrix, None, False),
    "rhs": (hl.TimeDependentVector, None, False),
    "bcs": (list, None, False),
}


# ---------------------------------------------------------------------------------------------------
def assemble_transient(
    funcSpace: fe.FunctionSpace,
    timeStepSize: float,
    stationaryForm: ufl.Form,
    boundCondList: list[fe.DirichletBC],
) -> Tuple:
    """Constructs lhs matrix and rhs vector for transient solve

    Assuming a homogeneous rhs for the PDE above, a uniform time step leads to a discretization of
    the form :math:`\overbrace{(M + \Delta t K)}^A u^{n+1} = Mu^n - \Delta t L`, where :math:`K` results from the
    discretization of the bilinear part of the weak form of the spatial operator and :math``L from
    its rhs contribution. This routine returns the matrix :math: `A` and the vector :math: `L`.

    Args:
        funcSpace (fe.FunctionSpace): Functions space of solution variable
        timeStepSize (float): Fixed time step size
        stationaryForm (ufl.Form): Stationary form operator
        boundCondList (list[fe.DirichletBC]): List of (Dirichlet) boundary conditions

    Raises:
        TypeError: Checks function space
        TypeError: Checks time step size
        TypeError: Checks stationary form
        TypeError: Checks boundary conditions

    Returns:
        Tuple: LHS matrix and rhs vector
    """

    if not isinstance(funcSpace, fe.FunctionSpace):
        raise TypeError(
            "Need valid FEniCS function space for trial and test functions."
        )
    if not (isinstance(timeStepSize, float) and timeStepSize > 0):
        raise TypeError("Time step size needs to be positive float.")
    if not isinstance(stationaryForm, ufl.form.Form):
        raise TypeError("Need valid UFL form for stationary operator.")
    if not (
        isinstance(boundCondList, list)
        and all(isinstance(boundCond, fe.DirichletBC) for boundCond in boundCondList)
    ):
        raise TypeError("Boundary conditions need to be list of FEniCS DirichletBC.")

    forwardVar = fe.TrialFunction(funcSpace)
    adjointVar = fe.TestFunction(funcSpace)
    massMatrixFunctional = forwardVar * adjointVar * fe.dx
    massMatrix = fe.assemble(massMatrixFunctional)

    spatialOpForm = fe.lhs(stationaryForm)
    rhsForm = fe.rhs(stationaryForm)
    lhsMatrix = massMatrix + timeStepSize * fe.assemble(spatialOpForm)

    if not rhsForm.empty():
        rhsVector = -fe.assemble(rhsForm)
    else:
        rhsVector = fe.Function(funcSpace).vector()

    for bc in boundCondList:
        bc.apply(lhsMatrix)

    return lhsMatrix, rhsVector


# ---------------------------------------------------------------------------------------------------
def solve_transient(
    simInds: np.ndarray,
    solver: fe.PETScLUSolver,
    problemStructs: dict[str, Any],
    resultVec: hl.TimeDependentVector,
) -> None:
    """Solves transient problem over time index array

    The ordering of the time indices determines whether problem is solved forwards or backwards
    in time.

    Args:
        simInds (np.ndarray): Simulation time indices
        solver (fe.PETScLUSolver): Matrix inversion solver, needs to be already initialized
        problemStructs (dict[str, Any]): Solver structures
            -> init_sol (fe.GenericVector): Initial condition
            -> mass matrix (fe.Matrix): Mass matrix assembled from solution variable function space
            -> rhs: Right hand side vector for solve
            -> bcs: Boundary conditions in space
        resultVec (hl.TimeDependentVector): Result

    Raises:
        TypeError: Checks simulation time indices
        TypeError: Checks solver object
    """

    if not isinstance(simInds, np.ndarray):
        raise TypeError(
            "Simulation time indices need to be provided as numpy array of ints."
        )
    if not isinstance(solver, fe.cpp.la.GenericLinearSolver):
        raise TypeError("Routine requires initialized Linear solver.")
    utils.check_settings_dict(problemStructs, _checkDictProblem)

    initSol = problemStructs["init_sol"]
    dt = problemStructs["time_step_size"]
    massMatrix = problemStructs["mass_matrix"]
    rhsVec = problemStructs["rhs"]
    boundConds = problemStructs["bcs"]
    resultVec.zero()

    rhs = fe.Vector(rhsVec.data[0])
    resultVec.data[simInds[0]].axpy(1, initSol)

    for i, iLast in zip(simInds[1:], simInds[:-1]):
        rhs.zero()
        rhs.axpy(1, massMatrix * resultVec.data[iLast] + dt * rhsVec.data[i])
        for bc in boundConds:
            bc.apply(rhs)

        solver.solve(resultVec.data[i], rhs)
