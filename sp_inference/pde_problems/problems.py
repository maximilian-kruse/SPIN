"""  FEM problem module

This module provides a high-level interface to the FEM problem setup. For transient problems, it
additionally contains assembly and solve wrappers, which can be used for artificial data generation.
The FEM problems are specifically designed for the generating equations of stochastic processes in
one and two dimensions. They further assume Dirichlet boundary conditions.

Classes:
--------
FEMProblem: General FEM problem class
TransientFEMProblem: Specialization for transient problems, derived from FEMProblem
"""

#====================================== Preliminary Commands =======================================
import warnings
import numpy as np
from typing import Any, Callable, Optional, Tuple, Union
from . import functions
from ..utilities import general as utils

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl

#======================================== FEM Problem Class ========================================
class FEMProblem:
    """General FEM problem class

    Sets up mesh, function spaces and boundary conditions. The mesh has uniform spacing per
    dimension. For 2D problem, it constitutes a rectangle.

    Attributes:
        mesh: FEM mesh
        funcSpaceVar: Function space of the FEM solution variable
        funcSpaceDrift: Space of the drift function
        funcSpaceDiffusion: Space of the diffusion function
        funcSpaceAll: Combined space of drift and diffusion function
        boundCondsForward: (Dirichlet) boundary conditions of the solution variable
        boundCondsAdjoint: (Homogeneous) boundary conditions of the adjoint/test function

    Methods:
        set_up_mesh: Constructs mesh object
        set_up_funcspaces: Constructs function space objects
        construct_boundary_functions: Sets up forward and adjoint boundary conditions
    """

    _checkDictFE = {
        "num_mesh_points": ((int, list), None, False),
        "boundary_locations": (list, None, False),
        "boundary_values": (list, None, False),
        "element_degrees": (list, None, False)
    }

    #-----------------------------------------------------------------------------------------------
    def __init__(self, domainDim: int, solutionDim: int, feSettings: dict[str, Any]) -> None:
        """Constructor

        Calls setup routines.

        Args:
            dimension (int): Spatial problem dimension (1 or 2)
            feSettings (dict[str, Any]): FEM problem configuration
                -> num_mesh_points (int, list): Number of mesh points, single value for 1D, list
                                                of two values for 2D
                -> boundary_locations (list): Locations of the boundary
                                              1D: [upper, lower]
                                              2D: [left, right, bottom, top]
                -> boundary_values (list): Function values at boundary locations (see above)
                -> element_degrees (list): FEM element degrees for forward/adjoint and parameter
                                           functions

        Raises:
            ValueError: Checks problem dimension
            MiscErrors: Checks FEM settings
        """

        if not domainDim in [1, 2]:
            raise ValueError("Problem dimension needs to be one or two.")
        if not solutionDim > 0:
            raise ValueError("Solution dimension needs to be positive integer.")
        utils.check_settings_dict(feSettings, self._checkDictFE)

        self._domainDim = domainDim
        self._solutionDim = solutionDim
        self._mesh = None
        self._funcSpaceVar = None
        self._funcSpaceDrift = None
        self._funcSpaceDiffusion = None
        self._funcSpaceAll = None
        self._boundCondsForward = None
        self._boundCondAdjoint = None

        self.mesh = self.set_up_mesh(feSettings["num_mesh_points"],
                                     feSettings["boundary_locations"])
        self.funcSpaceVar, self.funcSpaceDrift, self.funcSpaceDiffusion, self.funcSpaceAll \
             = self.set_up_funcspaces(self.mesh, feSettings["element_degrees"])
        self.boundCondsForward, self.boundCondAdjoint \
             = self.construct_boundary_functions(self.funcSpaceVar, 
                                                 feSettings["boundary_locations"],
                                                 feSettings["boundary_values"])

    #-----------------------------------------------------------------------------------------------
    def set_up_mesh(self, numMeshPoints: Union[int, list[int]], boundaryLocs: list) -> fe.Mesh:
        """Constructs mesh object

        For 1D problems, the mesh is a uniformly partitioned line segment, for 2D problems a 
        rectangle.

        Args:
            numMeshPoints (Union[int, list[int]]): Number of mesh points (per dimension)
            boundaryLocs (list): Boundary locations

        Raises:
            TypeError: Checks that for a 1D problem, a single value for the number of mesh points
                       and a list of two values for the boundary locations are provided.
            TypeError: Checks that for a 2D problem, a list of two values for the number of mesh
                       points and a list of four values for the boundary locations are provided.

        Returns:
            fe.Mesh: Mesh object
        """

        if self._domainDim == 1:
            if not isinstance(numMeshPoints, int):
                raise TypeError("1D problem requires single integer for number of mesh points.")
            if not len(boundaryLocs) == 2:
                raise TypeError("1D problem requires two boundary locations.")
            mesh = fe.IntervalMesh(numMeshPoints, *boundaryLocs)
            
        elif self._domainDim == 2:
            if not isinstance(numMeshPoints, list):
                raise TypeError("1D problem requires list of two integers for number of mesh points.")
            if not len(boundaryLocs) == 4:
                raise TypeError("2D problem requires four boundary locations.")
            boundOne = fe.Point(boundaryLocs[0], boundaryLocs[2])
            boundTwo = fe.Point(boundaryLocs[1], boundaryLocs[3])
            mesh = fe.RectangleMesh(boundOne, boundTwo, *numMeshPoints)

        return mesh

    #-----------------------------------------------------------------------------------------------
    def set_up_funcspaces(self, mesh: fe.Mesh, elemDegrees: list[int]) -> Tuple:
        """Sets up function spaces

        Args:
            mesh (fe.Mesh): Mesh
            elemDegrees (list[int]): FEM element degrees
                                     (two values, for forward/adjoint and parameter functions)

        Raises:
            TypeError: Checks mesh validity
            TypeError: Checks element degrees

        Returns:
            Tuple: Function spaces for forward/adjoint, drift and diffusion functions, as well as
                   composite space for drift and diffusion
        """
        if not isinstance(mesh, fe.Mesh):
            raise TypeError("Mesh needs to be proper FEniCS mesh object.")
        if not ( isinstance(elemDegrees, list) 
        and all(isinstance(degree, int) for degree in elemDegrees) ):
            raise TypeError("Need to provide element degrees as list of ints.")

        if self._solutionDim == 1:
            funcSpaceVar = fe.FunctionSpace(mesh, 'Lagrange', elemDegrees[0])
        else:
            funcSpaceVar = fe.VectorFunctionSpace(mesh,
                                                  'Lagrange',
                                                  elemDegrees[0],
                                                  dim=self._solutionDim)
        funcSpaceDrift = fe.VectorFunctionSpace(mesh,
                                                'Lagrange',
                                                elemDegrees[1],
                                                dim=self._domainDim)
        funcSpaceDiffusion = fe.TensorFunctionSpace(mesh, 
                                                    'Lagrange', 
                                                    elemDegrees[1],
                                                    shape=(self._domainDim, self._domainDim), 
                                                    symmetry=True)
                                  
        numElems = int(0.5 * self._domainDim * (self._domainDim + 1)) + self._domainDim
        funcSpaceAll = fe.VectorFunctionSpace(mesh, 'Lagrange', elemDegrees[1], dim=numElems)

        return funcSpaceVar, funcSpaceDrift, funcSpaceDiffusion, funcSpaceAll

    #-----------------------------------------------------------------------------------------------
    def construct_boundary_functions(self, 
                                     funcSpace: fe.FunctionSpace, 
                                     boundaryLocs: list[float],
                                     boundaryVals: list[float])\
                                     -> Tuple[list[fe.DirichletBC], fe.DirichletBC]:
        """Sets up Dirichlet boundary conditions

        Returns:
            Tuple: Boundary conditions for forward variable, homogeneous boundary condition for
                   adjoint/test function.
        """

        assert isinstance(boundaryLocs, list) and len(boundaryLocs) == 2*self._domainDim, \
            "Boundary locations need to be provided as list with two (1D) or 4 (2D) entries."
        assert isinstance(boundaryVals, list) and len(boundaryVals) == 2*self._domainDim, \
            "Boundary values need to be provided as list with two (1D) or 4 (2D) entries."
        assert all(isinstance(loc, (int, float)) for loc in boundaryLocs), \
            "Boundary locations need to be provided as numbers."
        
        boundCondsForward = []     
        for i in range(2*self._domainDim):
            _boundaryLocs[_boundaryNames[i]] = boundaryLocs[i]
            if self._solutionDim == 1:
                boundCondsForward.append(fe.DirichletBC(funcSpace,
                                         fe.Constant(boundaryVals[i]),
                                         _boundaryFuncs[i]))
            else:
                for j in range(self._solutionDim):
                    boundCondsForward.append(fe.DirichletBC(funcSpace.sub(j),
                                             fe.Constant(boundaryVals[i][j]),
                                             _boundaryFuncs[i]))

        if self._solutionDim == 1:
            boundCondAdjoint = fe.DirichletBC(funcSpace, fe.Constant(0.0), _on_boundary_dummy)
        else:
            boundCondAdjoint = []
            for j in range(self._solutionDim):
                boundCondAdjoint.append(fe.DirichletBC(funcSpace.sub(j),
                                                       fe.Constant(0.0),
                                                       _on_boundary_dummy))

        return boundCondsForward, boundCondAdjoint
    
    #-----------------------------------------------------------------------------------------------
    def solve(self, 
              formHandle: Callable, 
              driftFunction: Callable, 
              diffusionFunction: Callable,
              convert=True) -> None:
        
        if not callable(formHandle):
            raise TypeError("Form handle needs to be callable object (with 4 arguments).")
        if not all(callable(func) for func in [driftFunction, diffusionFunction]):
            raise TypeError("Drift and diffusion function must be callable objects.")

        forwardVar = fe.TrialFunction(self.funcSpaceVar)
        adjointVar = fe.TestFunction(self.funcSpaceVar)
        driftVar = utils.pyfunc_to_fefunc(driftFunction, self.funcSpaceDrift)
        diffusionVar = utils.pyfunc_to_fefunc(diffusionFunction, self.funcSpaceDiffusion)
        solutionVar = fe.Function(self.funcSpaceVar)

        weakForm = formHandle(forwardVar, driftVar, diffusionVar, adjointVar)
        lhs = fe.lhs(weakForm)
        rhs = fe.rhs(weakForm)
        fe.solve(lhs == rhs, solutionVar, self.boundCondsForward)

        solutionVec = solutionVar.vector()
        if convert:
            solutionVec = utils.reshape_to_np_format(solutionVec, self._solutionDim)
            solutionVec = utils.process_output_data(solutionVec)

        return solutionVec

    #-----------------------------------------------------------------------------------------------   
    @property
    def mesh(self) -> fe.Mesh:
        if self._mesh is None:
            raise ValueError("Property has not been initialized.")
        return self._mesh
    
    @property
    def funcSpaceVar(self) -> fe.FunctionSpace:
        if self._funcSpaceVar is None:
            raise ValueError("Property has not been initialized.")
        return self._funcSpaceVar

    @property
    def funcSpaceDrift(self) -> fe.VectorFunctionSpace:
        if self._funcSpaceDrift is None:
            raise ValueError("Property has not been initialized.")
        return self._funcSpaceDrift

    @property
    def funcSpaceDiffusion(self) -> fe.TensorFunctionSpace:
        if self._funcSpaceDiffusion is None:
            raise ValueError("Property has not been initialized.")
        return self._funcSpaceDiffusion

    @property
    def funcSpaceAll(self) -> fe.VectorFunctionSpace:
        if self._funcSpaceAll is None:
            raise ValueError("Property has not been initialized.")
        return self._funcSpaceAll

    @property
    def boundCondsForward(self) -> list[fe.DirichletBC]:
        if self._boundCondsForward is None:
            raise ValueError("Property has not been initialized.")
        return self._boundCondsForward

    @property
    def boundCondAdjoint(self) -> list[fe.DirichletBC]:
        if self._boundCondAdjoint is None:
            raise ValueError("Property has not been initialized.")
        return self._boundCondAdjoint

    #-----------------------------------------------------------------------------------------------
    @mesh.setter
    def mesh(self, mesh: fe.Mesh) -> None:
        if not isinstance(mesh, fe.Mesh):
            raise TypeError("Variable function space needs to be scalar function space.")
        self._mesh = mesh
    
    @funcSpaceVar.setter
    def funcSpaceVar(self, funcSpaceVar: fe.FunctionSpace) -> None:
        if not isinstance(funcSpaceVar, fe.FunctionSpace):
            raise TypeError("Variable function space needs to be scalar function space.")
        self._funcSpaceVar = funcSpaceVar

    @funcSpaceDrift.setter
    def funcSpaceDrift(self, funcSpaceDrift: fe.VectorFunctionSpace) -> None:
        if not isinstance(funcSpaceDrift, fe.FunctionSpace):
            raise TypeError("Drift function space needs to be vector function space.")
        self._funcSpaceDrift = funcSpaceDrift
    
    @funcSpaceDiffusion.setter
    def funcSpaceDiffusion(self, funcSpaceDiffusion: fe.VectorFunctionSpace) -> None:
        if not isinstance(funcSpaceDiffusion, fe.FunctionSpace):
            raise TypeError("Diffusion function space needs to be tensor function space.")
        self._funcSpaceDiffusion = funcSpaceDiffusion

    @funcSpaceAll.setter
    def funcSpaceAll(self, funcSpaceAll: fe.VectorFunctionSpace) -> None:
        if not isinstance(funcSpaceAll, fe.FunctionSpace):
            raise TypeError("Mixed function space needs to be vector function space.")
        self._funcSpaceAll = funcSpaceAll

    @boundCondsForward.setter
    def boundCondsForward(self, boundCondsForward: list[fe.DirichletBC]) -> None:
        if isinstance(boundCondsForward, fe.DirichletBC):
            boundCondsForward = [boundCondsForward]
        elif not (isinstance(boundCondsForward, list)
        and all(isinstance(bc, fe.DirichletBC) for bc in boundCondsForward)):
            raise TypeError("Forward boundary conditions need to be given as list of DirichletBCs.")
        self._boundCondsForward = boundCondsForward

    @boundCondAdjoint.setter
    def boundCondAdjoint(self, boundCondAdjoint: fe.DirichletBC) -> None:
        if isinstance(boundCondAdjoint, fe.DirichletBC):
            boundCondAdjoint = [boundCondAdjoint]
        elif not (isinstance(boundCondAdjoint, list)
        and all(isinstance(bc, fe.DirichletBC) for bc in boundCondAdjoint)):
            raise TypeError("Adjoint boundary condition needs to be given as list of DirichletBCs.")
        self._boundCondAdjoint = boundCondAdjoint


#=================================== Transient FEM Problem Class ===================================
class TransientFEMProblem(FEMProblem):
    """Transient FEM problem

    This class extends the functionality of the general FEM setup for transient problems. It provides
    convenient assembly and solver capabilities and offers a simple but complete FEM solver for
    artificial data generation for stochastic processes.
    Importantly, the current implementation  also PDEs of the form
    :math:`\frac{\partial u }{\partial t} + \mathcal{L}(u) = 0`, where the spatial operator :math:`L`
    is linear in the solution variable and also includes a potential rhs contribution.
    It further prescribes a fixed time step size :math:`\delta t`.

    Methods:
    --------
    assemble: Set up solver structures
    solve: Compute transient PDE problem

    """

    #-----------------------------------------------------------------------------------------------
    def __init__(self, 
                 domainDim: int,
                 solutionDim: int,
                 feSettings: dict[str, Any], 
                 simTimes: np.ndarray) -> None:
        """Constructor

        The constructor calls the base class setup and provides some additional data structures.

        Args:
            dimension (int): Spatial problem dimension
            feSettings (dict[str, Any]): FEM settings (see FEm model)
            simTimes (np.ndarray): Simulation time array

        Raises:
            TypeError: Checks type of simulation time array
            ValueError: Checks that simulation time points are evenly spaced
        """
        
        super().__init__(domainDim, solutionDim, feSettings)
        if not isinstance(simTimes, np.ndarray):
            raise TypeError("Need to provide simulation times as numpy array.")
        if not np.allclose(np.diff(simTimes), np.diff(simTimes)[0]):
            raise ValueError("Simulation time array needs to be evenly spaced.")
        
        self._massMatrix = None
        self._rhsVector = None
        self._solver = None
        
        self._simTimes = simTimes
        self._simTimeInds = np.indices(simTimes.shape).flatten()
        self._dt = simTimes[1] - simTimes[0]

    #-----------------------------------------------------------------------------------------------
    def assemble(self, 
                 stationaryFormHandle: Callable, 
                 driftFunction: Callable, 
                 diffusionFunction: Callable) -> None:
        """Assembles solver structures

        The assembly routine provides an initialized direct solver from the discretized system
        matrix. In addition, it assembles the right hand side vector.

        Args:
            stationaryFormHandle (Callable): Weak form handle of stationary PDE operator
            driftFunction (Callable): Drift function handle
            diffusionFunction (Callable): Diffusion function handle

        Raises:
            TypeError: Checks that form handle is callable
            TypeError: Checks that drift and diffusion functions are callable
        """

        if not callable(stationaryFormHandle):
            raise TypeError("Form handle needs to be callable object (with 4 arguments).")
        if not all(callable(func) for func in [driftFunction, diffusionFunction]):
            raise TypeError("Drift and diffusion function must be callable objects.")

        forwardVar = fe.TrialFunction(self.funcSpaceVar)
        adjointVar = fe.TestFunction(self.funcSpaceVar)
        driftVar = utils.pyfunc_to_fefunc(driftFunction, self.funcSpaceDrift)
        diffusionVar = utils.pyfunc_to_fefunc(diffusionFunction, self.funcSpaceDiffusion)

        massMatrixFunctional = forwardVar * adjointVar * fe.dx
        self._massMatrix = fe.assemble(massMatrixFunctional)
        weakForm = stationaryFormHandle(forwardVar, driftVar, diffusionVar, adjointVar)
        lhsMatrix, rhsConst = functions.assemble_transient(self.funcSpaceVar,
                                                           self._dt,
                                                           weakForm,
                                                           self.boundCondsForward)

        self._rhsVector = hl.TimeDependentVector(self._simTimes)
        self._rhsVector.initialize(self._massMatrix, 0)
        for i in self._simTimeInds:
            self._rhsVector.data[i].axpy(1, rhsConst)
                                                     
        self._solver = fe.LUSolver(lhsMatrix)

    #-----------------------------------------------------------------------------------------------
    def solve(self, 
              initFunc: Union[Callable, fe.GenericVector],
              convert: Optional[bool]=True) -> Union[np.ndarray, hl.TimeDependentVector]:
        """Solves the initial value problem

        Args:
            initFunc (Union[Callable, fe.GenericVector]): Initial condition
            convert (Optional[bool], optional): Determines if the result is converted into a
                                                numpy array. Defaults to True

        Raises:
            AttributeError: Checks if solver structures have been initialized through call to
                            assemble method
            TypeError: Checks initial condition

        Returns:
            Union[np.ndarray, hl.TimeDependentVector]: Solution vector in space and time
        """

        if any(struct is None for struct in [self._massMatrix, self._rhsVector, self._solver]):
            raise AttributeError("Solver structures are not initialized.")
        if callable(initFunc):
            initFunc = utils.pyfunc_to_fevec(initFunc, self.funcSpaceVar)
        elif not isinstance(initFunc, fe.GenericVector):
            raise TypeError("Initial condition needs to be a callable object of FEniCS vector.") 
        
        resultVec = hl.TimeDependentVector(self._simTimes)
        resultVec.initialize(self._massMatrix, 1)

        solveSettings = {"init_sol": initFunc,
                         "time_step_size": self._dt,
                         "mass_matrix": self._massMatrix,
                         "rhs": self._rhsVector,
                         "bcs": self.boundCondsForward}
        
        functions.solve_transient(self._simTimeInds, self._solver, solveSettings, resultVec)
        if convert:
            resultVec = utils.tdv_to_nparray(resultVec)

            if self._solutionDim > 1:
                numXValues = int(self._funcSpaceVar.dim() / self._solutionDim)
                resultVecStructured = np.zeros(numXValues, self._solutionDim, self._simTimes.size)

                for i in range(self._simTimeInds):
                    resultVecStructured[:, :, i] = utils.reshape_to_np_format(resultVec[:, i],
                                                                              self._solutionDim)
            else:
                resultVecStructured = resultVec

            resultVec = utils.process_output_data(resultVecStructured)

        return resultVec


#================================ Supplementary Boundary Functions =================================
"""
These boundary functions need to be placed out of class scope, since they cannot contain additional
arguments such as 'self' and 'cls'. Similar issues speak against a definition as static methods.
"""

#---------------------------------------------------------------------------------------------------
def _on_left_boundary(x: Any, on_boundary: bool) -> bool:
    """Checks if point lies on left bound of the domain"""

    assert _boundaryLocs["left"] is not None, \
        "Boundary value not given, initialize via construct function."
    return on_boundary and fe.near(x[0], _boundaryLocs["left"], _numericTol)

#---------------------------------------------------------------------------------------------------
def _on_right_boundary(x: Any, on_boundary: bool) -> bool:
    """Checks if point lies on right bound of the domain"""

    assert _boundaryLocs["right"] is not None, \
        "Boundary value not given, initialize via construct function."
    return on_boundary and fe.near(x[0], _boundaryLocs["right"], _numericTol)

#---------------------------------------------------------------------------------------------------
def _on_lower_boundary(x: Any, on_boundary: bool) -> bool:
    """Checks if point lies on lower bound of the domain"""

    assert _boundaryLocs["lower"] is not None, \
        "Boundary value not given, initialize via construct function."
    return on_boundary and fe.near(x[1], _boundaryLocs["lower"], _numericTol)

#---------------------------------------------------------------------------------------------------
def _on_upper_boundary(x: Any, on_boundary: bool) -> bool:
    """Checks if point lies on upper bound of the domain"""

    assert _boundaryLocs["upper"] is not None, \
        "Boundary value not given, initialize via construct function."
    return on_boundary and fe.near(x[1], _boundaryLocs["upper"], _numericTol)

#---------------------------------------------------------------------------------------------------
def _on_boundary_dummy(x: Any, on_boundary: bool) -> bool:
    """Dummy check for adjoint problem"""

    return on_boundary

#---------------------------------------------------------------------------------------------------   
_boundaryNames = ["left", "right", "lower", "upper"]
_boundaryFuncs = [_on_left_boundary, _on_right_boundary, _on_lower_boundary, _on_upper_boundary]
_boundaryLocs = dict.fromkeys(_boundaryNames, None)
_numericTol = fe.DOLFIN_EPS