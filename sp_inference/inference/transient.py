""" Core Module for Transient Inference Problems

This module provides the functionalities for linearized Bayesian inference with transient PDE model
constraints. It comprises a misfit and a pde problem class. These classes extend the hIPPYlib
functionality significantly. Their interface integrates seamlessly into the overall hIPPYlib
framework. As a consequence, function and property naming deviates from that of the remaining code.
The transient problem classes underlie certain assumptions. Most importantly, they only work
fro pde problems linear in the forward, parameter and adjoint variables. However, these restrictions
can be alleviated relatively easy, at the cost of making the procedure mode computationally expensive.
The precise code locations for extensions are highlighted accordingly. Moreover, all transient
solves work on uniform grids with fixed time step size :math: `dt`. Again, the solving routines
might be easily extended towards more sophisticated algorithms. Lastly, while the forward and 
adjoint variables are functions of time in the transient context, we assume that the parameter
function remains time-independent.
The theoretical background for the implementation mainly relies on 
`this article <https://apps.dtic.mil/sti/citations/ADA555315>`_. Prior knowledge of the methods
presented therein is necessary to understand the code.

NOTE: The code explanations often use the terms 'variable', 'function' and 'vector' interchangeably.
      This is the case because the variables of the inference problem are indeed functions. In the
      context of FEM discretization, in turn, these functions are expressed as coefficient vectors
      associated with the chosen basis functions.

Classes:
--------
TransientPointwiseStateObservation: Transient misfit functional
TransientPDEVariationalProblem: Transient pde problem
"""

#====================================== Preliminary Commands =======================================
import warnings
import numpy as np
from typing import Callable, Optional, Tuple, Union

from ..pde_problems import functions as femFunctions
from ..utilities import general as utils

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl


#===================================== Transient Misfit Class ======================================
class TransientPointwiseStateObservation(hl.Misfit):
    """ Transient Misfit Functional

    This class implements a fully differentiable cost functional for data points in space and time.
    It assumes these points to be discrete, rather then (continuous) functions. Space and time
    locations can be specified arbitrarily and independently. However, the class assumes the same
    spacial locations at every time step, hence a single projection matrix. At the cost of a higher
    memory footprint, this can be easily extended to more versatile cases.
    The cost function assumed for this class is of the form
    :math:`\frac{1}{2\sigma^2}\sum_{i=1}^{n_s}\int_0^T (Bu-d_i,Bu-d_i)\delta_{t_i}dt`. Apparently,
    the formulation contains Dirac delta contributions, which are discretized as rectangle pulses
    on the solver time grid. Further the functional assumes a scalar variance to characterize the
    noise in the data.

    Attributes:
        d (list[list[fe.GenericVector]]): Data, assigned to points of simulation grid
        noise_variance (float): Data noise variance

    Methods:
        cost: Evaluates cost functional for given state
        grad: Evaluates gradient with respect to given forward or parameter function
        setLinearizationPoint: Sets linearization point for quadratic approximation
        apply_ij: Applies second variation w.r.t. to forward and/or parameter functions to given
                  direction
    """

    #-----------------------------------------------------------------------------------------------
    def __init__(self,
                 funcSpace: fe.FunctionSpace,
                 obsPoints: Union[int, float, np.ndarray],
                 obsTimes: Union[int, float, np.ndarray],
                 simTimes: np.ndarray,
                 data: Optional[Union[hl.TimeDependentVector, np.ndarray]]=None,
                 noiseVar: Optional[float]=None) -> None:
        """Constructor

        Prepares computations, constructs mapping from data time points to simulation time grid.

        Args:
            funcSpace (fe.FunctionSpace): Forward variable function space
            obsPoints (Union[int, float, np.ndarray]): Spacial locations of given data points
            obsTimes (Union[int, float, np.ndarray]): Recording times of given data points
            simTimes (np.ndarray): Time grid used for transient solves
            data (Optional[Union[hl.TimeDependentVector, np.ndarray]], optional):
                Data point values, can be assigned later. Defaults to None
            noiseVar (Optional[float], optional):
                Data noise variance, can be assigned later. Defaults to None

        Raises:
            TypeError: Checks type of simulation time grid
            ValueError: Checks that simulation times are ordered
            ValueError: Checks that simulation time grid is evenly spaced
            ValueError: Checks that observation times lie within simulation times
            TypeError: Checks type of provided function space
        """

        obsPoints = utils.process_input_data(obsPoints)
        obsTimes = utils.process_input_data(obsTimes, enforce1D=True)
        if not isinstance(simTimes, np.ndarray):
            raise TypeError("Simulation times need to be given as numpy arrays.")
        if not np.all(np.diff(simTimes) > 0):
            raise ValueError("Simulation time array needs to be given in ascending order.")
        if not np.allclose(np.diff(simTimes), np.diff(simTimes)[0]):
            raise ValueError("Simulation time array needs to be evenly spaced.")
        if not (np.amin(obsTimes) >= np.amin(simTimes)) and (np.amax(obsTimes) <= np.amax(simTimes)):
            raise ValueError("Bounds of observation times need to lie within simulation times.")
        if not isinstance(funcSpace, fe.FunctionSpace):
            raise TypeError("Function space needs to be proper FEniCS object.")

        self._data = None
        self._noiseVar = None
        self._obsTimes = obsTimes
        self._simTimes = simTimes
        self._dt = simTimes[1] - simTimes[0]
        self._simTimeInds = np.indices(simTimes.shape).flatten()
        self._obsTimeInds = np.indices(obsTimes.shape).flatten()

        self._obsMap = self._project_obs_to_sim_grid()
        self._projMat = hl.pointwiseObservation.assemblePointwiseObservation(funcSpace, obsPoints)
        self._currentFullState = fe.Vector()
        self._currentProjState = fe.Vector()
        self._projMat.init_vector(self._currentFullState, 1)
        self._projMat.init_vector(self._currentProjState, 0)

        if data is not None:
            self.d = data
        if noiseVar is not None:
            self.noise_variance = noiseVar

    #-----------------------------------------------------------------------------------------------
    def cost(self, stateList: list) -> float:
        """Evaluates cost functional for given state

        Args:
            stateList (list): Forward, parameter and adjoint functions to evaluate cost for
                              (only forward function is necessary)

        Raises:
            TypeError: Checks type of provided state list
            TypeError: Checks type of forward function

        Returns:
            float: Cost or misfit value
        """

        if not (isinstance(stateList, list) and len(stateList) == 3):
            raise TypeError("States have to be given as list with three entries.")
        if not (isinstance(stateList[hl.STATE], hl.TimeDependentVector)
           and np.array_equal(stateList[hl.STATE].times, self._simTimes)):
           raise TypeError("Forward variable needs to be TDV over simulation times.")

        cost = 0 
        for i in self._simTimeInds:
            currentFwdVar = stateList[hl.STATE].data[i]

            for currentData in self.d[i]:
                self._projMat.mult(currentFwdVar, self._currentProjState)     
                self._currentProjState.axpy(-1., currentData)
                cost += self._currentProjState.inner(self._currentProjState)

        cost *= 1./ (2*self.noise_variance)
        return cost

    #-----------------------------------------------------------------------------------------------
    def grad(self, varInd: int, stateList: list, outVec: hl.TimeDependentVector) -> None:
        """Computes gradient of the cost functional with respect to state or parameter function

        Since the cost functional only depends on the forward variable, its variation w.r.t. to the
        parameter function is zero.
        The weak form of the gradient w.r.t. to the state function is

        .. math::
            \frac{1}{\sigma^2} \sum_{i=1}^{n_s}\int_0^T(B^*(Bu-d_i),\tilde{u})\delta_{t_i}dt

        Args:
            varInd (int): Variable w.r.t. to which gradient should be computed
                          (0 = forward, 1 = parameter)
            stateList (list): Forward, parameter and adjoint functions to evaluate gradient for
                              (only forward function is necessary)
            outVec (hl.TimeDependentVector): Gradient, input is overwritten

        Raises:
            ValueError: Checks that variable index is valid
            TypeError: Checks type of provided state list
            TypeError: Checks type of forward function
            TypeError: Checks type of output vector
        """

        if not varInd in [hl.STATE, hl.PARAMETER]:
            raise ValueError("Invalid value for index argument.")
        if not (isinstance(stateList, list) and len(stateList) == 3):
            raise TypeError("States have to be given as list with three entries.")

        outVec.zero()

        if varInd == hl.STATE:
            if not (isinstance(stateList[hl.STATE], hl.TimeDependentVector)
               and np.array_equal(stateList[hl.STATE].times, self._simTimes)):
                raise TypeError("Forward variable needs to be TDV over simulation times.")
            if not (isinstance(outVec, hl.TimeDependentVector) 
               and np.array_equal(outVec.times, self._simTimes)):
                raise TypeError("Output vector needs to be TDV over simulation times.")

            for i in self._simTimeInds:
                currentFwdVar = stateList[hl.STATE].data[i]

                for currentData in self.d[i]:
                    self._projMat.mult(currentFwdVar, self._currentProjState)    
                    self._currentProjState.axpy(-1., currentData)
                    self._projMat.transpmult(self._currentProjState, self._currentFullState)
                    outVec.data[i].axpy(1, self._currentFullState)

                outVec.data[i] *= 1./(self.noise_variance * self._dt)
        else:
            pass

    #-----------------------------------------------------------------------------------------------
    def setLinearizationPoint(self, stateList: list, 
                              gauss_newton_approx: Optional[bool]=False) -> None:
        """ Sets linearization point for quadratic approximation

        This function does nothing, since the cost functional is quadratic in the forward function.

        Args:
            stateList (list): Linearization point
            gauss_newton_approx (Optional[bool], optional): 
                Determines if Gauss-Newton approximation shall be employed. Defaults to False.
        """

        pass

    #-----------------------------------------------------------------------------------------------
    def apply_ij(self, 
                 iInd: int,
                 jInd: int,
                 direction: Union[fe. GenericVector, hl.TimeDependentVector],
                 outVec: hl.TimeDependentVector) -> None:
        """Applies second variation with respect to state and/or parameter functions

        Since the cost functional only depends on the forward variable, its variation w.r.t. to the
        parameter function is zero.
        The weak form of the gradient w.r.t. to the state function is

        .. math::
            \frac{1}{\sigma^2} \sum_{i=1}^{n_s}\int_0^T(B^*Bu,\tilde{u})\delta_{t_i}dt

        Args:
            iInd (int): Variable index for first variation (0 = forward, 1 = parameter)
            jInd (int): Variable index for second variation (0 = forward, 1 = parameter)
            direction (Union[fe. GenericVector, hl.TimeDependentVector]):
                Direction for which second variation is applied
            outVec (hl.TimeDependentVector): Result vector, input is overwritten

        Raises:
            ValueError: Checks that variable index is valid
            TypeError: Checks type of forward function
            TypeError: Checks type of result vector
        """

        if not all(ind in [hl.STATE, hl.PARAMETER] for ind in [iInd, jInd]):
            raise ValueError("Invalid value for index arguments.")

        outVec.zero()

        if iInd == hl.STATE and jInd == hl.STATE:
            if not (isinstance(direction, hl.TimeDependentVector) 
               and np.array_equal(direction.times, self._simTimes)):
                raise TypeError("Forward variable needs to be TDV over simulation times.")
            if not (isinstance(outVec, hl.TimeDependentVector) 
               and np.array_equal(outVec.times, self._simTimes)):
                raise TypeError("Output vector needs to be TDV over simulation times.")

            for i in self._simTimeInds:
                currentObsTimeInds = self._obsMap[i]
                currentFwdVar = direction.data[i]
                numObs = currentObsTimeInds.size

                if numObs > 0:
                    self._projMat.mult(currentFwdVar, self._currentProjState)
                    self._projMat.transpmult(self._currentProjState, outVec.data[i])
                    outVec.data[i] *= numObs/(self.noise_variance * self._dt)
        else:
            pass

    #-----------------------------------------------------------------------------------------------
    def _project_obs_to_sim_grid(self) -> list[np.ndarray]:
        """Sets up mapping between observation times and simulation time grid
        
        For every simulation time point (except the last), this algorithm associates all observation
        times that lie within the current grid point and the next with that grid point. This mapping
        can later be used to accumulated data on the simulation time grid.
        """

        observationMap = []
        for simInd, t in enumerate(self._simTimes[:-1]):
            tNext = self._simTimes[simInd+1]
            obsTimeIndsOnInterval = self._obsTimeInds[(self._obsTimes >= t) 
                                                    & (self._obsTimes < tNext)]
            observationMap.append(obsTimeIndsOnInterval)
        upperBoundPoints = self._obsTimeInds[(self._obsTimes == self._simTimes[-1])]
        observationMap.append(upperBoundPoints)

        return observationMap

    #-----------------------------------------------------------------------------------------------
    def _assign_obs_to_sim_intervals(self, obsVec: hl.timeDependentVector) \
        -> list[list[fe.GenericVector]]:
        """Assigns data to simulation time grid
        
        Using the previously assembled mapping, this routine assigns data from arbitrary observation
        times to their associated points on the simulation time grid.
        """

        assert np.array_equal(obsVec.times, self._obsTimes), \
            "Time points of the given vector need to match observation times."

        effDataVec = []
        for indSim in self._simTimeInds:
            obsTimeInds = self._obsMap[indSim]

            currentObs = []
            for indObs in obsTimeInds:
                currentObs.append(obsVec.data[indObs])
            effDataVec.append(currentObs)

        return effDataVec

    #-----------------------------------------------------------------------------------------------
    @property
    def d(self) -> list[list[fe.GenericVector]]:
        if self._data is None:
            raise ValueError("Property has not been initialized.")
        return self._data

    @property
    def noise_variance(self) -> float:
        if self._noiseVar is None:
            raise ValueError("Property has not been initialized.")
        return self._noiseVar

    #-----------------------------------------------------------------------------------------------
    @d.setter
    def d(self, data: Union[hl.TimeDependentVector, np.ndarray]) -> None:
        if isinstance(data, np.ndarray):
            data = utils.nparray_to_tdv(self._obsTimes, data)
        elif not isinstance(data, hl.TimeDependentVector):
            raise TypeError("Data needs to be given as numpy array or hIPPYlib TDV.")
        self._data = self._assign_obs_to_sim_intervals(data)

    @noise_variance.setter
    def noise_variance(self, noiseVar: float) -> None:
        if not isinstance(noiseVar, float) and noiseVar > 0:
            raise TypeError("Noise variance needs to be given as positive number.")
        self._noiseVar = noiseVar
    

#============================ Transient PDE Variational Problem Class ==============================
class TransientPDEVariationalProblem(hl.PDEProblem):
    """Transient PDE problem

    This class implements transient PDE problems governing the Bayesian inference problem. It
    provides all differentiation capabilities for second order optimization algorithms. The
    implementation assumes a PDE of the form :math: `\frac{\partial u}{\partial t} + f(u,m) = 0`.
    As mentioned in the module header, the spacial operator :math:`f` is further assumed to be 
    linear in :math: `u` and :math: `m`. All implemented routines compute portions of the Lagrangian
    governing the optimization problem of linearized inference.
    The implementation relies on low-level routines from the FEM sub-package for transient solves.

    Methods:
        generate_state: Generates forward variable
        generate_parameter: Generates parameter variable
        generate_parameter_timeseries: Generates series of parameter vectors over simulation times
        init_parameter: Initializes parameter vector to correct size
        solveFwd: Solves forward PDE problem (First variation of Lagrangian w.r.t. adjoint variable)
        solveAdj: Solves forward PDE problem (First variation of Lagrangian w.r.t. forward variable)
        evalGradientParameter: Computes Gradient of the PDE portion of the Lagrangian w.r.t.
                               parameter variable
        setLinearizationPoint: Sets point for linearized evaluation of second variations
        solveIncremental: Solves the incremental forward and adjoint problems resulting from a
                          Newton solver step
        apply_ij: Applies second variations necessary for the rhs construction in Newton solves   
    """

    #-----------------------------------------------------------------------------------------------
    def __init__(self,
                 funcSpaces: list[fe.FunctionSpace], 
                 weakFormHandle: Callable,
                 boundConds: list[fe.DirichletBC], 
                 boundCondsHomogeneous: list[fe.DirichletBC],
                 initFunc: Union[Callable, fe.GenericVector],
                 simTimes: np.ndarray) -> None:
        """Constructor

        Reads input and sets up supplementary data structures.

        Args:
            funcSpaces (list[fe.FunctionSpace]): Spaces of forward, parameter and adjoint functions
            weakFormHandle (Callable): Variational form handle
            boundConds (list[fe.DirichletBC]): Boundary conditions of the original pde problem
            boundCondsHomogeneous (list[fe.DirichletBC]): Homogeneous boundary conditions of same
                                                          type as the original BCs
            initFunc (Union[Callable, fe.GenericVector]): Initial condition of the PDE problem
            simTimes (np.ndarray): Time point grid

        Raises:
            TypeError: Checks type of function space list
            TypeError: Checks type of variational form handle
            TypeError: Checks boundary conditions
            TypeError: Checks type of simulation time array
            ValueError: Checks that simulation times are evenly spaced
        """

        if not (isinstance(funcSpaces, list) and len(funcSpaces) == 3 
           and all(isinstance(space, fe.FunctionSpace) for space in funcSpaces)):
            raise TypeError("Functions spaces need to be provided in list of three entries.")
        if not callable(weakFormHandle):
            raise TypeError("Weak form handle needs to be callable Object.")
        if not all(isinstance(bcList, list) for bcList in [boundConds, boundCondsHomogeneous]):
            raise TypeError("Boundary conditions have to be given in list format.")
        if not all((isinstance(bc, fe.DirichletBC) for bc in bcList) 
                   for bcList in [boundConds, boundCondsHomogeneous]):
            raise TypeError("All boundary conditions mus be FEniCS DirichletBC.")
        if not isinstance(simTimes, np.ndarray):
            raise TypeError("Simulation times need to be given as numpy array.")
        if not np.allclose(np.diff(simTimes), np.diff(simTimes)[0]):
            raise ValueError("Simulation time array needs to be evenly spaced.")

        self._linPointFwd = None
        self._linPointAdj = None
        self._linPointParam = None
        self._solverIncrFwd = None
        self._solverIncrAdj = None
        self._gaussNewtonApprox = None

        self._funcSpaces = funcSpaces
        self._weakFormHandle = weakFormHandle
        self._boundConds = boundConds
        self._boundCondsHomogeneous = boundCondsHomogeneous

        self._simTimes = simTimes
        self._simTimeInds = np.indices(simTimes.shape).flatten()
        self._revSimTimeInds = np.flip(self._simTimeInds)
        self._dt = simTimes[1] - simTimes[0]

        self._massMatrix = self._construct_mass_matrix()
        self._initSol, self._initSolHomogeneous = self._construct_initial_solution(initFunc)
        
        self._dummyFuncFwd = fe.Function(self._funcSpaces[hl.STATE])
        self._dummyFuncAdj = fe.Function(self._funcSpaces[hl.ADJOINT])
        self._dummyTrialFwd = fe.TrialFunction(self._funcSpaces[hl.STATE])
        self._dummyTrialAdj = fe.TrialFunction(self._funcSpaces[hl.ADJOINT])
        self._dummyTrialParam = fe.TrialFunction(self._funcSpaces[hl.PARAMETER])
        self._dummyTestFwd = fe.TestFunction(self._funcSpaces[hl.STATE])
        self._dummyTestAdj = fe.TestFunction(self._funcSpaces[hl.ADJOINT])
        self._dummyTestParam = fe.TestFunction(self._funcSpaces[hl.PARAMETER])

    #-----------------------------------------------------------------------------------------------
    def generate_state(self) -> hl.TimeDependentVector:
        """Generates forward function vector

        Returns:
            hl.TimeDependentVector: Forward variable
        """

        stateVec = hl.TimeDependentVector(self._simTimes)
        stateVec.initialize(self._massMatrix, 1)
        return stateVec

    #-----------------------------------------------------------------------------------------------
    def generate_parameter(self) -> fe.GenericVector:
        """Generates parameter function vector

        Returns:
            fe.GenericVector: Parameter variable
        """

        paramFunc = fe.Function(self._funcSpaces[hl.PARAMETER])
        paramVec = paramFunc.vector()
        return paramVec
    
    #-----------------------------------------------------------------------------------------------
    def generate_parameter_timeseries(self) -> hl.TimeDependentVector:
        """Generates series of parameter functions over simulation times

        Returns:
            hl.TimeDependentVector: Parameter function series
        """

        paramSeriesVec = hl.TimeDependentVector(self._simTimes)
        for d in paramSeriesVec.data:
            self.init_parameter(d)
        return paramSeriesVec

    #-----------------------------------------------------------------------------------------------
    def init_parameter(self, paramVec: fe.GenericVector) -> None:
        """Initialize parameter vector to have correct size

        Args:
            paramVec (fe.GenericVector): In/out, uninitialized parameter vector

        Raises:
            TypeError: Checks input size
        """

        if not isinstance(paramVec, fe.GenericVector):
            raise TypeError("Input needs to be proper FEniCS vector.")
        dummyVec = self.generate_parameter()
        paramVec.init(dummyVec.local_size())

    #-----------------------------------------------------------------------------------------------
    def solveFwd(self, forwardSol: hl.TimeDependentVector, stateList: list) -> None:
        """Solves the transient forward problem

        This routine solves the forward pde problem resulting from the variation of the Lagrangian
        w.r.t. to the adjoint variable. The  variation is simply given as the original PDE constraint,
        an initial value problem. For a given parameter function, it yields the forward solution.

        Args:
            forwardSol (hl.TimeDependentVector): Result vector
            stateList (list): Current state (only parameter variable is relevant)

        Raises:
            TypeError: Checks type of result vector
        """

        if not (isinstance(forwardSol, hl.TimeDependentVector)
           and np.array_equal(forwardSol.times, self._simTimes)):
            raise TypeError("Solution vector needs to be TDV over simulation times.") 
        
        forwardSol.zero()
        lhsMatrix, rhsVector =  self._assemble_forward(stateList)
        solver = fe.LUSolver(lhsMatrix)

        solveSettings = {"init_sol": self._initSol,
                         "time_step_size": self._dt,
                         "mass_matrix": self._massMatrix,
                         "rhs": rhsVector,
                         "bcs": self._boundConds}

        femFunctions.solve_transient(self._simTimeInds, solver, solveSettings, forwardSol)

    #-----------------------------------------------------------------------------------------------
    def solveAdj(self, 
                 adjointSol: hl.TimeDependentVector, 
                 stateList: list,
                 adjointRhs: hl.TimeDependentVector) -> None:
        """Solves the transient adjoint problem

        The adjoint problem results from the first variation of the Lagrangian w.r.t. to the forward
        variable. It is a final value problem with homogeneous final value and boundary conditions.
        Accordingly, the it needs to be solved backwards in time. For given parameter and forward
        function, the adjoint problem yields the adjoint variable, which corresponds to the Lagrange
        multiplier of the PDE constraint in the optimization setting. 

        Args:
            adjointSol (hl.TimeDependentVector): Result vector
            stateList (list): Current state (forward and parameter variables are relevant)
            adjointRhs (hl.TimeDependentVector): RHS, typically negative gradient of the misfit

        Raises:
            TypeError: Checks type of result vector
            TypeError: Checks type of rhs vector
        """

        if not (isinstance(adjointSol, hl.TimeDependentVector)
           and np.array_equal(adjointSol.times, self._simTimes)):
            raise TypeError("Solution vector needs to be TDV over simulation times.")
        if not (isinstance(adjointRhs, hl.TimeDependentVector)
           and np.array_equal(adjointRhs.times, self._simTimes)):
            raise TypeError("RHS vector needs to be TDV over simulation times.")

        adjointSol.zero()
        lhsMatrix = self._assemble_adjoint(stateList)
        solver = fe.LUSolver(lhsMatrix)
        
        solveSettings = {"init_sol": self._initSolHomogeneous,
                         "time_step_size": self._dt,
                         "mass_matrix": self._massMatrix,
                         "rhs": adjointRhs,
                         "bcs": self._boundCondsHomogeneous}

        femFunctions.solve_transient(self._revSimTimeInds, solver, solveSettings, adjointSol)

    #-----------------------------------------------------------------------------------------------
    def evalGradientParameter(self, stateList: list, grad: fe.GenericVector) -> None:
        """Evaluates gradient of the pde-related portion of the Lagrangian w.r.t. parameter function

        For given function, parameter and adjoint variables, this routine determines the gradient
        necessary for finding a descent direction in the optimization procedure.

        Args:
            stateList (list): Current "points"
            grad (fe.GenericVector): Gradient vector

        Raises:
            TypeError: Checks type of the state list
            TypeError: Checks validity of the state vector
            TypeError: Checks validity of the adjoint vector
            TypeError: Checks validity of the parameter vector
        """

        if not (isinstance(stateList, list) and len(stateList) == 3):
            raise TypeError("States have to be given as list with three entries.")
        if not (isinstance(stateList[hl.STATE], hl.TimeDependentVector)
           and np.array_equal(stateList[hl.STATE].times, self._simTimes)):
           raise TypeError("Forward variable needs to be TDV over simulation times.")
        if not (isinstance(stateList[hl.ADJOINT], hl.TimeDependentVector)
           and np.array_equal(stateList[hl.STATE].times, self._simTimes)):
           raise TypeError("Adjoint variable needs to be TDV over simulation times.")
        if not isinstance(grad, fe.GenericVector):
            raise TypeError("Parameter variable needs to be FEniCS vector.")
        
        forwardVec = stateList[hl.STATE]
        adjointVec = stateList[hl.ADJOINT]
        paramVec = stateList[hl.PARAMETER]
        paramFunc = hl.vector2Function(paramVec, self._funcSpaces[hl.PARAMETER])
        gradTimeSeries = self.generate_parameter_timeseries()

        for i in self._simTimeInds:
            forwardFunc = hl.vector2Function(forwardVec.data[i], self._funcSpaces[hl.STATE])
            adjointFunc = hl.vector2Function(adjointVec.data[i], self._funcSpaces[hl.ADJOINT])
            weakForm = self._weakFormHandle(forwardFunc, paramFunc, adjointFunc)
            gradVec = fe.assemble(fe.derivative(weakForm, paramFunc, self._dummyTestParam))
            gradTimeSeries.data[i] = gradVec

        self._integrate_time_trapezoidal(gradTimeSeries, grad)

    #-----------------------------------------------------------------------------------------------
    def setLinearizationPoint(self, 
                              stateList: list, 
                              gaussNewtonApprox: Optional[bool]=False) -> None:
        """Sets linearization point for evaluation of second variations (Newton step)

        This routines sets up the necessary structure for a Newton step about a given linearization
        point. In particular, it constructs the solver structures to compute the step increments for
        the forward and adjoint variables. Other structures could be pre-assembled as well, but they
        are instead computed on-the-fly in the respective sub-routines. This approach favors a 
        smaller memory footprint over performance.

        Args:
            stateList (list): Linearization point
            gaussNewtonApprox (Optional[bool], optional): Use Gauss-Newton approximation.
                                                          Defaults to False

        Raises:
            TypeError: Checks type of the state list
            TypeError: Checks type of the Gauss-Newton parameter
        """

        if not (isinstance(stateList, list) and len(stateList) == 3):
            raise TypeError("States have to be given as list with three entries.")
        if not isinstance(gaussNewtonApprox, bool):
            raise TypeError("Gauss Newton approximation needs to be given as boolean parameter.")

        self._gaussNewtonApprox = gaussNewtonApprox
        self._linPointFwd = []
        self._linPointAdj = []
        for i in self._simTimeInds:
            self._linPointFwd.append(hl.vector2Function(stateList[hl.STATE].data[i], 
                                                        self._funcSpaces[hl.STATE]))
            self._linPointAdj.append(hl.vector2Function(stateList[hl.ADJOINT].data[i], 
                                                        self._funcSpaces[hl.ADJOINT]))
        self._linPointParam = hl.vector2Function(stateList[hl.PARAMETER], 
                                                 self._funcSpaces[hl.PARAMETER])

        weakForm = self._weakFormHandle(self._dummyFuncFwd, self._linPointParam, self._dummyFuncAdj)
        hessAdjFwdForm = fe.derivative(fe.derivative(weakForm, self._dummyFuncAdj, self._dummyTestAdj),
                                       self._dummyFuncFwd, self._dummyTrialFwd)
        hessFwdAdjForm = fe.derivative(fe.derivative(weakForm, self._dummyFuncFwd, self._dummyTestFwd),
                                       self._dummyFuncAdj, self._dummyTrialAdj)
        
        solverMatFwdIncr, _ = femFunctions.assemble_transient(self._funcSpaces[hl.STATE],
                                                              self._dt,
                                                              hessAdjFwdForm,
                                                              self._boundCondsHomogeneous)
        solverMatAdjIncr, _ = femFunctions.assemble_transient(self._funcSpaces[hl.ADJOINT],
                                                              self._dt,
                                                              hessFwdAdjForm,
                                                              self._boundCondsHomogeneous)

        self._solverIncrFwd = fe.LUSolver(solverMatFwdIncr)
        self._solverIncrAdj = fe.LUSolver(solverMatAdjIncr)
      
    #-----------------------------------------------------------------------------------------------
    def solveIncremental(self,
                         solVec: hl.TimeDependentVector,
                         rhsVec: hl.TimeDependentVector,
                         isAdj: bool) -> None:
        """Solve incremental forward and adjoint problems for Newton step

        The incremental forward problem is an initial value problem, which is solved forward in time.
        In contrast, the incremental adjoint problem is a final value problem, which needs to be 
        solved backwards in time. Both problems obey homogeneous initial/final value and boundary
        conditions, since they solve for increments only.

        Args:
            solVec (hl.TimeDependentVector): Solution vector (forward or adjoint)
            rhsVec (hl.TimeDependentVector): RHS vector
            isAdj (bool): Decides if solve is forward or adjoint

        Raises:
            TypeError: Checks type of solution vector
            TypeError: Checks type of rhs vector
            TypeError: Checks type of isAdj parameter
        """

        if not (isinstance(solVec, hl.TimeDependentVector)
           and np.array_equal(solVec.times, self._simTimes)):
            raise TypeError("Solution vector needs to be TDV over simulation times.")
        if not (isinstance(rhsVec, hl.TimeDependentVector)
           and np.array_equal(rhsVec.times, self._simTimes)):
            raise TypeError("RHS vector needs to be TDV over simulation times.")
        if not isinstance(isAdj, bool):
            raise TypeError("isAdj needs to be boolean parameter.")
        
        solveSettings = {"init_sol": self._initSolHomogeneous,
                         "time_step_size": self._dt,
                         "mass_matrix": self._massMatrix,
                         "rhs": rhsVec,
                         "bcs": self._boundCondsHomogeneous}
        if isAdj:
            femFunctions.solve_transient(self._revSimTimeInds,
                                         self._solverIncrAdj,
                                         solveSettings,
                                         solVec)
        else:
            femFunctions.solve_transient(self._simTimeInds,
                                         self._solverIncrFwd,
                                         solveSettings,
                                         solVec)

    #-----------------------------------------------------------------------------------------------
    def apply_ij(self, 
                 iInd: int,
                 jInd: int,
                 direction: Union[fe.GenericVector, hl.TimeDependentVector],
                 outVec: Union[fe.GenericVector, hl.TimeDependentVector]) -> None:
        """Apply second variations of the pde-related Lagrangian part

        This routines applies the second variation of the Lagrangian in a given direction. It is
        merely a wrapper that calls different subroutines depending on the provided index pair.
        If the Gauss-Newton approximation has been enabled, the cross-terms of the Hessian of
        forward and parameter function are automatically set to zero.

        Args:
            iInd (int): Variable index for first variation (forward = 0, parameter = 1, adjoint = 2)
            jInd (int): Variable index for second variation (forward = 0, parameter = 1, adjoint = 2)
            direction (Union[fe.GenericVector, hl.TimeDependentVector]):
                Direction in which second variation is applied
            outVec (Union[fe.GenericVector, hl.TimeDependentVector]): Result vector

        Raises:
            ValueError: Checks for valid index combinations
        """
        
        if (self._gaussNewtonApprox and [iInd, jInd] in [[hl.PARAMETER, hl.STATE],
                                                         [hl.STATE, hl.PARAMETER]]):
            outVec.zero()
        elif [iInd, jInd] == [hl.STATE, hl.STATE]:
            self._apply_hess_state_state(direction, outVec)
        elif [iInd, jInd] == [hl.PARAMETER, hl.PARAMETER]:
            self._apply_hess_param_param(direction, outVec)
        elif [iInd, jInd] == [hl.PARAMETER, hl.STATE]:
            self._apply_hess_param_other(direction, outVec, isForward=True)
        elif [iInd, jInd] == [hl.PARAMETER, hl.ADJOINT]:
            self._apply_hess_param_other(direction, outVec, isForward=False)  
        elif [iInd, jInd] == [hl.STATE, hl.PARAMETER]:
            self._apply_hess_other_param(direction, outVec, isForward=True)
        elif [iInd, jInd] == [hl.ADJOINT, hl.PARAMETER]:
            self._apply_hess_other_param(direction, outVec, isForward=False)
        else:
            raise ValueError(f"Function not supported for the "
                              "combination of indices '{iInd}, {jInd}'")  

    #-----------------------------------------------------------------------------------------------
    def _construct_mass_matrix(self):
        """Constructs FEM mass matrix"""

        forwardVar = fe.TrialFunction(self._funcSpaces[hl.STATE])
        adjointVar = fe.TestFunction(self._funcSpaces[hl.ADJOINT])
        massMatrixFunctional = forwardVar * adjointVar * fe.dx
        massMatrix = fe.assemble(massMatrixFunctional)
        assert isinstance(massMatrix, fe.Matrix), \
            "Object is not a valid FEniCS matrix."

        return massMatrix

    #-----------------------------------------------------------------------------------------------
    def _construct_initial_solution(self, initFunc: Union[fe.GenericVector, Callable]) \
        -> Tuple[fe.GenericVector]:
        """Sets original and homogeneous initial conditions"""

        if isinstance (initFunc, fe.GenericVector):
            initSolFwd = initFunc
        elif callable(initFunc):
            initSolFwd = utils.pyfunc_to_fevec(initFunc, self._funcSpaces[hl.STATE])
        else:
            raise TypeError("Initial condition needs to be FEniCS vector or callable object.")  

        initSolAdj = fe.Function(self._funcSpaces[hl.ADJOINT])
        initSolAdj = initSolAdj.vector()

        return initSolFwd, initSolAdj

    #-----------------------------------------------------------------------------------------------
    def _assemble_forward(self, stateList: list) -> Tuple[fe.Matrix, fe.GenericVector]:
        """Assembles matrix & vector for forward transient solve"""

        assert (isinstance(stateList, list) and len(stateList) == 3), \
            "States have to be given as list with three entries."
        
        paramVar = hl.vector2Function(stateList[hl.PARAMETER], self._funcSpaces[hl.PARAMETER])
        weakForm = self._weakFormHandle(self._dummyTrialFwd, paramVar, self._dummyTestAdj)
        lhsMatrix, rhsConst = femFunctions.assemble_transient(self._funcSpaces[hl.STATE],
                                                              self._dt,
                                                              weakForm,
                                                              self._boundConds)
        
        rhsVector = hl.TimeDependentVector(self._simTimes)
        rhsVector.initialize(self._massMatrix, 0)
        for i in self._simTimeInds:
            rhsVector.data[i].axpy(1, rhsConst)

        return lhsMatrix, rhsVector

    #-----------------------------------------------------------------------------------------------
    def _assemble_adjoint(self, stateList: list) -> fe.Matrix:
        """Assembles matrix & vector for adjoint transient solve"""

        assert (isinstance(stateList, list) and len(stateList) == 3), \
            "States have to be given as list with three entries."
             
        paramFunc = hl.vector2Function(stateList[hl.PARAMETER], self._funcSpaces[hl.PARAMETER])
        weakForm = self._weakFormHandle(self._dummyFuncFwd, paramFunc, self._dummyTrialAdj)
        adjointForm = fe.derivative(weakForm, self._dummyFuncFwd, self._dummyTestFwd)

        lhsMatrix, _ = femFunctions.assemble_transient(self._funcSpaces[hl.STATE],
                                                       self._dt,
                                                       adjointForm,
                                                       self._boundCondsHomogeneous)

        return lhsMatrix

    #-----------------------------------------------------------------------------------------------
    def _integrate_time_trapezoidal(self, 
                                    tdVec: hl.TimeDependentVector, 
                                    resultVec: fe.GenericVector) -> None:
        """Integration of vectors over time with trapezoidal method"""

        assert (isinstance(tdVec, hl.TimeDependentVector)
           and np.array_equal(tdVec.times, self._simTimes)), \
            "Input vector needs to be TDV over simulation times."
        assert isinstance(resultVec, fe.GenericVector), \
            "Result vector needs to be FEniCS vector."

        resultVec.zero()
        resultVec.axpy(0.5*self._dt, tdVec.data[0])
        resultVec.axpy(0.5*self._dt, tdVec.data[-1])
        for i in self._simTimeInds[1:-1]:
            resultVec.axpy(self._dt, tdVec.data[i])

    #-----------------------------------------------------------------------------------------------
    def _apply_hess_state_state(self, 
                                direction: hl.TimeDependentVector, 
                                outVec: hl.TimeDependentVector) -> None:
        """Apply second variation with respect to forward & forward functions
        
        Under the assumption that the pde problem is linear in the forward variable, this is zero.
        """

        assert (isinstance(direction, hl.TimeDependentVector) 
           and np.array_equal(direction.times, self._simTimes)), \
           "Direction variable needs to be TDV over simulation times."
        assert (isinstance(outVec, hl.TimeDependentVector) 
           and np.array_equal(outVec.times, self._simTimes)), \
           "Output vector needs to be TDV over simulation times."
        outVec.zero()

    #-----------------------------------------------------------------------------------------------
    def _apply_hess_param_param(self, 
                                direction: fe.GenericVector, 
                                outVec: fe.GenericVector) -> None:
        """Apply second variation with respect to parameter & parameter functions
        
        Under the assumption that the pde problem is linear in the parameter variable, this is zero.
        """

        assert isinstance(direction, fe.GenericVector), \
           "Direction needs to be FEniCS vector."
        assert isinstance(outVec, fe.GenericVector), \
           "Output vector needs to be FEniCS vector."
        outVec.zero()

    #-----------------------------------------------------------------------------------------------
    def _apply_hess_param_other(self, 
                                direction: hl.TimeDependentVector, 
                                outVec: fe.GenericVector,
                                isForward: bool) -> None:
        """Apply second variation with respect to parameter & forward/adjoint functions
        
        The application direction is a time-dependent vector. The result is a time-independent
        vector, which is obtained through integration over time.
        """

        assert (isinstance(direction, hl.TimeDependentVector) 
           and np.array_equal(direction.times, self._simTimes)), \
           "Direction variable needs to be TDV over simulation times."
        assert isinstance(outVec, fe.GenericVector), \
           "Output vector needs to be FEniCS vector."
        outVec.zero()

        if isForward:
            derivativeInd = 0
            trialDir = self._dummyTrialFwd
        else:
            derivativeInd = 1
            trialDir = self._dummyTrialAdj

        resultTimeSeries = self.generate_parameter_timeseries()
        
        for i in self._simTimeInds:
            forwardFunc = self._linPointFwd[i]
            adjointFunc = self._linPointAdj[i]
            weakForm = self._weakFormHandle(forwardFunc, self._linPointParam, adjointFunc)
            derivativeFuncs = [forwardFunc, adjointFunc]

            hessForm = fe.derivative(fe.derivative(weakForm, self._linPointParam, self._dummyTestParam),
                                     derivativeFuncs[derivativeInd], trialDir)
            hessMat = fe.assemble(hessForm)
            hessMat.mult(direction.data[i], resultTimeSeries.data[i])
            
        self._integrate_time_trapezoidal(resultTimeSeries, outVec)

    #-----------------------------------------------------------------------------------------------
    def _apply_hess_other_param(self, 
                                direction: fe.GenericVector, 
                                outVec: hl.TimeDependentVector,
                                isForward: bool) -> None:
        """Apply second variation with respect to forward/adjoint & parameter functions
        
        The application direction is time-independent. The application to a series of Hessian
        matrices yields a time-dependent vector.
        """

        assert isinstance(direction, fe.GenericVector), \
           "Direction vector needs to be FEniCS vector."
        assert (isinstance(outVec, hl.TimeDependentVector)
           and np.array_equal(outVec.times, self._simTimes)), \
           "Output vector needs to be TDV over simulation times."
        outVec.zero()
        
        if isForward:
            derivativeInd = 0
            testDir = self._dummyTestFwd
        else:
            derivativeInd = 1
            testDir = self._dummyTestAdj

        for i in self._simTimeInds: 
            forwardFunc = self._linPointFwd[i]
            adjointFunc = self._linPointAdj[i]
            weakForm = self._weakFormHandle(forwardFunc, self._linPointParam, adjointFunc)
            derivativeFuncs = [forwardFunc, adjointFunc]

            hessForm = fe.derivative(fe.derivative(weakForm, derivativeFuncs[derivativeInd], testDir),
                                     self._linPointParam, self._dummyTrialParam)
            hessMat = fe.assemble(hessForm)
            hessMat.mult(direction, outVec.data[i])
