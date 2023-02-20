"""Stochastic processes module

This module implements a collection of simple stochastic processes for artificial data generation.
It allows for the computation of stationary and transient distributions for these processes, as well
as for the mean exit time problem. The processes are implemented in a class hierarchy, all 
concrete implementations inherit their interface and common functionality for an abstract base class.
Further, the naming of the subclasses follows certain conventions:
- Processes with a well-known name are named accordingly
- The other implemented proceses have drift and diffusion function in the form of polynomials.
  Therefore, their name is assembled from the prefixed 'Dr' (drift) and 'Di' (diffusion), following
  by a sequence of numbers indicating the terms in the associated polynomials with their respective
  degrees.

NOTE: The current implementeation is restricted to problems of ONE spatial dimension

Classes:
--------
BaseProcess: Abstract base class for stochastic processes
OUProcess: Ornstein-Uhlenbeck process
Dr3Di0Process: Process with cubic drift and constant diffusion
Dr31Di0Process: Process with cubic + linear drift and constant diffusion
Dr31Di21Process: Process with cubic + linear drift and quadratic + linear diffusion ("Landau-Stuart")

Methods:
--------
get_option_list: Returns list of all implemented processes inheriting from BaseProcess
get_process: Returns class object for given string identifier
"""

#====================================== Preliminary Commands =======================================
import numpy as np
from typing import Any, Callable, Optional, Tuple, Union
from abc import ABC, abstractmethod
from scipy import interpolate
from scipy.special import gamma
from scipy.integrate import quad, cumulative_trapezoid

from .pde_problems import forms as femForms, problems as femProblems
from .utilities import general as utils, interpolation as interp, logging


#============================================ Base Class ===========================================
class BaseProcess(ABC):
    """Abstract base class for stochastic processes

    The base class enforces a uniform interface and implements common functionalities. The spatial
    problem dimension is hard-coded to one.

    Attributes:
    -----------
    driftCoeff (float, list): Parameter(s) of the drift function
    diffusionCoeff (float, list); Parameter(s) of the diffusion function

    Methods:
    --------
    generate_data: Generic interface for data generation
    generate_data_stationary_fpe: Generate noisy data from stationray pdf
    generate_data_mean_exit_time: Generate noisy data from mean exit time function
    generate_data_transient_fpe: Generate noisy data from transient pdf
    compute_transient_distribution_fem: Solve transient FPE with FEM
    compute_transient_distribution_exact: Compute transient pdf from analytical expression
    compute_mean_exit_time: Compute mean exit time for given domain
    compute_stationary_distribution: Compute stationary pdf
    compute_drift: Compute drift function for given points
    compute_squared_diffusion: Compute squared diffusion function for given points
    """

    _sizeDriftCoeff = None
    _sizeDiffusionCoeff = None
    _defaultDriftCoeff = None
    _defaultDiffusionCoeff = None

    _requiredData = ["_sizeDriftCoeff",
                     "_sizeDiffusionCoeff",
                     "_defaultDriftCoeff",
                     "_defaultDiffusionCoeff"]

    _printWidth = 35
    _subDir = "data"
    _statFPEDataFile = "data_stationary_fpe"
    _transFPEDataFile = "data_transient_fpe"
    _METDataFile = "data_met"

    _checkDictStatFPE = {
        "domain_points": ((int, float, np.ndarray), None, False),
        "standard_deviation": ((int, float), [0, 1e10], False), 
        "rng_seed": (int, None, False)
    }

    _checkDictMET = {
        "domain_points": ((int, float, np.ndarray), None, False),
        "standard_deviation": ((int, float), [0, 1e10], False), 
        "rng_seed": (int, None, False),
        "domain_bounds": (list, None, False)
    }

    _checkDictTransFPE = {
        "domain_points": ((int, float, np.ndarray), None, False),
        "time_points": ((int, float, np.ndarray), None, False),
        "standard_deviation": ((int, float), [0, 1e10], False), 
        "rng_seed": (int, None, False),
        "fem_settings": (dict, None, False),
        "solver_settings": (dict, None, False)
    }

    _checkDictFE = {
        "num_mesh_points": ((int, list), None, False),
        "boundary_locations": (list, None, False),
        "boundary_values": (list, None, False),
        "element_degrees": (list, None, False)
    }

    _checkDictTransient = {
        "start_time": ((int, float), [0, 1e10], False),
        "end_time": ((int, float), [0, 1e10], False),
        "time_step_size": ((int, float), [0, 1e10], False),
        "initial_condition": (Callable, None, False)
    }

    #-----------------------------------------------------------------------------------------------
    def __init__(self,
                 driftCoeff: Optional[Union[int,float, list]]=None, 
                 diffusionCoeff: Optional[Union[int,float, list]]=None,
                 logger: Optional[logging.Logger]=None) -> None:
        """Base class constructor

        Args:
            driftCoeff (Optional[Union[int,float, list]], optional):
                Parameterization of the drift function. If None, default values are picked.
                Defaults to None
            diffusionCoeff (Optional[Union[int,float, list]], optional):
                Parameterization of the diffusion function. If None, default values are picked.
                Defaults to None
            logger (logging.Logger, optional): Logging object, if None a default logger is
                                               constructed. Defaults to None
        """

        self._problemDim = 1

        if driftCoeff is None:
            self.driftCoeff = self._defaultDriftCoeff
        else:
            self.driftCoeff = driftCoeff
        if diffusionCoeff is None:
            self.diffusionCoeff = self._defaultDiffusionCoeff
        else:
            self.diffusionCoeff = diffusionCoeff

        if logger is None:
            self._logger = logging.Logger()
        else:
            if not isinstance(logger, logging.Logger):
                raise TypeError("Given logger does not have correct type.")
            self._logger = logger

        self._logger.print_centered(f"Invoke {self.__class__.__name__}", "=")
        self._logger.print_ljust("")
        self._logger.print_ljust(f"Drift Coefficient(s): {self._driftCoeff}")
        self._logger.print_ljust(f"Diffusion Coefficient(s): {self._diffusionCoeff}", end="\n\n")

    #-----------------------------------------------------------------------------------------------
    def __init_subclass__(cls) -> None:
        """Subclass constructor

        Raises:
            AttributeError: Checks that number of drift and diffusion parameters is specified for
                            that process, es well as corresponding default values.
        """

        for requiredData in cls._requiredData:
            if not getattr(cls, requiredData):
                raise AttributeError(f"Can't instantiate class {cls.__name__}"
                                     f" without {requiredData} attribute defined")

    #-----------------------------------------------------------------------------------------------
    def generate_data(self,
                      modelType: str,
                      isStationary: bool,
                      dataStructs: dict[str, Any]) -> Tuple:
        """ Generic interface for data generation

        The data is generated by adding a zero-centered Gaussian noisy of specific variance to the
        computed/exact values. For more details on the different use cases, please refer to the 
        respective sub-functions.

        Args:
            modelType (str): Identifier of the generating equation
            isStationary (bool): Determines if model is stationary
            dataStructs (dict[str, Any]): Information to compute the data. This corresponds to the
                                          function arguments of the called sub-functions.
                All methods require the following data:
                -> domain_points (int, float, np.ndarray): Domain points to compute data for
                -> standard_deviation (int, float): Standard deviation of the data noise
                -> rng_seed (int): Seed for the noise generator
                Transient models additionally require:
                -> fem_settings (dict[str, Any]): FEM model configuration, see the respective
                                                  sub-methods
                -> solver_settings(dict[str, Any]): Configuration of transient solver, see the
                                                    respective sub-methods
                The mean exit time model additionally requires:
                -> domain_bounds [list]: Boundaries of the exit domain


        Raises:
            ValueError: COmplains if given identifiers are invalid.

        Returns:
            Tuple: Perturbed and exact data values at given domain (and time) points
        """

        if modelType == "fokker_planck":
            if isStationary:
                utils.check_settings_dict(dataStructs, self._checkDictStatFPE)
                perturbedValues, exactValues = \
                    self.generate_data_stationary_fpe(dataPoints=dataStructs["domain_points"],
                                                      dataStd=dataStructs["standard_deviation"],
                                                      seed=dataStructs["rng_seed"])
            else:
                utils.check_settings_dict(dataStructs, self._checkDictTransFPE)
                perturbedValues, exactValues = \
                    self.generate_data_transient_fpe(obsDomainPoints=dataStructs["domain_points"],
                                                     obsTimePoints=dataStructs["time_points"],
                                                     dataStd=dataStructs["standard_deviation"],
                                                     seed=dataStructs["rng_seed"],
                                                     feSettings=dataStructs["fem_settings"],
                                                     solverSettings=dataStructs["solver_settings"])
        elif modelType == "mean_exit_time":
            if isStationary:
                utils.check_settings_dict(dataStructs, self._checkDictStatFPE)
                perturbedValues, exactValues = \
                    self.generate_data_mean_exit_time(dataPoints=dataStructs["domain_points"],
                                                      dataStd=dataStructs["standard_deviation"],
                                                      seed=dataStructs["rng_seed"],
                                                      domainBounds=dataStructs["domain_bounds"])
            else:
                raise ValueError("Mean exit time data is always stationary.")
        else:
            raise ValueError("Cannot find data generation function for given model type.")

        return perturbedValues, exactValues

    #-----------------------------------------------------------------------------------------------
    def generate_data_stationary_fpe(self, 
                                     dataPoints: Union[int, float, np.ndarray], 
                                     dataStd: Union[int, float], 
                                     seed: int) -> Tuple:
        """Generates noisy data from stationary pdf of the process

        The data is generated by adding a zero-centered Gaussian noisy of specific variance to the
        computed/exact values.

        Args:
            dataPoints (np.ndarray): Points to compute data values for
            dataStd (Union[int,float]): Standard deviation of the noise to add
            seed (int): Seed for the noise RNG

        Returns:
            np.ndarray: Noisy data
        """

        self._logger.print_ljust("Generate stationary FPE data:", width=self._printWidth, end="")
        exactValues = self.compute_stationary_distribution(dataPoints)
        perturbedValues = self._perturb_data(exactValues, dataStd, seed)
        
        self._logger.print_arrays_to_file(self._statFPEDataFile, 
                                          ["x", "rho(x)"], 
                                          [dataPoints, perturbedValues],
                                          self._subDir)
        self._logger.print_ljust("Successful", end="\n\n")

        return perturbedValues, exactValues

    #-----------------------------------------------------------------------------------------------
    def generate_data_mean_exit_time(self, 
                                     dataPoints: Union[int, float, np.ndarray], 
                                     dataStd: Union[int,float], 
                                     seed: int, 
                                     domainBounds: list, 
                                     numTrapzPoints: Optional[int] = 1000) -> Tuple:
        """Generates noisy data from the mean exit time of the process

        The data is generated by adding a zero-centered Gaussian noisy of specific variance to the
        computed/exact values. The mean exit time is computed as a cumulative integral via the
        trapezoidal rule.

        Args:
            dataPoints (np.ndarray): Points to generate data for
            dataStd (Union[int,float]): Standard deviation of the data noise
            seed (int): Seed for the noise RNG
            domainBounds (list): Domain bounds to compute mean exit tme for
            numTrapzPoints (Optional[int], optional): Number of trapezoidal integration points. 
                                                      Defaults to 1000

        Returns:
            np.ndarray: Noisy data
        """

        self._logger.print_ljust("Generate MET data:", width=self._printWidth, end="")
        exactValues = self.compute_mean_exit_time(dataPoints, domainBounds, numTrapzPoints)
        perturbedValues = self._perturb_data(exactValues, dataStd, seed)
        
        self._logger.print_arrays_to_file(self._METDataFile, 
                                          ["x", "tau(x)"], 
                                          [dataPoints, perturbedValues],
                                          self._subDir)
        self._logger.print_ljust("Successful", end="\n\n")

        return perturbedValues, exactValues

    #-----------------------------------------------------------------------------------------------
    def generate_data_transient_fpe(self,
                                    obsDomainPoints: Union[int, float, np.ndarray],
                                    obsTimePoints: Union[int, float, np.ndarray],
                                    dataStd: Union[int,float], 
                                    seed: int,
                                    feSettings: dict[str, Any],
                                    solverSettings: dict[str, Any]) -> Tuple:
        """Generates noisy data from the stationary pdf of the process

        The data is generated by adding a zero-centered Gaussian noisy of specific variance to the
        computed/exact values. The unperturbed values are found via an FEM simulation, since
        closed-form expressions for the transient pdf are not always available. For every time point,
        the same spacial points are used to generate the data.

        Args:
            obsDomainPoints (np.ndarray): Points to compute data for
            obsTimePoints (Union[int, float, np.ndarray]): Times to compute data for
            dataStd (Union[int,float]): Standard deviation of the data noise
            seed (int): Seed for the noise RNG
            feSettings (dict[str, Any]): FEM setup settings
            solverSettings (dict[str, Any]): Transient solver settings

        Returns:
            np.ndarray: Noisy data in space and time
        """

        self._logger.print_ljust("Generate transient FPE data:", width=self._printWidth, end="")                     
        simTimes, femSol, funcSpace = self.compute_transient_distribution_fem(feSettings, 
                                                                              solverSettings, 
                                                                              convert=False)
        interpHandle = interp.InterpolationHandle(simTimes,
                                                  obsTimePoints,
                                                  obsDomainPoints,
                                                  funcSpace)
        interpSol = interpHandle.interpolate_and_project(femSol)
        interpSol = utils.tdv_to_nparray(interpSol)
        perturbedValues = self._perturb_data(interpSol, dataStd, seed)

        self._logger.log_transient_vector(self._transFPEDataFile,
                                          ["x", "rho(x)"],
                                          [obsDomainPoints, obsTimePoints, perturbedValues],
                                          printInterval=1,
                                          subDir=self._subDir)
        self._logger.print_ljust("Successful", end="\n\n")

        return perturbedValues, interpSol

    #-----------------------------------------------------------------------------------------------
    def compute_transient_distribution_fem(self,
                                           feSettings: dict[str, Any],
                                           solverSettings: dict[str, Any],
                                           convert: Optional[bool]=True)\
                                           -> Tuple:
        """Computes transient pdf by numerically solving the Fokker-Planck equation

        Args:
            feSettings (dict[str, Any]): FEM setup settings
            solverSettings (dict[str, Any]): Transient solver settings
                -> start_time (float): Start time
                -> end_time (float): End time
                -> time_step_size (float): Time step size
                -> initFunc (Callable): Initial condition
            convert (bool): Determines if solution is converted to numpy array
        Returns:
            Tuple: Simulation times, result vector and result variable function space
        """

        utils.check_settings_dict(feSettings, self._checkDictFE)
        utils.check_settings_dict(solverSettings, self._checkDictTransient)

        tStart = solverSettings["start_time"]
        tEnd = solverSettings["end_time"]
        dt = solverSettings["time_step_size"]
        initFunc = solverSettings["initial_condition"]
        simTimePoints = np.arange(tStart, tEnd+dt, dt)
        
        femFormHandle = femForms.VariationalFormHandler.get_form("fokker_planck")
        femModel = femProblems.TransientFEMProblem(self._problemDim, feSettings, simTimePoints)
        femModel.assemble(femFormHandle, self.compute_drift, self.compute_squared_diffusion)       
        femSol = femModel.solve(initFunc, convert=convert)

        return simTimePoints, femSol, femModel.funcSpaceVar

    #-----------------------------------------------------------------------------------------------
    def compute_transient_distribution_exact(spacePoints: Union[int, float, np.ndarray], 
                                             timePoints: Union[int, float, np.ndarray]) \
                                             -> np.ndarray:
        """Computes transient distribution from analyticala expression

        Per default this option is not available. It needs to be implemented in the respective
        subclasses (if possible).

        Raises:
            NotImplementedError: Warns that function needs to be re-implemented in subclass
        """

        raise NotImplementedError("Exact computation not implemented for this process.")

    #-----------------------------------------------------------------------------------------------
    def compute_mean_exit_time(self, 
                               domainPoints: Union[int, float, np.ndarray],
                               domainBounds: list, 
                               numTrapzPoints: Optional[int]=1000)\
                               -> Union[int, float, np.ndarray]:
        """Computes mean exit time of a process. The mean exit time is computed as a cumulative
        integral via the trapezoidal rule.

        Args:
            domainPoints (np.ndarray): Points to compute mean exit time for
            domainBounds (list): Domain to compute mean exit time on
            numTrapzPoints (Optional[int], optional): Number of trapezoidal integration points. 
                                                      Defaults to 1000

        Returns:
            Union[int, float, np.ndarray]: Mean exit time for given points
        """

        metInterpHandle = self._construct_met_interp_handle(*domainBounds, numTrapzPoints)
        metTimes = metInterpHandle(domainPoints)
        metTimes = utils.process_output_data(metTimes)

        return metTimes

    #-----------------------------------------------------------------------------------------------
    def _construct_met_interp_handle(self, 
                                     lowerBound : Union[int, float], 
                                     upperBound: Union[int, float], 
                                     numTrapzPoints: int) -> None:
        """Constructs interpolation handle on domain, function is computed as cumulative integral"""

        if not (isinstance(lowerBound, (int, float)) and isinstance(upperBound, (int, float))):
            raise TypeError("Lower and upper bound need to be provided as int or float.")
        if lowerBound >= upperBound:
            raise ValueError("Upper bound needs to be larger than lower bound.")
        if not (isinstance(numTrapzPoints, int) and numTrapzPoints > 0):
            raise TypeError("Number of integration points needs to be positive integer.")

        domainPoints = np.linspace(lowerBound, upperBound, numTrapzPoints)
        psiVar = 2 * cumulative_trapezoid(self.compute_drift(domainPoints)/
                                          self.compute_squared_diffusion(domainPoints), 
                                          domainPoints, initial=0)
        expPsiVar = np.exp(-psiVar)
        singleIntegral = cumulative_trapezoid(expPsiVar, domainPoints, initial=0)
        innerIntegral = cumulative_trapezoid(np.exp(psiVar)/
                                             self.compute_squared_diffusion(domainPoints),
                                             domainPoints, initial=0)
        doubleIntegral = cumulative_trapezoid(innerIntegral * expPsiVar, domainPoints, initial=0)
        preFactor = 2 * doubleIntegral[-1] / singleIntegral[-1]

        metArray = -2 * doubleIntegral + preFactor * singleIntegral
        metInterpHandle = interpolate.interp1d(domainPoints, metArray, bounds_error=True)

        return metInterpHandle      

    #-----------------------------------------------------------------------------------------------
    def _perturb_data(self, 
                      exactData: np.ndarray, 
                      dataStd: Union[int,float], 
                      seed: int) -> np.ndarray:
        """Perturbs data by adding zero-centered Gaussian noise of defined std"""

        assert ( isinstance(dataStd, (int,float)) and dataStd > 0 ), \
            "Data standard deviation needs to be positive number."
        assert isinstance(seed, int), \
            "RNG seed needs to be an integer."
        exactData = utils.process_input_data(exactData)

        noiseGenerator = np.random.default_rng(seed)
        randIncs = noiseGenerator.normal(0, dataStd, exactData.shape)
        perturbedValues = np.where(exactData + randIncs >= 0,
                                   exactData + randIncs,
                                   exactData - randIncs)

        assert perturbedValues.size == exactData.size, \
            "Exact and perturbed arrays do not have same size"
        perturbedValues = utils.process_output_data(perturbedValues)
        
        return perturbedValues

    #-----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_drift(self, domainPoints: Union[int, float, np.ndarray]) -> np.ndarray:
        """Computes drift function

        This function needs to be implemented in all subclasses. Importantly, it needs to be
        vectorized (take arrays as input).

        Args:
            domainPoints (Union[int, float, np.ndarray]): Points to compute drift function for

        Returns:
            np.ndarray: Drift function values
        """
        pass

    #-----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_squared_diffusion(self, domainPoints: Union[int, float, np.ndarray]) -> np.ndarray:
        """Computes squared diffusion function

        This function needs to be implemented in all subclasses. Importantly, it needs to be
        vectorized (take arrays as input).

        Args:
            domainPoints (Union[int, float, np.ndarray]): Points to compute diffusion function for

        Returns:
            np.ndarray: Squared diffusion function values
        """
        pass
    
    #-----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_stationary_distribution(self, domainPoints: Union[int, float, np.ndarray])\
        -> np.ndarray:
        """Computes stationary pdf via closed form expression.

        This function needs to be implemented in all subclasses. Importantly, it needs to be
        vectorized (take arrays as input).

        Args:
            domainPoints (Union[int, float, np.ndarray]): Points to compute pdf for

        Returns:
            np.ndarray: Stationary pdf values
        """
        pass

    #-----------------------------------------------------------------------------------------------
    @property
    def driftCoeff(self) -> Union[int, float, list]:

        if self._driftCoeff is None: 
            raise ValueError("Property has not been initialized.")
        return self._driftCoeff

    @property
    def diffusionCoeff(self) -> Union[int, float, list]:

        if self._diffusionCoeff is None: 
            raise ValueError("Property has not been initialized.")
        return self._diffusionCoeff

    #-----------------------------------------------------------------------------------------------
    @driftCoeff.setter
    def driftCoeff(self, driftCoeff: Union[int, float, list]) -> None:

        if isinstance(driftCoeff, (int, float)):
            driftCoeffCheck = [driftCoeff]
        elif isinstance (driftCoeff, list):
            driftCoeffCheck = driftCoeff
        else:
            raise TypeError("Drift coefficient needs to be a number or list of numbers.")
        if len(driftCoeffCheck) != self._sizeDriftCoeff:
            raise ValueError(f"Number of drift coefficients has to be {self._sizeDriftCoeff:},"
                             f"But the given argument has size {len(driftCoeffCheck)}")

        self._driftCoeff = driftCoeff

    @diffusionCoeff.setter
    def diffusionCoeff(self, diffusionCoeff: Union[int, float, list]) -> None:

        if isinstance(diffusionCoeff, (int, float)):
            diffusionCoeffCheck = [diffusionCoeff]
        elif isinstance (diffusionCoeff, list):
            diffusionCoeffCheck = diffusionCoeff
        else:
            raise TypeError("Diffusion coefficient needs to be a number or list of numbers.")
        if len(diffusionCoeffCheck) != self._sizeDiffusionCoeff:
            raise ValueError(f"Number of diffusion coefficients has to be {self._sizeDiffusionCoeff:}, "
                             f"but the given argument has size {len(diffusionCoeffCheck)}")

        self._diffusionCoeff = diffusionCoeff


#========================================== Factory Methods ========================================
def get_option_list():
    """Returns all implemented subclasses of BaseProcess/implemented processes"""

    optList = [subClass.__name__ for subClass in BaseProcess.__subclasses__()]
    return optList

def get_process(identifier: str) -> BaseProcess:
    """Returns process type for given identifier

    This method searches all registered subclasses of the base process and returns the class
    object corresponding to the given identifier.

    Args:
        identifier (str): Process type identifier

    Raises:
        TypeError: Checks that identifier is a string
        ValueError: Complains if process type has not been found

    Returns:
        BaseProcess: Class object for given identifier
    """
    if not isinstance(identifier, str):
        raise TypeError("Identifier needs to be string.")

    processType = None
    for subClass in BaseProcess.__subclasses__():
        if subClass.__name__ == identifier:
            processType = subClass
            break

    if processType is None:
        raise ValueError("Could not find process type for provided identifier")

    return processType


#======================================= Ornstein-Uhlenbeck ========================================
class OUProcess(BaseProcess):
    """Ornstein-Uhlenbeck process (linear drift and constant diffusion)"""

    _sizeDriftCoeff = 1
    _sizeDiffusionCoeff = 1
    _defaultDriftCoeff = 1
    _defaultDiffusionCoeff = 1

    #-----------------------------------------------------------------------------------------------
    def compute_drift(self, 
                      domainPoints: Union[int, float,  np.ndarray]) \
                      -> Union[float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        driftValues = -self._driftCoeff * domainPoints
        driftValues = utils.process_output_data(driftValues)

        return driftValues

    #-----------------------------------------------------------------------------------------------
    def compute_squared_diffusion(self, 
                                  domainPoints: Union[int, float, np.ndarray]) \
                                  -> Union[float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        diffusionValues = self._diffusionCoeff**2 * np.ones(domainPoints.shape)
        diffusionValues = utils.process_output_data(diffusionValues)

        return diffusionValues

    #-----------------------------------------------------------------------------------------------
    def compute_stationary_distribution(self, 
                                        domainPoints: Union[int, float, np.ndarray]) \
                                        -> Union[float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        pStat = np.sqrt(self._driftCoeff/(np.pi*self._diffusionCoeff**2)) \
              * np.exp(-self._driftCoeff/self._diffusionCoeff**2 * np.square(domainPoints))
        pStat = utils.process_output_data(pStat)

        return pStat

    #-----------------------------------------------------------------------------------------------
    def compute_transient_distribution_exact(self,
                                             spacePoints: Union[int, float, np.ndarray], 
                                             timePoints: Union[int, float, np.ndarray]) \
                                             -> np.ndarray:

        spacePoints = utils.process_input_data(spacePoints)
        timePoints = utils.process_input_data(timePoints, enforce1D=True)
        
        pdf = np.zeros(timePoints.shape + spacePoints.shape)
        coeffFrac = self._driftCoeff / self._diffusionCoeff**2       
        expr1 = np.sqrt(coeffFrac / (np.pi * (1 - np.exp(-2*self._driftCoeff*timePoints))))
        expr2 = (1 - np.exp(-2*self._driftCoeff*timePoints))
        expr3 = np.square(spacePoints)

        for i, _ in enumerate(timePoints):
            pdf[i, ...] = expr1[i] * np.exp(-coeffFrac * expr3 / expr2[i])

        pdf = utils.process_output_data(pdf)
        return pdf

#=========================================== Cubic Drift ===========================================
class Dr3Di0Process(BaseProcess):
    """Process with cubic drift and constant diffusion"""

    _sizeDriftCoeff = 1
    _sizeDiffusionCoeff = 1
    _defaultDriftCoeff = 1
    _defaultDiffusionCoeff = 1

    #-----------------------------------------------------------------------------------------------
    def compute_drift(self, 
                      domainPoints: Union[int, float, np.ndarray])\
                      -> Union[int, float, np.ndarray]:
        
        domainPoints = utils.process_input_data(domainPoints)
        driftValues = -self._driftCoeff * np.power(domainPoints, 3)
        driftValues = utils.process_output_data(driftValues)

        return driftValues

    #-----------------------------------------------------------------------------------------------
    def compute_squared_diffusion(self, 
                                  domainPoints: Union[int, float, np.ndarray]) \
                                  -> Union[int, float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        diffusionValues = self._diffusionCoeff**2 * np.ones(domainPoints.shape)
        diffusionValues = utils.process_output_data(diffusionValues)

        return diffusionValues

    #-----------------------------------------------------------------------------------------------
    def compute_stationary_distribution(self, 
                                        domainPoints: Union[int, float, np.ndarray]) \
                                        -> Union[int, float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        pStat = 1/gamma(0.25) * np.power(8 * self._driftCoeff / self._diffusionCoeff**2, 0.25) \
              * np.exp(-0.5 * self._driftCoeff / self._diffusionCoeff**2 
              * np.power(domainPoints, 4))
        pStat = utils.process_output_data(pStat)

        return pStat

#=========================== Cubic and Linear Drift, Constant Diffusion ============================
class Dr31Di0Process(BaseProcess):
    """Process with cubic + linear drift and constant diffusion"""

    _sizeDriftCoeff = 2
    _sizeDiffusionCoeff = 1
    _defaultDriftCoeff = [1, 1]
    _defaultDiffusionCoeff = 1

    #-----------------------------------------------------------------------------------------------
    def compute_drift(self, 
                      domainPoints: Union[int, float, np.ndarray]) \
                      -> Union[int, float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        driftValues = -self._driftCoeff[0] * np.power(domainPoints, 3) \
                    + self._driftCoeff[1] * domainPoints
        driftValues = utils.process_output_data(driftValues)

        return driftValues

    #-----------------------------------------------------------------------------------------------
    def compute_squared_diffusion(self, 
                                  domainPoints: Union[int, float, np.ndarray]) \
                                  -> Union[int, float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        diffusionValues = self._diffusionCoeff**2 * np.ones(domainPoints.shape)
        diffusionValues = utils.process_output_data(diffusionValues)

        return diffusionValues

    #-----------------------------------------------------------------------------------------------
    def compute_stationary_distribution(self, 
                                        domainPoints: Union[int, float, np.ndarray]) \
                                        -> Union[int, float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        normFactor, _ = quad(self._fpe_auxiliary_exp_func, -np.inf, np.inf)
        pStat = 1/normFactor * self._fpe_auxiliary_exp_func(domainPoints)
        pStat = utils.process_output_data(pStat)

        return pStat

    #-----------------------------------------------------------------------------------------------
    def _fpe_auxiliary_exp_func(self, domainPoints: np.ndarray) -> np.ndarray:

        expFunc = np.exp(-0.5 * self._driftCoeff[0] / self._diffusionCoeff**2 
                * np.power(domainPoints, 4)
                + self._driftCoeff[1] / self._diffusionCoeff**2 * np.power(domainPoints, 2))
        return expFunc

#===================== Cubic and Linear Drift, Quadratic and Linear Diffusion ======================
class Dr31Di21Process(BaseProcess):
    """Landau-Stuart process (cubic + linear drift and quadratic + linear diffusion)"""
    
    _sizeDriftCoeff = 2
    _sizeDiffusionCoeff = 2
    _defaultDriftCoeff = [1, 1]
    _defaultDiffusionCoeff = [1, 1]

    #-----------------------------------------------------------------------------------------------
    def compute_drift(self, 
                      domainPoints: Union[int, float, np.ndarray]) \
                      -> Union[int, float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        driftValues = -self._driftCoeff[0] * np.power(domainPoints, 3) \
                    + self._driftCoeff[1] * domainPoints
        driftValues = utils.process_output_data(driftValues)

        return driftValues

    #-----------------------------------------------------------------------------------------------
    def compute_squared_diffusion(self, 
                                  domainPoints: Union[int, float, np.ndarray]) \
                                  -> Union[int, float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        diffusionValues = self._diffusionCoeff[0] * np.square(domainPoints) \
                        + self._diffusionCoeff[1] * np.ones(domainPoints.shape)
        diffusionValues = utils.process_output_data(diffusionValues)

        return diffusionValues

    #-----------------------------------------------------------------------------------------------
    def compute_stationary_distribution(self, 
                                        domainPoints: Union[int, float, np.ndarray]) \
                                        -> Union[int, float, np.ndarray]:

        domainPoints = utils.process_input_data(domainPoints)
        normFactor, _ = quad(self._fpe_auxiliary_exp_func, -np.inf, np.inf)
        pStat = 1/normFactor * self._fpe_auxiliary_exp_func(domainPoints)
        pStat = utils.process_output_data(pStat)

        return pStat

    #-----------------------------------------------------------------------------------------------
    def _fpe_auxiliary_exp_func(self, domainPoints: np.ndarray) -> np.ndarray:

        auxCoeff = (self._diffusionCoeff[0]*self._driftCoeff[1] 
                  + self._diffusionCoeff[1]*self._driftCoeff[0] - self._diffusionCoeff[0]**2) \
                 / (self._diffusionCoeff[0]**2)
        expFunc = np.exp(-self._driftCoeff[0]/self._diffusionCoeff[0]*np.square(domainPoints)) \
                * np.power(self._diffusionCoeff[0]*np.square(domainPoints) 
                         + self._diffusionCoeff[1], auxCoeff)    
        return expFunc
