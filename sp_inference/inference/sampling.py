""" MCMC Sampler Module

This module provides MCMC sampling capabilities for nonlinear inference problems. In addition to the
sampler itself, it provides QOI (quantity of interest) classes that are derived from a base class.
New classes can be easily implemented by adhering to the derivation procedure.

Classes:
--------
MCMCSampler: MCMC Sampler base on  and MUQ
BaseQOI: Base class for the definition of quantities of interest for sampler monitoring
QOISquaredNorm: Implementation of the parameter function L2-norm as QOI
"""

#====================================== Preliminary Commands =======================================
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

from . import model
from ..utilities import general as utils, logging
import muq.Modeling as mm
import muq.SamplingAlgorithms as ms

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl
    import hippylib2muq as hm

#========================================== Sampler Class ==========================================
class MCMCSampler:
    """MCMC Sampler

    This class implements an MCMC sampler for constructing posterior distributions from nonlinear
    inference problems. Starting point for the sampling is a hIPPYlib posterior object resulting
    from the computation of the MAP point. Configuration of the sampler is done via a settings
    dictionary. The class implements pCN (random walk) and MALA (gradient-informed) algorithms.
    For large problems, MALA is the preferable option.
    The sampling functionalities of this class heavily rely on the MUQ library via hippylib2muq
    bindings. For more detailed information, please refer to the respective projects.

    Methods:
        run: Performs sampling with configured MCMC sampler
    """

    _printWidth = 35
    _mcmcFileNames = ["mean_mcmc", "variance_mcmc", "forward_mcmc"]
    _qoiFile = "qoi_trace"
    _subDir = "mcmc"

    _checkDictSampler = {
        "algorithm": (str, None, False),
        "use_gr_posterior": (bool, None, False),
        "Beta": ((int, float), [0, 1], True),
        "StepSize": ((int, float), [0, 1], True)
    }

    _checkDictRun = {
        "NumSamples": ((int, float), [1, 1e20], False),
        "BurnIn": ((int, float), [0, 1e20], False),
        "init_variance": ((int, float), [0, 1e20], True),
        "init_seed": (int, [0, 1e20], True)
    }
    
    #-----------------------------------------------------------------------------------------------
    def __init__(self,
                 hlModel: model.SDEInferenceModel,
                 samplerSettings: dict[str, Any],
                 logger: Optional[logging.Logger]=None) -> None:
        """Constructor

        The constructor calls the methods to set up the MCMC Kernel and work graph. The latter is
        important to generate an AD tape over the entire sampling problem.

        Args:
            hlModel (model.SDEInferenceModel): Inference model with already generated MAP
            samplerSettings (dict[str, Any]): Sampler configuration
                -> algorithm (str): MCMC algorithm, options are 'pCN' and 'MALA'
                -> use_gr_posterior (bool): If true, proposal distribution will be the computed
                                            Gaussian distribution of the posterior (recommended).
                                            Otherwise, the model prior will be used
                -> beta (float): Tuning parameter for pCN algorithm
                -> StepSize (float): Tuning parameter for MALA algorithm
                -> verbose(bool): Controls verbosity of MUQ sampler

        Raises:
            TypeError: Checks if valid hIPPYlib model has been provided
            MiscErrors: Checks sampler settings dict
        """

        # NOTE: the sampling problem components need to be class attributes with lifetime of the 
        #       sampler class itself for the work graph to function correctly.
        self._samples = None
        self._idparam = None
        self._gaussprior = None
        self._log_gaussprior = None
        self._param2loglikelihood = None
        self._log_target = None
        self._workGraph = None
        self._samplingProblem = None
        self._propDistr = None
        self._proposal = None
        self._MCMCKernel = None

        if not (isinstance(hlModel, model.SDEInferenceModel)
           and hlModel.grPosterior.mean is not None):
            raise TypeError("Need to provide hIPPYlib model for which"
                            " linearized MAP has been evaluated.")

        utils.check_settings_dict(samplerSettings, self._checkDictSampler)

        if logger is None:
            self._logger = logging.Logger()
        else:
            if not isinstance(logger, logging.Logger):
                raise TypeError("Given logger does not have correct type.")
            self._logger = logger

        self._logger.print_ljust("")
        self._logger.print_centered("Invoke Sampling", "=")
        self._logger.print_ljust("")
        self._logger.print_ljust("Construct Sampler:", width=self._printWidth, end="")
        self._logger.print_dict_to_file("Sampler Settings", samplerSettings)
        
        self._inferenceModel = hlModel
        self._workGraph, self._samplingProblem = self._assemble_workgraph(self._inferenceModel)
        self._propDistr, self._proposal, self._MCMCKernel = self._init_kernel(samplerSettings,
                                                                              self._inferenceModel,
                                                                              self._samplingProblem)
        self._logger.print_ljust("Successful", end="\n\n")

    #-----------------------------------------------------------------------------------------------
    def run(self, 
            runSettings: dict[str, Any], 
            initialState: Optional[np.ndarray] = None) -> Tuple[np.ndarray]:
        """Conducts sampling

        This method generates a sample collection from the previously generated sampling object.
        All relevant data is logged to standardized output files.

        Args:
            runSettings (dict[str, Any]): Run configurations
                -> NumSamples: Number of overall samples to compute
                -> BurnIn: Number of Burn-In samples
                -> init_variance: Variance for drawing of initial sampling, optional
                -> init_seed: RNG seed for drawing of initial sampling, optional
            initialState (Optional[np.ndarray], optional): First sample. If not provided,
                It is drawn from the MAP distribution. Defaults to None

        Raises:
            MiscErrors: Checks run settings dict
            KeyError: Checks if initial state or variance & seed is provided
            TypeError: Checks if provided initial state has correct type and shape
            
        Returns:
            ms.SampleCollection: MUQ object holding the sample collection
        """

        utils.check_settings_dict(runSettings, self._checkDictRun)
        self._logger.print_ljust("Start Sampling:", width=self._printWidth, end="\n\n")
        self._logger.print_dict_to_file("Sampling Run Settings", runSettings)

        if self._logger.verbose:
            runSettings["PrintLevel"] = 2
        else:
            runSettings["PrintLevel"] = 0

        if initialState is None:
            if all(setting in runSettings.keys() for setting in ["init_variance", "init_seed"]):
                initialState = self._init_solution(runSettings["init_variance"], 
                                                   runSettings["init_seed"])
            else:
                raise KeyError("Need to provide initial state or variance & seed.")
        elif not (isinstance(initialState, np.ndarray) 
             and initialState.size == self._inferenceModel.funcSpaces[hl.PARAMETER].dim()):
            raise TypeError("Provided initial state needs to be numpy array with dimension"
                            " of the parameter FE space.")

        for key in ["NumSamples", "BurnIn"]:
            runSettings[key] = int(runSettings[key])
        
        # NOTE: The MUQ implementation enforces the whole collection to be computed at once. This
        #       causes memory issues for larger sample sizes. A possible remedy is the repeated call
        #       to the run function with corresponding initial states.
        sampler = ms.SingleChainMCMC(runSettings, [self._MCMCKernel])
        self._samples = sampler.Run([initialState])

        assert isinstance(self._samples, ms.SampleCollection), \
            "Samples are not a valid SampleCollection object"
        self._logger.print_centered("", " ")

        meanVec = utils.nparray_to_fevec(self._samples.Mean())
        varVec = utils.nparray_to_fevec(self._samples.Variance())
        forwardVec = self._inferenceModel.inferenceModel.generate_vector(hl.STATE)
        self._inferenceModel.inferenceModel.solveFwd(forwardVec, [None, meanVec, None])

        dataStructs = {"file_names": self._mcmcFileNames,
                       "function_spaces": self._inferenceModel.funcSpaces,
                       "solution_data": [meanVec, varVec, forwardVec]}
        if not self._inferenceModel.isStationary:
            dataStructs["simulation_times"] = self._inferenceModel.simTimes

        meanData, varianceData, forwardData = \
            self._logger.log_solution(paramsToInfer=self._inferenceModel.paramsToInfer,
                                      isStationary=self._inferenceModel.isStationary,
                                      dataStructs=dataStructs,
                                      subDir=self._subDir)

        return meanData, varianceData, forwardData

    #-----------------------------------------------------------------------------------------------
    def evaluate_qoi(self, qoi: Optional['BaseQOI'] = None) -> np.ndarray:
        """Computes quantity of interest for computed sample collection

        This function can only be called after sampling.
        All relevant data is logged to standardized output files.

        Args:
            qoi (Optional[, optional): QOI to evaluate. If not provided, the 'QOISquaredNorm'
                                       object is used. Defaults to None

        Raises:
            ValueError: Checks if a sample collection has been generated
            TypeError: Check if QOI has been derived from prescribed base class

        Returns:
            np.ndarray: QOI array over all samples
        """
        
        self._logger.print_ljust("Evaluate QOI:", width=self._printWidth, end="")

        if self._samples is None:
            raise ValueError("Need to generate MCMC samples before evaluating QOI.")

        if qoi is None: 
            qoi = QOISquaredNorm(self._inferenceModel.funcSpaces)
        elif not isinstance(qoi, BaseQOI):
            raise TypeError("Provided QOI needs to be valid object derived from BaseQOI.")
        else:
            qoi = qoi(self._inferenceModel.funcSpaces)

        qoiTrace = hm.cal_qoiTracer(self._inferenceModel.pdeProblem, qoi, self._samples)
        assert isinstance(qoiTrace, hl.QoiTracer), "QOI trace is not a valid QoiTracer object"
        self._logger.print_ljust("Successful")
        self._logger.print_arrays_to_file(self._qoiFile, ["QOI"], [qoiTrace.data], self._subDir)

        return qoiTrace.data

    #-----------------------------------------------------------------------------------------------
    def _assemble_workgraph(self, hlModel: model.SDEInferenceModel) -> ms.SamplingProblem:
        """Assembles differentiable work graph for the sampling problem"""
    
        self._idparam = mm.IdentityOperator(hlModel.funcSpaces[hl.PARAMETER].dim())
        self._gaussprior = hm.BiLaplaceGaussian(hlModel.prior)
        self._log_gaussprior = self._gaussprior.AsDensity()
        self._param2loglikelihood = hm.Param2LogLikelihood(hlModel.inferenceModel)
        self._log_target = mm.DensityProduct(2)

        workgraph = mm.WorkGraph()
        workgraph.AddNode(self._idparam, 'Identity')
        workgraph.AddNode(self._log_gaussprior, "Log_prior")
        workgraph.AddNode(self._param2loglikelihood, "Log_likelihood")
        workgraph.AddNode(self._log_target, "Log_target")
        workgraph.AddEdge("Identity", 0, "Log_prior", 0)
        workgraph.AddEdge("Log_prior", 0, "Log_target", 0)
        workgraph.AddEdge("Identity", 0, "Log_likelihood", 0)
        workgraph.AddEdge("Log_likelihood", 0, "Log_target", 1)
        samplingProblem = ms.SamplingProblem(workgraph.CreateModPiece("Log_target"))

        assert isinstance(samplingProblem, ms.SamplingProblem), \
            "Work graph does not constitute a valid SamplingProblem."
        return workgraph, samplingProblem

    #-----------------------------------------------------------------------------------------------
    def _init_kernel(self,
                     samplerSettings: dict[str,Any], 
                     hlModel: model.SDEInferenceModel,
                     samplingProblem: ms.SamplingProblem) -> Tuple:
        """Initializes the MCMC kernel according to provided settings"""

        if samplerSettings["algorithm"] == "pCN":
            if "Beta" not in samplerSettings.keys():
                raise KeyError("Need to provide a value for setting 'Beta' when using pCN.")
            propFunc = ms.CrankNicolsonProposal
            useZeroMean = False

        elif samplerSettings["algorithm"] == "MALA":
            if "StepSize" not in samplerSettings.keys():
                raise KeyError("Need to provide a value for setting 'StepSize' when using MALA.")
            propFunc = ms.MALAProposal
            useZeroMean = True
        else:
            raise ValueError("Unknown option for algorithm, supported are 'pCN' and 'MALA'.")

        if samplerSettings["use_gr_posterior"]:
            propDistr = hm.LAPosteriorGaussian(hlModel.grPosterior, use_zero_mean=useZeroMean)        
        else:
            propDistr = hm.BiLaplaceGaussian(hlModel.prior, use_zero_mean=useZeroMean)
        
        proposal = propFunc(samplerSettings, samplingProblem, propDistr)
        MCMCKernel = ms.MHKernel(samplerSettings, samplingProblem, proposal)
        assert isinstance(MCMCKernel, ms.MHKernel), "Object is not a valid MH Kernel."

        return propDistr, proposal, MCMCKernel

    #-----------------------------------------------------------------------------------------------
    def _init_solution(self, variance: Union[int, float], seed: int) -> np.ndarray:
        """Proposes an initial sample from posterior distribution"""

        if not (isinstance(variance, (int, float)) and variance >= 0):
            raise TypeError("Variance needs to be a non-negative float or integer.")
        if not (isinstance(seed, int) and seed >= 0):
            raise TypeError("Seed needs to be a non-negative integer.")

        rng = hl.Random(seed=seed)
        noise = fe.Vector()
        self._inferenceModel.grPosterior.init_vector(noise, "noise")
        rng.normal(variance, noise)
        hl.parRandom.normal(1., noise)
        priorSample = self._inferenceModel.inferenceModel.generate_vector(hl.PARAMETER)
        posteriorSample = self._inferenceModel.inferenceModel.generate_vector(hl.PARAMETER)
        self._inferenceModel.grPosterior.sample(noise, priorSample, posteriorSample, add_mean=True)
        x0 = hm.dlVector2npArray(posteriorSample)

        assert isinstance(x0, np.ndarray), "Initial state is not a numpy array."
        return x0


#============================================ QOI Class ============================================
class BaseQOI(ABC):
    """Base class for quantities of interest

    With hippylib2muq, QOIs are provided as objects with an eval method, which takes a sample and 
    computes the associated scalar form. This class provides the interface for such objects.
    Its children simply need to define the concrete form to be evaluated.
    """

    #-----------------------------------------------------------------------------------------------
    def __init__(self, funcSpaces: list[fe.FunctionSpace]) -> None:
        """Constructor

        Args:
            funcSpaces (list[fe.FunctionSpace]): [description]

        Raises:
            TypeError: [description]
        """

        if not ( isinstance (funcSpaces, list) and len(funcSpaces) == 3
           and all(isinstance(space, fe.FunctionSpace) for space in funcSpaces) ):
           raise TypeError("Need to provide list of three function spaces for forward variable,"
                           "parameter and adjoint variable.")
        self._funcSpaces = funcSpaces
        self._numParamSubspaces = self._funcSpaces[hl.PARAMETER].num_sub_spaces()

    #-----------------------------------------------------------------------------------------------
    def eval(self, stateList: list) -> float:
        """Evaluates qoi for given state

        Args:
            stateList (list): Sample state list (only parameter function is relevant)

        Returns:
            float: QOI value
        """

        paramVec = hl.vector2Function(stateList[hl.PARAMETER], self._funcSpaces[hl.PARAMETER])
        return fe.assemble(self.form(paramVec))

    #-----------------------------------------------------------------------------------------------
    @abstractmethod
    def form(self, paramVec: fe.Function) -> fe.Form:
        """Variational form that describes the qoi

        This function needs to be overloaded in derived classes.

        Args:
            paramVec (fe.Function): Sample parameter function

        Returns:
            fe.Form: Associated qoi FEniCS form
        """
        pass

#---------------------------------------------------------------------------------------------------
class QOISquaredNorm(BaseQOI):
    """ L2-Norm of the parameter function"""

    def form(self, paramVec: fe.Function):
        if self._numParamSubspaces == 0:
            form = paramVec * paramVec * fe.dx
        else:
            form = 0
            for i in range(self._numParamSubspaces):
                form += paramVec[i] * paramVec[i] * fe.dx

        return form
  