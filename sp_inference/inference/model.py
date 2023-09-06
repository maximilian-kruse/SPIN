""" Linearized inference wrapper

This module contains a wrapper class for the linearized bayesian inference of drift and/or diffusion
functions for stochastic processes. It provides the highest level access to the SP Inference
routines solely via settings dictionaries.
The current model wrapper is tailored towards stochastic processes. However, the basic workflow
might be easily adapted for different scenarios. In this case, it might be beneficial to define a
common base class and derive the 'SDEInferenceModel' class and others from that base.

Classes:
--------
SDEInferenceModel: Wrapper class for stochastic process inference
"""

#====================================== Preliminary Commands =======================================
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

import hippylib.utils.fenics_enhancer as fee
from ..pde_problems import forms, problems
from ..utilities import general as utils
from ..utilities import logging
from . import transient

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl

#=========================================== Base Class ============================================
class SDEInferenceModel:
    """Linearized inference model wrapper for stochastic processes

    This wrapper class evolves around the generic capabilities of the hIPPYlib library for Bayesian
    inference on function spaces, tailored towards the inference of drift and/or diffusion functions
    of stochastic processes. Its high-level interface allows for numerous customizations via settings
    dictionaries. For more detailed information on these options, please refer to the methods that
    utilize them. Most importantly, however, the model allows to infer either the drift function,
    diffusion function, or both for any stochastic process (assuming the available data contains
    the necessary information). Furthermore, the model allows for stationary as well as transient
    pde constraints in the optimization formalism.

    NOTE: The spacial dimension of the pde problems in this model is hard-coded to 1. The core
          functionality is also implemented for 2D problems. However, the processing of results
          requires additional changes for higher dimensions. As an alternative, one can simply
          access the SP Inference library on the lower level of the individual components to compute
          problems of two or three dimensions (with the restrictions of the FEniCS library).

    Attributes:
        funcSpaces (list[Union[fe.FunctionSpace, fe.VectorFunctionSpace]]):
            Function spaces of forward, adjoint and parameter functions in FEM setting
        grPosterior (hl.GaussianLRPosterior):
            hIPPYlib object containing information on linearized posterior/MAP
        inferenceModel (hl.Model):
            hIPPYlib wrapper object combining prior, pde problem and misfit
        misfitFunctional (hl.Misfit):
            Cost functional, also implementing first and second variations
        pdeProblem (hl.PDEProblem):
            PDE variational problem necessary for Bayesian optimization procedure
        prior (hl.modeling.prior.SqrtPrecisionPDE_Prior):
            Prior with predefined precision operator
        simTimes (np.ndarray):
            Simulation time array for transient PDE problem solves

    Methods:
        construct_pde_problem:
            Sets up the variational PDE/FEM problem, relying on FEniCS
        construct prior:
            Sets up hIPPYlib prior with given mean and variance
        construct misfit:
            Sets up cost functional from data information
        compute_gr_posterior:
            Computes Map estimate, including mean and variance for Gaussian approximation
        check_gradient:
            Computes the gradient of the pde problem variation w.r.t. given parameter function
        get_prior_info:
            Gets prior mean function and point-wise variance field
    """
    
    _domainDim = 1
    _printWidth = 35
    _numericTol = fe.DOLFIN_EPS
    _inferenceOpts = ["drift", "diffusion", "all"]
    _priorFileNames = ["mean_prior", "variance_prior", "forward_prior"]
    _mapFileNames = ["mean_map", "variance_map", "forward_map"]
    _subDir = "linearized_inference"

    _checkDictModel = {
        "params_to_infer": (str, None, False),
        "model_type": (str, None, False),
        "is_stationary": (bool, None, False),
        "verbose": (bool, None, True)
    }

    _checkDictPrior = {
        "mean_function": ((Callable, list), None, False),
        "robin_bc": (bool, None, False)
    }

    _checkDictFE = {
        "num_mesh_points": ((int, list), None, False),
        "boundary_locations": (list, None, False),
        "boundary_values": (list, None, False),
        "element_degrees": (list, None, False),  
        "drift_function": (Callable, None, True),
        "squared_diffusion_function": (Callable, None, True)
    }

    _checkDictMisfit = {
        "data_locations": ((int, float, np.ndarray), None, False),
        "data_times": ((int, float, np.ndarray), None, True),
        "data_values": ((int, float, np.ndarray), None, False)
    }

    _checkDictSolver = {
        "initial_guess": (Callable, None, True),
        "rel_tolerance": ((int, float), [_numericTol, 1e10], False),
        "abs_tolerance": ((int, float), [_numericTol, 1e10], False),
        "max_iter": (int, [1, 1e10], False),
        "GN_iter": (int, [1, 1e10], False),
        "c_armijo": ((int, float), [0, 1e10], False),
        "max_backtracking_iter": (int, [1, 1e10], False)
    }

    _checkDictHessian = {
        "num_eigvals": (int, [1, 1e10], False),
        "num_oversampling": (int, [1, 1e10], False)
    }

    _checkDictTransient = {
        "start_time": ((int,float), [-1e10, 1e10], False),
        "end_time": ((int,float), [-1e10, 1e10], False),
        "time_step_size": ((int,float), [_numericTol, 1e10], False),
        "initial_condition": ((Callable, fe.GenericVector), None, False)
    }

    #-----------------------------------------------------------------------------------------------
    def __init__(self,
                 modelSettings: dict[str, Any],       
                 priorSettings: dict[str, Any],
                 feSettings: dict[str, Any], 
                 misfitSettings: dict[str, Any],
                 transientSettings: Optional[dict[str, Any]]=None,
                 logger: Optional[logging.Logger]=None) -> None:
        """Constructor of SDE inference wrapper class

        The constructor is a centralized initialization point. It takes some general settings as 
        well as settings for all model components. Subsequently, it initializes the logger and 
        calls all construction routines.

        Args:
            modelSettings (dict[str, Any]):
                General settings
                -> params_to_infer (str): Inference mode, 'drift', 'diffusion' or 'all'
                -> model_type (str): Generating equation, e.g. 'fokker_planck'. Note that this
                                     has to correspond to a method in the forms module
                -> is_stationary (bool): Specifies stationary or transient model
                -> verbose (bool): Verbosity of the hIPPYlib solver
            priorSettings (dict[str, Any]):
                Prior configuration (see construct_prior)
            feSettings (dict[str, Any]):
                PDE problem configuration (see construct_pde_problem)
            misfitSettings (dict[str, Any]):
                Misfit configuration (see construct_misfit)
            transientSettings (Optional[dict[str, Any]], optional):
                Additional settings for transient models. Defaults to None
            logger (logging.Logger, optional):
                Logging object, if None a default logger (output to console but not to files) is
                generated. Defaults to None

        Raises:
            TypeError: Checks type of logger
            ValueError: Checks for correct inference option
            ValueError: Checks for correct model type
            MiscErrors: Checks Model settings dict
        """

        self._simTimes = None
        self._funcSpaces = None
        self._fixedParamFunction = None
        self._inferenceModel = None
        self._grPosterior = None

        if logger is None:
            self._logger = logging.Logger()
        else:
            if not isinstance(logger, logging.Logger):
                raise TypeError("Given logger does not have correct type.")
            self._logger = logger

        utils.check_settings_dict(modelSettings, self._checkDictModel)

        if modelSettings["params_to_infer"] not in self._inferenceOpts:
            raise ValueError("Inference options are: " +  ', '.join(self._inferenceOpts))
        self.paramsToInfer = modelSettings["params_to_infer"]
        self.isStationary = modelSettings["is_stationary"]
        self._weakForm, self._solutionDim = forms.get_form(modelSettings["model_type"]) 
        
        self._logger.print_centered("Invoke Inference Model", "=")
        self._logger.print_ljust("")

        self.pdeProblem = self.construct_pde_problem(feSettings, transientSettings)
        self.prior = self.construct_prior(priorSettings)
        self.misfitFunctional = self.construct_misfit(misfitSettings)
        self._logger.print_ljust("")

    #-----------------------------------------------------------------------------------------------
    def construct_pde_problem(self, 
                              feSettings: dict[str, Any], 
                              transientSettings: Optional[dict[str, Any]]=None) -> hl.PDEProblem:
        """Sets up variational pde problem

        This method sets up the variational pde problem necessary for the MAP optimization problem.
        For stationary problems, it returns a hIPPYlib object. In the transient case, it yields a
        self-implemented object with the same interface. The FEM problem construction relies on the
        FEMModel object from the pde_problems sub-package. Importantly, the FEM settings need to
        contain a callable to compute the diffusion function if the drift is supposed to be
        inferred, and vice versa. The routine will complain if this is not the case.

        Args:
            feSettings (dict[str, Any]): FEM Problem settings (see FEMProblem)
            transientSettings (dict[str, Any]): Extra settings for transient model. Defaults to None
                -> start_time (int, float): Initial time for transient solve
                -> end_time (int, float): Stopping time for transient solve
                -> time_step_size (int, float): Solver step size (constant)
                -> initial_condition (Callable, fe.GenericVector): Initial state function

        Raises:
            ValueError: Checks for drift (diffusion) callable if (diffusion) drift is inferred
            MiscErrors: Checks FEM settings dict
            MiscErrors: Checks Transient settings dict

        Returns:
            hl.PDEProblem: PDE variational problem
        """

        self._logger.print_ljust("Construct PDE Problem:", width=self._printWidth, end="")
        self._logger.print_dict_to_file("FEM Problem Settings", feSettings)

        if self.paramsToInfer == "drift" and "squared_diffusion_function" not in feSettings.keys():
            raise ValueError("Need to specify diffusion for drift inference.")
        if self.paramsToInfer == "diffusion" and "drift_function" not in feSettings.keys():
            raise ValueError("Need to specify drift for diffusion inference.")

        femProblem = problems.FEMProblem(self._domainDim, self._solutionDim, feSettings)
        self.funcSpaces = self._set_up_funcspaces(femProblem)
        variationalForm = self._construct_variational_form(feSettings, femProblem)

        if self.isStationary:
            pdeProblem = hl.PDEVariationalProblem(self.funcSpaces, 
                                                  variationalForm,
                                                  femProblem.boundCondsForward,
                                                  femProblem.boundCondAdjoint,
                                                  is_fwd_linear=True)
        else:
            tStart = transientSettings["start_time"]
            tEnd = transientSettings["end_time"]
            dt = transientSettings["time_step_size"]
            initFunc = fee.convert_to_np_callable(transientSettings["initial_condition"],
                                                  self._domainDim)
            self.simTimes = np.arange(tStart, tEnd+dt, dt)

            pdeProblem = transient.TransientPDEVariationalProblem(self.funcSpaces, 
                                                                  variationalForm,
                                                                  femProblem.boundCondsForward, 
                                                                  femProblem.boundCondAdjoint,
                                                                  initFunc,
                                                                  self.simTimes)

        assert isinstance(pdeProblem, hl.PDEProblem), \
            "PDE problem has not been constructed correctly."

        self._logger.print_ljust("Successful", end="\n\n")
        return pdeProblem

    #-----------------------------------------------------------------------------------------------
    def construct_prior(self, priorSettings: dict[str, Any]) \
        -> hl.modeling.prior.SqrtPrecisionPDE_Prior:
        """Sets up Prior with mean function and precision operator

        Sets up a Gaussian functional prior :math:`\mu_{prior}~N(m_{prior}, C_{prior})`.
        The precision operator is given as the inverse of an elliptic differential operator,
        :math:`C_{prior} = (\delta I - \gamma\nabla)^{-\alpha}`. :math:`\alpha` is determined by
        the problem dimension, :math:`\delta` and :math: `\gamma` are parameters to control the
        shape of the variance field.

        Args:
            priorSettings (dict[str, Any]): Prior configuration
                -> mean_function (Callable): Prior mean function
                -> gamma (float): Variance field tuning parameter
                -> delta (float): Variance field tuning parameter
                -> robin_bc (bool): Use Robin BCs for precision operator action to mitigate 
                                    boundary effects.

        Raises:
            MiscErrors: Checks Prior settings dict

        Returns:
            hl.modeling.prior.SqrtPrecisionPDE_Prior: Hippylib prior object
        """

        self._logger.print_ljust("Construct Prior:", width=self._printWidth, end="")
        self._logger.print_dict_to_file("Prior Settings", priorSettings)       
        
        funcSpaceParam = self.funcSpaces[hl.PARAMETER]
        if funcSpaceParam.num_sub_spaces() == 1:
            funcSpaceParam = funcSpaceParam.extract_sub_space([0]).collapse()

        priorMeanFunc = fee.convert_to_fe_function(priorSettings["mean_function"],
                                                   funcSpaceParam)
        gamma, delta = hl.BiLaplacianComputeCoefficients(priorSettings["variance"],
                                                         priorSettings["correlation_length"],
                                                         funcSpaceParam)

        prior = hl.BiLaplacianPrior(funcSpaceParam,
                                    gamma,
                                    delta,
                                    mean=priorMeanFunc.vector(),
                                    robin_bc=priorSettings["robin_bc"],
                                    robin_const=priorSettings["robin_bc_const"])

        assert isinstance(prior, hl.modeling.prior.SqrtPrecisionPDE_Prior), \
            "Prior has not been constructed correctly."

        self._logger.print_ljust("Successful", end="\n\n")
        return prior

    #-----------------------------------------------------------------------------------------------
    def construct_misfit(self, misfitSettings: dict[str, Any]) -> hl.Misfit:
        """Sets up misfit functional

        This methods constructs the misfit functional between the given data and the forward
        solution of the associated PDE problem. The resulting object additionally provides first
        and second variation w.r.t. the forward solution, which is necessary for the computation of
        the MAP. The misfit functional is defined as the norm of the difference between data and
        forward solution, weighted by a diagonal noise covariance operator, 
        :math: `||F(m)-d||^2_{\Tau_{noise}^-1}`.
        For transient problems, the misfit is summed over all observation times.

        Args:
            misfitSettings (dict[str, Any]): Misfit configuration
                -> data_locations (float, np.ndarray): Spatial location of data point(s)
                -> data_times (float, np.ndarray): Data recording times, only for transient problems
                -> data_values (float, np.ndarray): Values of data points
                -> data_std (float): Scalar standard deviation characterizing data noise

        Raises:
            MiscErrors: Checks Misfit settings dict
            KeyError: Checks if time points are given for transient problem

        Returns:
            hl.Misfit: Differentiable misfit functional
        """

        utils.check_settings_dict(misfitSettings, self._checkDictMisfit)
        self._logger.print_ljust("Construct Misfit:", width=self._printWidth, end="")     
        
        if self.isStationary:
            misfitFunctional = hl.PointwiseStateObservation(self.funcSpaces[hl.STATE],
                                                            misfitSettings["data_locations"],
                                                            misfitSettings["data_var"])

            data = utils.reshape_to_fe_format(misfitSettings["data_values"])
            misfitFunctional.d.set_local(data)
        else:
            if "data_times" not in misfitSettings.keys():
                raise KeyError("Key 'data_times' missing for transient solve.")
            numSpacePoints = misfitSettings["data_locations"].size
            numTimePoints = misfitSettings["data_times"].size
            misfitFunctional = \
                transient.TransientPointwiseStateObservation(self.funcSpaces[hl.STATE],
                                                             misfitSettings["data_locations"],
                                                             misfitSettings["data_times"],
                                                             self.simTimes,noiseVar
                                                             =misfitSettings["data_var"])
            inputData = misfitSettings["data_values"]
            if self._solutionDim > 1:
                if not inputData.shape == (numSpacePoints, self._solutionDim, numTimePoints):
                    raise ValueError("Data array has wrong shape.")
                structuredData = np.array((numSpacePoints*self._solutionDim, numTimePoints))

                for i in range(numTimePoints):
                    structuredData[:, i] = utils.reshape_to_fe_format(inputData[:,:,i],
                                                                      self._solutionDim)
            else:
                structuredData = inputData
            misfitFunctional.d = structuredData
        
        assert isinstance(misfitFunctional, hl.Misfit), \
            "Misfit functional has not been constructed correctly."

        self._logger.print_ljust("Successful", end="\n\n")
        return misfitFunctional

    #-----------------------------------------------------------------------------------------------
    def compute_gr_posterior(self, 
                             solverSettings: dict[str, Any], 
                             hessianSettings: dict[str, Any]) -> Tuple[list[np.ndarray]]:
        """ Computes the MAP of the linearized inference problem

        This is the main "run" method for conducting the linearized Bayesian inference. It computes 
        the MAP for the Gaussian approximation of the inversion problem through a minimization of
        the log posterior. This optimization is performed by an inexact Newton-CG algorithm, which
        requires the first and second variations of the problem components (pde constraint, 
        likelihood, prior). Furthermore, the algorithm employs a globalization strategy in the form
        of Armijo line search.
        After the determination of the MAP "point", the routine computes a low-rank approximation of
        the posterior Hessian. The approximation relies on the dominant eigenvalues of the Hessian
        and allows for a computationally efficient evaluation of the posterior covariance field at
        the MAP point. A randomized algorithm computes these eigenvalues.
        Further information on the underlying procedures can be found in the hIPPYlib library.
        All relevant data is logged to standardized output files.

        Args:
            solverSettings (dict[str, Any]): Solver configuration
                -> initial_guess (Callable): Solver starting value, defaults to prior mean
                -> rel_tolerance (float): Relative stopping tolerance for gradient reduction
                -> abs_tolerance (float): Absolute stopping tolerance for gradient reduction
                -> max_iter (int): Maximum number of overall iterations
                -> GN_iter (int): Number of Gauss Newton iterations before switching to Newton
                -> c_armijo: Tuning parameter for armijo condition
                -> max_backtracking_iter: Number of reduction steps during line search
            hessianSettings (dict[str, Any]): Reduced Hessian configuration
                -> num_eigvals (int): Number of presumably dominant eigenvalues to determine
                -> num_oversampling (int): Number of EV to oversample for algorithm robustness

        Raises:
            MiscErrors: Checks solver settings dict
            MiscErrors: Checks hessian settings dict

        Returns:
            Tuple[list[np.ndarray]]: Arrays for MAP data (mean, point-wise variance, forward solution)
                                     and reduced Hessian eigenvalues
        """

        utils.check_settings_dict(hessianSettings, self._checkDictHessian)
        self._logger.print_centered("Conduct Linearized Inference", "=")
        self._logger.print_ljust("")
        self._logger.print_dict_to_file("Solver Settings", solverSettings)
        self._logger.print_dict_to_file("Hessian Settings", hessianSettings)
        
        assert isinstance(self.inferenceModel, hl.Model), \
            "Inference model has not been constructed correctly."

        mapForward, mapParam, mapAdjoint = self._compute_map(solverSettings)   
        hessEigVals, hessEigVecs = self._compute_reduced_hessian([mapForward, mapParam, mapAdjoint],
                                                                  hessianSettings)
        self.grPosterior = hl.GaussianLRPosterior(self.prior, hessEigVals, hessEigVecs)

        assert isinstance(self.grPosterior, hl.GaussianLRPosterior), \
            "Posterior has not been constructed correctly."
        self._logger.print_ljust("")

        self.grPosterior.mean = mapParam
        mapPwVariance, _, _ = self.grPosterior.pointwise_variance(method="Randomized", r=200)

        dataStructs = {"file_names": self._mapFileNames,
                       "function_spaces": self.funcSpaces,
                       "solution_data": [mapParam, mapPwVariance, mapForward]}
        if not self.isStationary:
            dataStructs["simulation_times"] = self.simTimes
        
        mapMeanData, mapVarianceData, mapForwardData = \
            self._logger.log_solution(paramsToInfer=self.paramsToInfer,
                                      isStationary=self.isStationary,
                                      dataStructs=dataStructs,
                                      subDir=self._subDir)

        return mapMeanData, mapVarianceData, mapForwardData, hessEigVals

    #-----------------------------------------------------------------------------------------------
    def check_gradient(self, paramFunc: str) -> fe.Function:
        """Computes the gradient of the PDE constraint form w.r.t. a given parameter function

        This method may be used to check the feasibility of the gradient computation, as well as to
        find a suitable initial condition for the solver.

        Args:
            paramFunc (Callable): Evaluation point for the gradient

        Returns:
            fe.Function: Gradient function
        """

        assert isinstance(self.inferenceModel, hl.Model), \
            "Inference model has not been constructed correctly."

        forwardVec = self.inferenceModel.generate_vector(hl.STATE)
        adjointVec = self.inferenceModel.generate_vector(hl.ADJOINT)
        paramFunc = fee.convert_to_fe_function(paramFunc, self.funcSpaces[hl.PARAMETER])
        paramVec = paramFunc.vector()

        self.inferenceModel.solveFwd(forwardVec, [None, paramVec, None])
        self.inferenceModel.solveAdj(adjointVec, [forwardVec, paramVec, None])

        paramGrad = self.inferenceModel.generate_vector(hl.PARAMETER)
        self.pdeProblem.evalGradientParameter([forwardVec, paramVec, adjointVec], paramGrad)
        paramGradFunc = hl.vector2Function(paramGrad, self.funcSpaces[hl.PARAMETER])
        return paramGradFunc

    #-----------------------------------------------------------------------------------------------
    def get_prior_info(self, method: str) -> Tuple[fe.Function, fe.Function]:
        """Returns prior mean and point-wise variance function

        For an efficient computation of the variance field, use the 'Randomized' method. Other
        methods are 'Exact' and 'Estimator'.
        All relevant data is logged to standardized output files.

        Args:
            method (str): Method for variance field computation

        Raises:
            NameError: Checks for correct algorithm name

        Returns:
            Tuple[list[np.ndarray]]: Mean and variance functions on spacial domain
        """

        if method not in ["Exact", "Estimator", "Randomized"]:
            raise NameError("Unknown method for prior variance computation.")

        meanVec = self.prior.mean
        varVec = self.prior.pointwise_variance(method=method)
        forwardVec = self.inferenceModel.generate_vector(hl.STATE)
        self.inferenceModel.solveFwd(forwardVec, [None, self.prior.mean, None])

        dataStructs = {"file_names": self._priorFileNames,
                       "function_spaces": self.funcSpaces,
                       "solution_data": [meanVec, varVec, forwardVec]}
        if not self.isStationary:
            dataStructs["simulation_times"] = self.simTimes
        
        priorMeanData, priorVarianceData, priorForwardData = \
            self._logger.log_solution(paramsToInfer=self.paramsToInfer,
                                      isStationary=self.isStationary,
                                      dataStructs=dataStructs,
                                      subDir=self._subDir)

        return priorMeanData, priorVarianceData, priorForwardData

    #-----------------------------------------------------------------------------------------------
    def _set_up_funcspaces(self, femProblem: problems.FEMProblem) -> None:
        """Sets up function spaces

        The parameter function space is either that of the drift function, the diffusion function,
        or a combination of the two.
        """

        if self.paramsToInfer == "drift":
            funcSpaceParam = femProblem.funcSpaceDrift
        elif self.paramsToInfer == "diffusion":
            funcSpaceParam = femProblem.funcSpaceDiffusion
        elif self.paramsToInfer == "all":
            funcSpaceParam = femProblem.funcSpaceAll
        
        return [femProblem.funcSpaceVar, funcSpaceParam, femProblem.funcSpaceVar]

    #-----------------------------------------------------------------------------------------------
    def _construct_variational_form(self,
                                    feSettings: dict[str, Any],
                                    femProblem: problems.FEMProblem) -> Callable:
        """Constructs a suitable variational form

        Depending on the inference mode, one parameter function in the variational form might be
        fixed by a given function. Otherwise, it is part of the parameter variable.
        """

        if self.paramsToInfer == "drift":
            diff_func = fee.convert_to_fe_function(feSettings["squared_diffusion_function"],
                                                   femProblem.funcSpaceDiffusion)
            self._fixedParamFunction = diff_func
            return  self._form_wrapper_drift
        elif self.paramsToInfer == "diffusion":
            drift_func = fee.convert_to_fe_function(feSettings["drift_function"],
                                                   femProblem.funcSpaceDrift)
            self._fixedParamFunction = drift_func
            return  self._form_wrapper_diffusion
        elif self.paramsToInfer == "all":
            return self._form_wrapper_all

    #-----------------------------------------------------------------------------------------------
    def _form_wrapper_drift(self, forwardVar: Any, paramVar: Any, adjointVar: Any) -> Any:
        """Variational form wrapper for drift function inference"""

        assert self._fixedParamFunction is not None, \
            "Diffusion function is not set, use construct function."
        return self._weakForm(forwardVar, paramVar, self._fixedParamFunction, adjointVar)

    #-----------------------------------------------------------------------------------------------
    def _form_wrapper_diffusion(self, forwardVar: Any, paramVar: Any, adjointVar: Any) -> Any:
        """Variational form wrapper for diffusion function inference"""

        assert self._fixedParamFunction is not None, \
            "Drift function is not set, use construct function."
        return self._weakForm(forwardVar, self._fixedParamFunction, paramVar, adjointVar)

    #-----------------------------------------------------------------------------------------------
    def _form_wrapper_all(self, forwardVar: Any, paramVar: Any, adjointVar: Any) -> Any:
        """Variational form wrapper for inference of drift and diffusion"""

        numElems = int(0.5 * self._domainDim * (self._domainDim + 1)) + self._domainDim
        assert paramVar.ufl_shape[0] == numElems, \
            "Mixed parameter function has wrong shape"

        if self._domainDim == 1:
            driftVar = fe.as_vector((paramVar[0],))
            diffVar = fe.as_matrix(((paramVar[1],),))
        elif self._domainDim == 2:
            driftVar = fe.as_vector((paramVar[0], paramVar[1]))
            diffVar = fe.as_matrix(((paramVar[2], paramVar[3]), (paramVar[3]), paramVar[4]))

        return self._weakForm(forwardVar, driftVar, diffVar, adjointVar)
               
    #-----------------------------------------------------------------------------------------------
    def _compute_map(self, solverSettings: dict[str, Any]) -> list[Any]:
        """Computes the MAP with Newton-CG algorithm
        
        Note that the globalization option "LS" (line search) is hard-coded. An alternative from 
        hIPPYlib is "TS" (trust region).
        """

        self._logger.print_ljust("Solve for MAP:")

        if "initial_guess" in solverSettings.keys():
            initFunc = fee.convert_to_fe_function(solverSettings["initial_guess"],
                                                  self.funcSpaces[hl.PARAMETER])
            initParam = initFunc.vector()
        else:
            initParam = self.prior.mean.copy()

        solver = hl.ReducedSpaceNewtonCG(self.inferenceModel)
        solver.parameters["rel_tolerance"] = solverSettings["rel_tolerance"]
        solver.parameters["abs_tolerance"] = solverSettings["abs_tolerance"]
        solver.parameters["max_iter"] = solverSettings["max_iter"]
        solver.parameters["GN_iter"] = solverSettings["GN_iter"]
        solver.parameters["globalization"] = "LS"
        solver.parameters["LS"]["c_armijo"] = solverSettings["c_armijo"]
        solver.parameters["LS"]["max_backtracking_iter"] = solverSettings["max_backtracking_iter"]
        solver.parameters["print_level"] = self._logger.verbose-1

        [mapSol, mapParam, mapAdj] = solver.solve([None, initParam, None])
        assert isinstance([mapSol, mapParam, mapAdj], list), \
            "Solver has not produced proper solution vectors."

        if solver.converged:
            self._logger.print_ljust("\nConverged in " + str(solver.it) + " iterations.")
        else:
            self._logger.print_ljust("\nNot Converged")
        
        self._logger.print_ljust("Termination reason:", width=self._printWidth, end="")
        self._logger.print_ljust(f"{solver.termination_reasons[solver.reason]}")
        self._logger.print_ljust("Final gradient norm:", width=self._printWidth, end="")
        self._logger.print_ljust(f"{solver.final_grad_norm}")
        self._logger.print_ljust("Final cost:", width=self._printWidth, end="")
        self._logger.print_ljust(f"{solver.final_cost}", end="\n\n")

        return [mapSol, mapParam, mapAdj]

    #-----------------------------------------------------------------------------------------------
    def _compute_reduced_hessian(self, mapVars: list, hessianSettings: dict[str, Any]) \
        -> list[np.ndarray, hl.MultiVector]:
        """Computes low-rank approximation of the posterior Hessian at the MAP point
        
        The chosen algorithm is a randomized double-pass procedure. hIPPYlib also provides a
        single-pass alternative.
        """

        self._logger.print_ljust("Construct Reduced Hessian:", width=self._printWidth, end="")

        self.inferenceModel.setPointForHessianEvaluations(mapVars, gauss_newton_approx=False)
        misfitHessian = hl.ReducedHessian(self.inferenceModel, misfit_only=True)

        randMultiVec = hl.MultiVector(mapVars[1], hessianSettings["num_eigvals"]
                                      + hessianSettings["num_oversampling"])
        hl.parRandom.normal(1., randMultiVec)
        eigVals, eigVecs = hl.doublePassG(misfitHessian, self.prior.R, self.prior.Rsolver,
                                          randMultiVec, hessianSettings["num_eigvals"])

        assert isinstance(eigVals, np.ndarray) and isinstance(eigVecs, hl.MultiVector), \
            "Solver has not produced a proper solution."

        self._logger.print_ljust("Successful", end="\n\n")
        return [eigVals, eigVecs]

    #-----------------------------------------------------------------------------------------------
    @property
    def isStationary(self) -> bool:
        if self._isStationary is None:
            raise ValueError("Property has not been initialized.")
        return self._isStationary

    @property
    def paramsToInfer(self) -> bool:
        if self._paramsToInfer is None:
            raise ValueError("Property has not been initialized.")
        return self._paramsToInfer
    
    @property
    def simTimes(self) -> Union[np.ndarray, None]:
        if self._simTimes is None:
            raise ValueError("Property has not been initialized.")
        return self._simTimes

    @property
    def funcSpaces(self) -> list[Union[fe.FunctionSpace, fe.VectorFunctionSpace]]:
        if self._funcSpaces is None:
            raise ValueError("Property has not been initialized.")
        return self._funcSpaces

    @property
    def prior(self) -> hl.modeling.prior.SqrtPrecisionPDE_Prior:
        if self._prior is None:
            raise ValueError("Property has not been initialized.")
        return self._prior
    
    @property
    def pdeProblem(self) -> hl.PDEProblem:
        if self._pdeProblem is None:
            raise ValueError("Property has not been initialized.")
        return self._pdeProblem

    @property
    def misfitFunctional(self) -> hl.Misfit:
        if self._misfitFunctional is None:
            raise ValueError("Property has not been initialized.")
        return self._misfitFunctional

    @property
    def inferenceModel(self) -> hl.Model:
        if self._inferenceModel is None:
            try:
                self._inferenceModel = hl.Model(self.pdeProblem, self.prior, self.misfitFunctional)
            except:
                raise ValueError("Need to initialize PDE problem, prior and misfit functional for "
                                 "construction of hIPPYlib model.")
        return self._inferenceModel

    @property
    def grPosterior(self) -> hl.GaussianLRPosterior:
        if self._grPosterior is None:
            raise ValueError("Property has not been initialized.")
        return self._grPosterior

    #-----------------------------------------------------------------------------------------------
    @isStationary.setter
    def isStationary(self, isStationary: bool) -> None:
        if not isinstance(isStationary, bool):
            raise TypeError("Input does not have valid data type 'bool'.")
        self._isStationary = isStationary

    @paramsToInfer.setter
    def paramsToInfer(self, paramsToInfer: str) -> None:
        if not isinstance(paramsToInfer, str):
            raise TypeError("Input does not have valid data type 'str'.")
        self._paramsToInfer = paramsToInfer

    @simTimes.setter
    def simTimes(self, simTimes: np.ndarray) -> None:
        if not isinstance(simTimes, np.ndarray):
            raise TypeError("Input does not have valid data type 'np.ndarray'.")
        self._simTimes = simTimes

    @funcSpaces.setter
    def funcSpaces(self, funcSpaces: list[Union[fe.FunctionSpace, fe.VectorFunctionSpace]]) -> None:
        if not isinstance(funcSpaces, list):
            raise TypeError("Input does not have valid data type 'list'.")
        self._funcSpaces = funcSpaces

    @prior.setter
    def prior(self, prior: hl.modeling.prior.SqrtPrecisionPDE_Prior) -> None:
        if not isinstance(prior, hl.modeling.prior.SqrtPrecisionPDE_Prior):
            raise TypeError("Input does not have valid data type "
                            "'modeling.prior.SqrtPrecisionPDE_Prior'.")
        self._prior = prior
    
    @pdeProblem.setter
    def pdeProblem(self, pdeProblem: hl.PDEProblem) -> None:
        if not isinstance(pdeProblem, hl.PDEProblem):
            raise TypeError("Input does not have valid data type 'hl.PDEProblem'")
        self._pdeProblem = pdeProblem

    @misfitFunctional.setter
    def misfitFunctional(self, misfitFunctional: hl.Misfit) ->  None:
        if not isinstance(misfitFunctional, hl.Misfit):
            raise TypeError("Input does not have valid data type 'hl.Misfit'")
        self._misfitFunctional = misfitFunctional

    @inferenceModel.setter
    def inferenceModel(self, inferenceModel: hl.Model) -> None:
        if not isinstance(inferenceModel, hl.Model):
            raise TypeError("Input does not have valid data type 'hl.Model'")
        self._inferenceModel = inferenceModel

    @grPosterior.setter
    def grPosterior(self, grPosterior: hl.GaussianLRPosterior) -> None:
        if not isinstance(grPosterior, hl.GaussianLRPosterior):
            raise TypeError("Input does not have valid data type 'hl.GaussianLRPosterior'")
        self._grPosterior = grPosterior
