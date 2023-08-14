"""Postprocessing module

This module is mainly concerned with the visualization of inference results. It comprises matplotlib
wrappers to conveniently plot the respective data. Generally, the module offers to access levels.
Firstly, the Postprocessor class provides high-level plotting capabilities specifically designed
for the inference results for stochastic processes. In addition, low-level functions offer more
flexible plotting routines for functions, points and intervals in a standardized format. Figure
settings end annotations are provided via settings dictionaries. A large number of such dictionaries
for standard visualization procedures is stored in the PostprocessingData class. This data base can
be easily extended.

NOTE: The plotting routines are designed for functions of ONE spacial dimension

Classes:
--------
PostprocessingData: Settings for figure and plot customization
Postprocessor: High-level plotting wrapper

Functions:
----------
create_figure_handle: Returns matplolib figure handle in standardized format
plot_2d: Simple 2D plot with customized layout
plot_histogram: Histogram plot with customized layout
plot_functions: Plots list of functions
plot_points: Plots list of point arrays
plot_intervals: Plots list of interval arrays
plot_3d: Simple 3D plot with customized layout
plot_contour: Contour plot
plot_supplement_2D: Enriches figure with information for 2D plot
plot_supplement_3D: Enriches figure with information for 3D plot
"""

#====================================== Preliminary Commands =======================================
import os
import copy
import warnings
import numpy as np
from typing import Any, Optional, Union
from matplotlib import cm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from .utilities import general as utils, logging

#plt.rcParams.update({"text.usetex": True})

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import hippylib as hl


#====================================== Postprocessing Data ========================================
class PostprocessingData:
    """Data class holding visualization settings

    This class consists of two main dictionaries that in turn hold a collection of settings dicts.
    The figureSettings dictionaries define the configuration of the overall figure or plot window.
    On the other hand, the plotSettings dictionaries allow for the customization and labeling of
    individual functions within a figure.
    The naming of the figure settings follows specific conventions, such that the high-level wrapper
    routines can automatically detect them depending on the use case to visualize:
        - The inference algorithm is appended to the dictionary name as 'linearized' or 'mcmc'
        - Generating equations used for stationary and transient PDEs are registered as 
          'possiblyTransient'
        - If an equation type is possibly transient, 'stat' or 'trans' is appended to the name stem
          depending on the use case
    """

    possiblyTransient = ["fokker_planck"]

    #-----------------------------------------------------------------------------------------------
    figureSettings = {

        "hessian_eigenvalues":            {"outFile": "hessian_ev",
                                           "title": r"$\textrm{Hessian Eigenvalues}$",
                                           "xlabel": r"$\textrm{Index}$", 
                                           "ylabel": r"$\lambda$", 
                                           "yscale": "log", 
                                           "show": None},

        "drift_linearized":               {"outFile": "drift_func_lin", 
                                           "title": r"$\textrm{Drift Function Linearized}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$f(x)$",
                                           "show": None},

        "diffusion_linearized":           {"outFile": "diffusion_func_lin", 
                                           "title": r"$\textrm{Diffusion Function Linearized}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$g^2(x)$",
                                           "show": None},

        "drift_mcmc":                     {"outFile": "drift_func_mcmc", 
                                           "title": r"$\textrm{Drift Function MCMC}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$f(x)$",
                                           "show": None},

        "diffusion_mcmc":                 {"outFile": "diffusion_func_mcmc", 
                                           "title": r"$\textrm{Diffusion Function MCMC}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$g^2(x)$",
                                           "show": None},

        "fokker_planck_stat_linearized":  {"outFile": "stat_distr_lin", 
                                           "title": r"$\textrm{Stationary Distribution Linearized}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$\rho(x)$",
                                           "show": None},

        "fokker_planck_trans_linearized": {"outFile": "trans_distr_lin", 
                                           "title": r"$\textrm{Transient Distribution Linearized}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$\rho(x)$",
                                           "show": None},

        "fokker_planck_stat_mcmc":        {"outFile": "stat_distr_mcmc", 
                                           "title": r"$\textrm{Stationary Distribution MCMC}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$\rho(x)$",
                                           "show": None},

        "fokker_planck_trans_mcmc":       {"outFile": "trans_distr_mcmc", 
                                           "title": r"$\textrm{Transient Distribution MCMC}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$\rho(x)$",
                                           "show": None},
        
        "mean_exit_time_linearized":      {"outFile": "mean_exit_time_lin", 
                                           "title": r"$\textrm{Mean Exit Time Linearized}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$\tau(x)$",
                                           "show": None},

        "mean_exit_time_mcmc":            {"outFile": "mean_exit_time_mcmc", 
                                           "title": r"$\textrm{Mean Exit Time MCMC}$",
                                           "xlabel": r"$x$",
                                           "ylabel": r"$\tau(x)$",
                                           "show": None},

        "qoi_samples":                    {"outFile": "qoi_samples",
                                           "title": r"$\textrm{QOI Samples}$",
                                           "xlabel": r"$\textrm{Sampler Number}$",
                                           "ylabel": r"$\textrm{QOI}$",
                                           "show": None},

        "qoi_autocorrelation":            {"outFile": "qoi_autocorrelation", 
                                           "title": r"$\textrm{QOI Autocorrelation}$",
                                           "xlabel": r"$\textrm{Lag}$",
                                           "ylabel": r"$\textrm{Autocorrelation}$",
                                           "show": None},

        "qoi_histogram":                  {"outFile": "qoi_histogram", 
                                           "title": r"$\textrm{QOI Histogram}$",
                                           "xlabel": r"$\textrm{QOI Value}$",
                                           "ylabel": r"$\textrm{Bin Value}$",
                                           "show": None}
    }     

    #-----------------------------------------------------------------------------------------------
    plotSettings = {
        "hessian_function":               {"linestyle": "-",
                                           "color": "darkred",
                                           "label": None},

        "hessian_points":                 {"color": "tab:blue",
                                           "label": r"$\textrm{Eigenvalues}$",
                                           "marker": "."},

        "parameter_functions":            [{"linestyle": "-",
                                            "color": "tab:blue",
                                            "label": r"$\textrm{Posterior Mean}$"},
                                           {"linestyle": "dotted",
                                            "color": "darkorange",
                                            "label": r"$\textrm{Prior Mean}$"}],

        "parameter_intervals":            [{"color": "tab:blue",
                                            "label": r"$\textrm{95\% MAP Prediction Interval}$"},
                                           {"color": "darkorange",
                                            "label": r"$\textrm{95\% Prior Prediction Interval}$"}],

        "fwd_fem_functions":              [{"linestyle": "-",
                                            "color": "tab:blue",
                                            "label": r"$\textrm{MAP FEM}$"},
                                           {"linestyle": "dotted",
                                            "color": "darkorange",
                                            "label": r"$\textrm{Prior FEM}$"}],

        "exact_points":                   {"color": "darkgreen",
                                           "label": r"$\textrm{Exact}$",
                                           "marker": "."},

        "data_points":                    {"color": "darkred",
                                           "label": r"$\textrm{Data}$",
                                           "marker": "x"},

        "qoi_function":                  [{"linestyle": "-",
                                           "color": "tab:blue",
                                           "label": r"$\textrm{Mean}$"},
                                          {"linestyle": "--",
                                           "color": "darkred",
                                           "label": None}],

        "qoi_points":                     {"color": "tab:blue",
                                           "marker": ".",
                                           "label": None},

        "qoi_intervals":                  {"color": "tab:blue",
                                           "label": r"$\textrm{Confidence Interval}$"}
    }


#====================================== Postprocessor Class ========================================
class Postprocessor:
    """Visualization wrapper class

    This class provides a high-level interface for the specific visualization of inference output
    It relies on the structures in the PostprocessingData class and automatically fetches the
    correct settings depending on the use case.

    NOTE: Some of the visualization methods take many arguments, as they visualize many different 
          data arrays combined. Preferably invoke these methods with key-value pairs for the
          arguments to avoid confusion.

    Methods:
        visualize_parameters: Visualizes parameter function(s) for prior and posterior
        visualize_forward_solution: Visualizes forward solutions associated with prior and posterior
        visualize_hessian_data: Visualizes generalized Hessian eigenvalues
        postprocess_qoi: Processes and visualizes MCMC Quantity of interest
    """

    _printWidth = 35
    _predIntervalSize = 1.96

    _inferenceOpts = ["drift", "diffusion", "all"]
    _algorithms = ["linearized", "mcmc"]
    _subDir = "postprocessing"
    _fileType = "pdf"

    _checkDictParam = {
        "prior_mean": (list, None, False),
        "prior_variance": (list, None, False),
        "posterior_mean": (list, None, False),
        "posterior_variance": (list, None, False),
        "exact": (list, None, True)
    }

    _checkDictForward = {
        "prior": (list, None, False),
        "posterior": (list, None, False),
        "noisy": (list, None, False),
        "exact": (list, None, True),
        "times": (list, None, True)
    }

    #-----------------------------------------------------------------------------------------------
    def __init__(self,
                 show: Optional[bool] = False,
                 logger: Optional[logging.Logger]=None) -> None:
        """Constructor

        Initializes the logger, gets a copy of the settings data and modifies it for the specified
        output.

        Args:
            show (Optional[bool], optional): Determines if plots are shown on screen.
                                             Defaults to False.
            logger (logging.Logger, optional): Logging object, if None a default logger is
                                               constructed. Defaults to None

        Raises:
            TypeError: Checks type of 'show' parameter
        """
        
        if not isinstance(show, bool):
            raise TypeError("Show needs to be provided as boolean.")

        if logger is None:
            self._logger = logging.Logger()
        else:
            if not isinstance(logger, logging.Logger):
                raise TypeError("Given logger does not have correct type.")
            self._logger = logger

        self._logger.print_ljust("")
        self._logger.print_centered("Invoke Postprocessor", "=")
        self._logger.print_ljust("")

        if self._logger.outDir is not None:
            concatDir = os.path.join(self._logger.outDir, self._subDir)
            os.makedirs(concatDir, exist_ok=True)

        self._figureSettings = copy.deepcopy(PostprocessingData.figureSettings)
        for settings in self._figureSettings.values():
            settings["show"] = show
            if self._logger.outDir is not None:
                settings["outFile"] = \
                    os.path.join(self._logger.outDir, self._subDir, settings["outFile"]) \
                    + "." + self._fileType
            else:
                settings["outFile"] = None     
    
    #-----------------------------------------------------------------------------------------------
    def visualize_parameters(self,
                             paramsInferred: str,
                             mode: str,
                             data: dict[str, Any]) -> None:
        """Visualizes parameter function(s) before and after inference

        This routine visualizes the parameter sets resulting from the MAP estimate and the prior.
        The visualization includes the mean as well as variance bands. If available, point data of
        the exact solution can additionally be shown.
        The method allows for the visualization of different parameter options and inference
        logarithms. It is important to note that all data needs to be provided in the form of
        x-y-value pair, as generated as the output of the inference procedures.

        Args:
            paramsInferred (str): 'drift', 'diffusion' or 'all'
            mode (str): 'linearized' or 'mcmc'

        Raises:
            ValueError: Checks that inference parameters are valid
            ValueError: Checks that inference algorithm is valid
            TypeError: Checks that data is provided in the form of lists of two arrays each
        """

        if not paramsInferred in self._inferenceOpts:
            raise ValueError("Inference options are: " +  ', '.join(self._inferenceOpts))
        if mode not in self._algorithms:
            raise ValueError("Inference modes are: " +  ', '.join(self._algorithms))

        utils.check_settings_dict(data, self._checkDictParam)
        for dataList in data.values():
            if dataList[1].ndim == 1:
                dataList[1] = np.expand_dims(dataList[1], 1)
                
        priorMeanData = data["prior_mean"]
        priorVarianceData = data["prior_variance"]
        posteriorMeanData = data["posterior_mean"]
        posteriorVarianceData = data["posterior_variance"]
        if "exact" in data.keys():
            exactData = data["exact"]
        else:
            exactData = None

        for data in (priorMeanData, priorVarianceData, posteriorMeanData, posteriorVarianceData):
            if not len(data) == 2:
                raise TypeError("Need to provide list of two arrays for data functions.")
        if exactData is not None \
        and not len(exactData) == 2:
           raise TypeError("Exact data needs to be provided as list of two arrays.")

        self._logger.print_ljust(f"Visualize Parameter Functions, mode='{mode}'", end="\n\n")

        try:
            if paramsInferred == "all":
                numParams = 2
                figureSettings = [self._figureSettings["drift_" + mode],
                                  self._figureSettings["diffusion_" + mode]]
            else:
                numParams = 1
                figureSettings = [self._figureSettings[paramsInferred + "_" + mode]]
        except:
            KeyError("Could not find figure settings corresponding to input string.")

        funcSettings = PostprocessingData.plotSettings["parameter_functions"]
        intervalSettings = PostprocessingData.plotSettings["parameter_intervals"]
        pointSettings = PostprocessingData.plotSettings["exact_points"]

        priorPredInterval = self._predIntervalSize * np.sqrt(priorVarianceData[1])
        posteriorPredInterval = self._predIntervalSize * np.sqrt(posteriorVarianceData[1])
        xRange = posteriorMeanData[0]

        for i in range(numParams):
            funcData = [[posteriorMeanData[0], posteriorMeanData[1][:, i]], 
                        [priorMeanData[0], priorMeanData[1][:, i]]]
            intervalData = \
                [[posteriorMeanData[0], posteriorMeanData[1][:, i], posteriorPredInterval[:, i]],
                [priorMeanData[0], priorMeanData[1][:, i], priorPredInterval[:, i]]]
        
            ax = create_figure_handle()
            plot_functions(funcData, funcSettings, ax=ax)
            plot_intervals(intervalData, intervalSettings, ax=ax)

            if exactData is not None:
                pointData = [exactData[0], exactData[1][:, i]]
                plot_points(pointData, pointSettings, ax=ax)
            
            ax.legend()
            plot_supplement_2D(ax, xRange, figureSettings[i])

    #-----------------------------------------------------------------------------------------------
    def visualize_forward_solution(self, 
                                   modelType: str, 
                                   mode: str, 
                                   isStationary: bool,
                                   data: dict[str, Any]) -> None:
        """Visualizes forward solution

        This method visualizes the generating PDE solution associated with the mean prior and MAP
        parameter functions. It additionally plots the utilized data points and, if available, the
        exact forwards solution. As for the parameter visualization, different inference algorithms
        are labelled accordingly. The method further supports stationary and transient PDE problems.
        Note that all input data needs to be provided as x-y-value pairs, as generated as the output
        of the inference procedures.

        Args:
            modelType (str): Generating equation used for inference, e.g. 'fokker_planck'
            mode (str): 'linearized' or 'mcmc'
            isStationary (bool): Determines if problem is stationary or transient

        Raises:
            TypeError: Checks for valid model type
            ValueError: Checks for valid inference algorithm
            TypeError: Checks for type of 'isStationary' parameter
            TypeError: Checks validity of data arrays
            KeyError: Checks that valid data settings key can be constructed from given input
        """

        if not isinstance(modelType, str):
            raise TypeError("modelType needs to be provided as string.")
        if mode not in self._algorithms:
            raise ValueError("Inference modes are: " +  ', '.join(self._algorithms))
        if not isinstance(isStationary, bool):
            raise TypeError("isStationary needs to be provided as boolean.")


        utils.check_settings_dict(data, self._checkDictForward)
        priorData = data["prior"]
        posteriorData = data["posterior"]
        randData = data["noisy"]

        if "exact" in data.keys():
            exactData = data["exact"]
        else:
            exactData = None

        if not isStationary:
            if "times" not in data.keys():
                raise KeyError("Need to provide time points for transient model.")
            timePoints = data["times"]

        self._logger.print_ljust(f"Visualize PDE Solution, mode='{mode}'", end="\n\n")

        if isStationary:
            numDataArrays = 2
            if modelType in PostprocessingData.possiblyTransient:
                middleStr = "stat_"
            else:
                middleStr = ""
        else:
            numDataArrays = 3
            middleStr = "trans_"

        if not (isinstance(priorData, list) and len(priorData) == numDataArrays
           and all(isinstance(array, np.ndarray) for array in priorData)):
            raise TypeError(f"Posterior data needs to be provided as list of {numDataArrays} arrays.")
        if not (isinstance(posteriorData, list) and len(posteriorData) == numDataArrays
           and all(isinstance(array, np.ndarray) for array in posteriorData)):
            raise TypeError(f"Posterior data needs to be provided as list of {numDataArrays} arrays.")
        if not (isinstance(randData, list) and len(randData) == numDataArrays
           and all(isinstance(array, np.ndarray) for array in randData)):
            raise TypeError(f"Observations need to be provided as list of {numDataArrays} arrays.")
        if exactData is not None \
           and not (isinstance(exactData, list) and len(exactData) == numDataArrays
           and all(isinstance(array, np.ndarray) for array in exactData)):
            raise TypeError(f"Exact data needs to be provided as list of {numDataArrays} arrays.")  

        try:
            figureSettings = self._figureSettings[modelType + "_" + middleStr + mode]
        except:
            raise KeyError("Could not find figure settings corresponding to input string.")

        if isStationary:
            self._visualize_forward_solution_stationary(figureSettings,
                                                        priorData,
                                                        posteriorData,
                                                        randData,
                                                        exactData)
        else:
            self._visualize_forward_solution_transient(figureSettings,
                                                       timePoints,
                                                       priorData,
                                                       posteriorData,
                                                       randData,
                                                       exactData)

    #-----------------------------------------------------------------------------------------------
    def visualize_hessian_data(self, hessianEigVals: np.ndarray) -> None:
        """Visualizes generalized eigenvalues of the MAP Hessian

        Args:
            hessianEigVals (np.ndarray): Eigenvalue array

        Raises:
            TypeError: Checks input type
        """

        self._logger.print_ljust("Visualize Hessian Data", end="\n\n")

        if not isinstance(hessianEigVals, np.ndarray) and hessianEigVals.ndim == 1:
            raise TypeError("Eigenvalues need to be provided as 1D numpy array.")

        figureSettings = self._figureSettings["hessian_eigenvalues"]
        funcSettings = PostprocessingData.plotSettings["hessian_function"]
        pointSettings = PostprocessingData.plotSettings["hessian_points"]
        indexValues = np.arange(1, hessianEigVals.size+1)
        sepLine = np.ones(indexValues.size) 
        funcData = [indexValues, sepLine]
        pointData = [indexValues, hessianEigVals]

        ax = create_figure_handle()
        plot_functions(funcData, funcSettings, ax=ax)
        plot_points(pointData, pointSettings, ax=ax)
        ax.legend()
        plot_supplement_2D(ax, indexValues, figureSettings)

    #-----------------------------------------------------------------------------------------------
    def postprocess_qoi(self, qoiTrace: np.ndarray, maxLag: int) -> None:
        """Processes and visualizes MCMC Quantity of interest

        This routine visualizes the qoi points over a sample collection, the corresponding histogram
        and the autocorrelation over the chosen maximum lag.
        The newly compute autocorrelation is additionally saved to a file.

        Args:
            qoiTrace (np.ndarray): QOI over samples
            maxLag (int): Lag range for visualization of autocorrelation

        Raises:
            TypeError: Checks type of qoi trace
            TypeError: Checks that lag parameter is valid
        """

        if not (isinstance(qoiTrace, np.ndarray) and qoiTrace.ndim == 1):
            raise TypeError("QOI data needs to be provided as 1D array.")
        if not (isinstance(maxLag, int) and maxLag < qoiTrace.size):
            raise TypeError("Maximum lag must be smaller than overall number of samples.")

        self._logger.print_ljust("Postprocess QOI Data:", end="\n\n")
        average = np.mean(qoiTrace)
        autoCorrTime, _, _ = hl.integratedAutocorrelationTime(qoiTrace, max_lag=maxLag)
        effSampleSize = qoiTrace.size / autoCorrTime
        
        self._logger.print_ljust(f"{'Mean Value:':<25}{average:<6.1f}")  
        self._logger.print_ljust(f"{'Autocorrelation length:':<25}{autoCorrTime:<6.1f}")
        self._logger.print_ljust(f"{'Effective sample size:':<25}{effSampleSize:<6.1f}")

        figureSPSettings = self._figureSettings["qoi_samples"]
        figureACSettings = self._figureSettings["qoi_autocorrelation"]
        figureHGSettings = self._figureSettings["qoi_histogram"]
        funcSettings = PostprocessingData.plotSettings["qoi_function"]
        intervalSettings = PostprocessingData.plotSettings["qoi_intervals"]
        pointSettings = PostprocessingData.plotSettings["qoi_points"]

        sampleInds = np.arange(1, qoiTrace.size + 1, 1)
        plot_points([sampleInds, qoiTrace], pointSettings, figureSPSettings)

        autoCorrFunc, confInt = sm.tsa.stattools.acf(qoiTrace, nlags=maxLag, alpha=.1, fft=False)
        xLags = np.arange(autoCorrFunc.size)
        conf0 = confInt.T[0]
        conf1 = confInt.T[1]
        funcData = [[xLags, autoCorrFunc], [xLags, np.zeros(xLags.size)]]
        intervalData = [xLags, (conf1+conf0)/2, (conf1-conf0)/2]

        ax = create_figure_handle()
        plot_functions(funcData, funcSettings, ax=ax)
        plot_intervals(intervalData, intervalSettings, ax=ax)
        ax.legend()
        plot_supplement_2D(ax, xLags, figureACSettings)
        plot_histogram(qoiTrace, figureHGSettings)

    #-----------------------------------------------------------------------------------------------
    def _visualize_forward_solution_stationary(self, 
                                               figureSettings: dict[str, Any],
                                               priorData: list[np.ndarray],
                                               posteriorData: list[np.ndarray], 
                                               randData: list[np.ndarray], 
                                               exactData: Optional[list[np.ndarray]] = None) -> None:
        """Visualizes forward solution for stationary problem"""

        funcSettings = PostprocessingData.plotSettings["fwd_fem_functions"]
        pointSettings = [PostprocessingData.plotSettings["data_points"]]
        funcData = [posteriorData, priorData]
        pointData = [randData]
        xRange = posteriorData[0]

        ax = create_figure_handle()
        plot_functions(funcData, funcSettings, ax=ax)
        if exactData is not None:
            pointSettings.append(PostprocessingData.plotSettings["exact_points"])
            pointData.append(exactData)
        
        plot_points(pointData, pointSettings, ax=ax)
        ax.legend()
        plot_supplement_2D(ax, xRange, figureSettings)

    #-----------------------------------------------------------------------------------------------
    def _visualize_forward_solution_transient(self,
                                              figureSettings: dict[str, Any],
                                              timePoints: list[float],
                                              priorData: list[np.ndarray],
                                              posteriorData: list[np.ndarray],
                                              randData: list[np.ndarray],
                                              exactData: list[Optional[np.ndarray]] = None) -> None:
        """Visualizes forward solution for transient problem"""

        if not (isinstance(timePoints, list)
        and all(isinstance(t, (int, float)) for t in timePoints)):
            raise TypeError("Time points need to be given as list of floats.")
        if len (timePoints) > 4:
            raise ValueError("Can only visualize up to four time points.")

        alphaValues = [1, 0.7, 0.4, 0.1]
        priorFuncData = []
        posteriorFuncData = []
        pointData = []
        exactFuncData = []
        priorFuncSettings = []
        posteriorFuncSettings = []
        pointSettings = []
        exactFuncSettings = []
        xRange = posteriorData[0]

        dataTimes = [posteriorData[1], priorData[1], randData[1]]
        if exactData is not None:
            dataTimes.append(exactData[1])

        for i, t in enumerate(timePoints):
            currFuncSettings = copy.deepcopy(PostprocessingData.plotSettings["fwd_fem_functions"])
            currPosteriorFuncSettings = currFuncSettings[0]
            currPriorFuncSettings = currFuncSettings[1] 
            currPointSettings = copy.deepcopy(PostprocessingData.plotSettings["data_points"])
            currExactSettings = copy.deepcopy(PostprocessingData.plotSettings["exact_points"])

            timeInds = []
            for timeArray in dataTimes:
                closeTimePoint = timeArray[np.isclose(timeArray, t)]
                if not len(closeTimePoint) == 1:
                    raise ValueError(f"Time array needs to have exactly one point"
                                     f" close to {t}.")
                timeInds.append(np.argwhere(timeArray == closeTimePoint)[0][0])

            posteriorFuncData.append([posteriorData[0], posteriorData[2][timeInds[0],:]])
            priorFuncData.append([priorData[0], priorData[2][timeInds[1],:]])       
            pointData.append([randData[0], randData[2][timeInds[2],:]])
            
            for settingsDict in (currPriorFuncSettings, currPosteriorFuncSettings, currPointSettings):
                settingsDict["alpha"] = alphaValues[i]
                settingsDict["label"] += rf", $t={t:<6.1f}$"
            posteriorFuncSettings.append(currPosteriorFuncSettings)
            priorFuncSettings.append(currPriorFuncSettings)
            pointSettings.append(currPointSettings)

            if exactData is not None:
                exactFuncData.append([exactData[0], exactData[2][timeInds[3],:]])
                currExactSettings["alpha"] = alphaValues[i]
                currExactSettings["label"] += rf" $t={t:<6.1f}$"
                exactFuncSettings.append(currExactSettings)

        ax = create_figure_handle()
        funcSettings = posteriorFuncSettings + priorFuncSettings
        pointSettings = pointSettings + exactFuncSettings
        funcData = posteriorFuncData + priorFuncData
        pointData = pointData + exactFuncData
        plot_functions(funcData, funcSettings, ax=ax)
        plot_points(pointData, pointSettings, ax=ax)
        ax.legend()
        plot_supplement_2D(ax, xRange, figureSettings)

        
#========================================= Plot functions ==========================================

_checkDict2D = {
    "title": (str, None, False),
    "xlabel": (str, None, False),
    "ylabel": (str, None, False),
    "yScale": (bool, None, True),
    "alpha": (float, [0, 1], True),
    "outFile": ((str, type(None)), None, False),
    "show": (bool, None, False)
}

_checkDictSurf = {
    "title": (str, None, False),
    "xlabel": (str, None, False),
    "ylabel": (str, None, False),
    "zlabel": (str, None, False),
    "stride": (list, None, False),
    "outFile": ((str, type(None)), None, False),
    "show": (bool, None, False)
}

_checkDictContour = {
    "title": (str, None, False),
    "xlabel": (str, None, False),
    "ylabel": (str, None, False),
    "cbarlabel": (str, None, False),
    "levels": (int, [1, 1e10], False),
    "outFile": ((str, type(None)), None, False),
    "show": (bool, None, False)
}

_checkDictInfoFunc = {
    "linestyle": ((tuple, str), None, False),
    "color": (str, None, False),
    "label": ((str, type(None)), None, False)
}

_checkDictInfoOther = {
    "color": (str, None, False),
    "label": ((str, type(None)), None, False)
}

_figSize = [5, 5]
_dpiSize = 150
_pointSize = 10

#---------------------------------------------------------------------------------------------------
def create_figure_handle():
    """Returns a blank figure handle"""

    _, axis = plt.subplots(figsize=_figSize, dpi=_dpiSize)
    return axis
    
#---------------------------------------------------------------------------------------------------
def plot_2d(xValues: np.ndarray, yValues: np.ndarray, settings: dict[str, Any]) -> None:
    """Plots 2D function

    Args:
        xValues (np.ndarray): x-values
        yValues (np.ndarray): y-values
        settings (dict[str, Any]): Figure settings

    Raises:
        TypeError: Checks type of x- and y-values 
        ValueError: Checks that x- and y-values are valid
    """

    if not ( isinstance(xValues, np.ndarray) and isinstance(yValues, np.ndarray) ):
        raise TypeError("x and y values need to be numpy arrays.")
    if not ( xValues.ndim == yValues.ndim == 1 and xValues.size == yValues.size):
        raise ValueError("x and y values need to be one-dimensional arrays of same size.")
    utils.check_settings_dict(settings, _checkDict2D)

    ax = create_figure_handle()
    ax.plot(xValues, yValues)
    plot_supplement_2D(ax, xValues, settings)

#---------------------------------------------------------------------------------------------------
def plot_histogram(dataValues: np.ndarray, settings: dict[str, Any]) -> None:
    """Plots histogram

    Args:
        dataValues (np.ndarray): Data points
        settings (dict[str, Any]): Figure settings

    Raises:
        TypeError: Checks input array type
    """

    if not isinstance(dataValues, np.ndarray):
        raise TypeError("Data values need to be provided as numpy array.")
    utils.check_settings_dict(settings, _checkDict2D)

    ax = create_figure_handle()
    nBins = max(10, min(30, int(dataValues.size/10)))
    plt.hist(dataValues, nBins)

    ax.set_title(settings['title'])
    ax.set_xlabel(settings['xlabel'])
    ax.set_ylabel(settings['ylabel'])
    ax.grid('on', linestyle='--')
    if settings['outFile'] is not None:
        plt.savefig(settings['outFile'])
    if settings['show']:
        plt.show()

#---------------------------------------------------------------------------------------------------
def plot_functions(functionData: Union[list, list[list]], 
                   functionInfo: Union[list, dict],
                   settings: Optional[dict[str, Any]]=None,
                   ax: Optional[plt.Axes]=None) -> None:
    """Plots a collection of functions
    
    The input data is a list of x-y-pairs. Next to the figure settings, a list of settings for the
    customization of each curve needs to be provided. Thus, the number of function settings needs
    to match the number of data tuples. This function can be used stand-alone, in which case it
    generates and annotates a separate figure. It can also be provided an existing handle to be used
    in a larger procedure.

    Args:
        functionData (Union[list, list[list]]): List of x-y-data pairs
        functionInfo (Union[list, dict]): List of function settings dictionaries
            -> linestyle (str): line style
            -> color (str): Line color
            -> label (str): Function label for legend
        settings (Optional[dict[str, Any]], optional): Figure settings. Only need to be provided if
                                                       this function is used stand-alone.
                                                       Defaults to None
        ax (Optional[plt.Axes], optional): Axis object handle. If provided, data is added to this
                                           figure. Defaults to None

    Raises:
        TypeError: Checks type of data and info lists
        ValueError: Checks that number of function tuples and info dicts match
        ValueError: Checks that either figure settings or axis handle is provided
        ValueError: Checks that x-y-pairs match
    """

    if isinstance(functionInfo, dict):
        functionData = [functionData]
        functionInfo = [functionInfo]
    if not ( isinstance(functionData, list)
         and isinstance(functionInfo, list) ):
        raise TypeError("functionData and functionInfo need to be a list of"
                         "x/y arrays and dictionaries, respectively.")
    if not (len(functionData) == len(functionInfo)):
        raise ValueError("Number of function arrays and info dicts needs to be equal.")
    if (settings == None) and (ax == None):
        raise ValueError("Need to provide figure handle or settings.")

    generateFigure = False
    if ax is None:
        ax = create_figure_handle()
        generateFigure = True

    for (function, info) in zip(functionData, functionInfo):
        xValues = function[0]
        yValues = function[1]
        utils.check_settings_dict(info, _checkDictInfoFunc)
        if not ( xValues.ndim == yValues.ndim == 1 and xValues.size == yValues.size):
            raise ValueError("x and y values need to be one-dimensional arrays of same size.")
        opacity = info.get("alpha", 1)
        ax.plot(xValues,
                yValues,
                linestyle=info['linestyle'],
                color=info['color'],
                alpha=opacity,
                label=info['label'])

    if generateFigure == True:
        if any(info['label'] is not None for info in functionInfo):
            ax.legend()
        xValues = functionData[0][0]
        plot_supplement_2D(ax, xValues, settings)

#---------------------------------------------------------------------------------------------------
def plot_points(pointData: Union[list, list[list]], 
                pointInfo: Union[list, dict],
                settings: Optional[dict[str, Any]]=None,
                ax: Optional[plt.Axes]=None) -> None:
    """Scatter plot collection of point arrays
    
    The input data is a list of x-y-pairs. Next to the figure settings, a list of settings for the
    customization of each point ensemble needs to be provided. Thus, the number of point settings
    needs to match the number of data tuples. This function can be used stand-alone, in which case it
    generates and annotates a separate figure. It can also be provided an existing handle to be used
    in a larger procedure.

    Args:
        pointData (Union[list, list[list]]): List of x-y-pairs
        pointInfo (Union[list, dict]): List of point settings dictionaries
            -> color (str): Point ensemble color
            -> label (str): Point ensemble label for legend
        settings (Optional[dict[str, Any]], optional): Figure settings. Only need to be provided if
                                                       this function is used stand-alone.
                                                       Defaults to None
        ax (Optional[plt.Axes], optional): Axis object handle. If provided, data is added to this
                                           figure. Defaults to None

    Raises:
        TypeError: Checks type of data and info lists
        ValueError: Checks that number of point tuples and info dicts match
        ValueError: Checks that either figure settings or axis handle is provided
        ValueError: Checks that x-y-pairs match
    """

    if isinstance(pointInfo, dict):
        pointData = [pointData]
        pointInfo = [pointInfo]
    if not ( isinstance(pointData, list)
         and isinstance(pointInfo, list) ):
        raise TypeError("pointData and pointInfo need to be a list of"
                         "x/y arrays and dictionaries, respectively.")
    if not (len(pointData) == len(pointInfo)):
        raise ValueError("Number of point arrays and info dicts needs to be equal.")
    if (settings == None) and (ax == None):
        raise ValueError("Need to provide figure handle or settings.")

    generateFigure = False
    if ax is None:
        ax = create_figure_handle()
        generateFigure = True

    for (points, info) in zip(pointData, pointInfo):
        xValues = points[0]
        yValues = points[1]
        utils.check_settings_dict(info, _checkDictInfoOther)
        if not ( xValues.ndim == yValues.ndim == 1 and xValues.size == yValues.size):
            raise ValueError("x and y values need to be one-dimensional arrays of same size.")
        opacity = info.get("alpha", 1)
        ax.scatter(xValues,
                   yValues,
                   marker=info['marker'],
                   s=_pointSize,
                   alpha=opacity,
                   color=info['color'],
                   label=info['label'])

    if generateFigure == True:
        if any(info['label'] is not None for info in pointInfo):
            ax.legend()
        xValues = pointData[0][0]
        plot_supplement_2D(ax, xValues, settings)

#---------------------------------------------------------------------------------------------------
def plot_intervals(intervalData: Union[list, list[list]], 
                   intervalInfo: Union[list, dict],
                   settings: Optional[dict[str, Any]]=None,
                   ax: Optional[plt.Axes]=None) -> None:
    """Plots collection of confidence intervals

    The input data is a list of x-mean-deviation pairs. Next to the figure settings, a list of
    settings for the customization of each curve needs to be provided. Thus, the number of function
    settings needs to match the number of data tuples. This function can be used stand-alone, in
    which case it generates and annotates a separate figure. It can also be provided an existing
    handle to be used in a larger procedure.

    Args:
        intervalData (Union[list, list[list]]): List of tuple with x-values, mean and interval size
        intervalInfo (Union[list, dict]): List of interval settings dicts
            -> color (str): Interval color
            -> label (str): Interval label for legend
        settings (Optional[dict[str, Any]], optional): Figure settings. Only need to be provided if
                                                       this function is used stand-alone.
                                                       Defaults to None
        ax (Optional[plt.Axes], optional): Axis object handle. If provided, data is added to this
                                           figure. Defaults to None

    Raises:
        TypeError: Checks type of data and info lists
        ValueError: Checks that number of interval tuples and info dicts match
        ValueError: Checks that either figure settings or axis handle is provided
        ValueError: Checks that x-mean-interval size pairs match
    """

    if isinstance(intervalInfo, dict):
        intervalData = [intervalData]
        intervalInfo = [intervalInfo]
    if not ( isinstance(intervalData, list)
         and isinstance(intervalInfo, list) ):
        raise TypeError("intervalData and intervalInfo need to be a list of"
                         "x/mean/dev arrays and dictionaries, respectively.")
    if not (len(intervalData) == len(intervalInfo)):
        raise ValueError("Number of interval arrays and info dicts needs to be equal.")
    if (settings == None) and (ax == None):
        raise ValueError("Need to provide figure handle or settings.")

    generateFigure = False
    if ax is None:
        ax = create_figure_handle()
        generateFigure = True

    for (interval, info) in zip(intervalData, intervalInfo):
        xValues = interval[0]
        meanValues = interval[1]
        devValues = interval[2]
        utils.check_settings_dict(info, _checkDictInfoOther)
        if not ( xValues.ndim == meanValues.ndim == devValues.ndim == 1
             and xValues.size == meanValues.size == devValues.size):
            raise ValueError("x mean and dev values need to be one-dimensional arrays of same size.")

        ax.fill_between(xValues,
                        (meanValues-devValues),
                        (meanValues+devValues),
                        alpha=.25,
                        color=info['color'],
                        label=info['label'])
    
    if generateFigure == True:
        if any(info['label'] is not None for info in intervalInfo):
            ax.legend()
        plot_supplement_2D(ax, intervalData[0][0], settings)     

#---------------------------------------------------------------------------------------------------
def plot_3d(xValues: np.ndarray,
            yValues: np.ndarray,
            zValues: np.ndarray,
            settings: dict[str, Any]) -> None:
    """Plots function over 2D domain

    The x-, y- and z-values need to be defined over meshgrid.

    Args:
        xValues (np.ndarray): x-values
        yValues (np.ndarray): y-values
        zValues (np.ndarray): z-values
        settings (dict[str, Any]): Figure settings

    Raises:
        TypeError: Checks type of x-, y- and z-values
        ValueError: Checks that x-, y- and z-values match in shape
    """

    if not ( isinstance(xValues, np.ndarray)
         and isinstance(yValues, np.ndarray)
         and isinstance(zValues, np.ndarray) ):
        raise TypeError("x, y and z values need to be numpy arrays.")
    if not ( xValues.ndim == yValues.ndim == zValues.ndim == 2
         and xValues.shape == yValues.shape == zValues.shape):
        raise ValueError("x, y and z values need to be two-dimensional arrays of same shape.")
    utils.check_settings_dict(settings, _checkDictSurf)

    fig = plt.figure(figsize=_figSize, dpi=_dpiSize)
    ax = fig.add_subplot(1, 1, 1, projection='3d', proj_type = 'ortho')
    ax.plot_surface(xValues, yValues, zValues,
                    cmap=cm.jet, linewidth=0, antialiased=True, alpha=0.75,
                    rstride=settings['stride'][0], cstride=settings['stride'][1])
    ax.set_zlabel(settings['zlabel'])
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    plot_supplement_3D(ax, xValues, yValues, settings) 

#---------------------------------------------------------------------------------------------------
def plot_contour(xValues: np.ndarray, 
                 yValues: np.ndarray, 
                 zValues: np.ndarray,
                 settings: dict[str, Any]) -> None:
    """ Plots a contour function over 2D domain

    The x-, y- and z-values need to be defined over meshgrid.

    Args:
        xValues (np.ndarray): x-values
        yValues (np.ndarray): y-values
        zValues (np.ndarray): z-values
        settings (dict[str, Any]): Figure settings

    Raises:
        TypeError: Checks type of x-, y- and z-values
        ValueError: Checks that x-, y- and z-values match in shape
    """

    if not ( isinstance(xValues, np.ndarray)
         and isinstance(yValues, np.ndarray)
         and isinstance(zValues, np.ndarray) ):
        raise TypeError("x, y and z values need to be numpy arrays.")
    if not ( xValues.ndim == yValues.ndim == zValues.ndim == 2
         and xValues.shape == yValues.shape == zValues.shape ):
        raise ValueError("x, y and z values need to be two-dimensional arrays of same shape.")
    utils.check_settings_dict(settings, _checkDictContour)

    fig = plt.figure(figsize=[_figSize[0]+1, _figSize[1]], dpi=_dpiSize)
    ax = fig.add_subplot(1, 1, 1)
    contPlot = ax.contourf(xValues, yValues, zValues,
               levels=settings['levels'], cmap=cm.jet, antialiased=True)
    cbar = fig.colorbar(contPlot, ax=ax)
    cbar.set_label(settings['cbarlabel']) 
    plot_supplement_3D(ax, xValues, yValues, settings)

#---------------------------------------------------------------------------------------------------
def plot_supplement_2D(axis: plt.axis, xValues: list, settings: dict[str, Any]) -> None:
    """Annotates 2D figure objects, shows figure and saves to file

    Args:
        axis (plt.axis): Figure handle
        xValues (list): x-values to determine plot interval
        settings (dict[str, Any]): Figure settings
            -> title (str): Figure header
            -> xlabel (str): x-axis label
            -> ylabel (str): y-axis label
            -> yScale (log): Scaling of the y-Axis. If not given, linear scaling is applied
            -> alpha (float): Opacity of the plotted curve
            -> outfile (str): File to save plot to. If None, the figure is not saved
            -> show (bool): Determines is figure is shown on screen
    """

    utils.check_settings_dict(settings, _checkDict2D)
    axis.set_title(settings['title'])
    axis.set_xlabel(settings['xlabel'])
    axis.set_ylabel(settings['ylabel'])
    axis.grid('on', linestyle='--')
    axis.set_xlim((np.amin(xValues), np.amax(xValues)))
    plt.tight_layout()
    if 'yscale' in settings.keys():
        axis.set_yscale(settings['yscale'])
    if settings['outFile'] is not None:
        plt.savefig(settings['outFile'])
    if settings['show']:
        plt.show()

#---------------------------------------------------------------------------------------------------
def plot_supplement_3D(axis: plt.axis,
                       xValues: np.array,
                       yValues: np.array,
                       settings: dict[str, Any]) -> None:
    """Annotates 3D figure objects, shows figure and saves to file

    Args:
        axis (plt.axis): Figure handle
        xValues (np.array): x-values to determine plot interval
        yValues (np.array): y-values to determine plot interval
        settings (dict[str, Any]): Figure settings
            -> title (str): Figure header
            -> xlabel (str): x-axis label
            -> ylabel (str): y-axis label
            -> outfile (str): File to save plot to. If None, the figure is not saved
            -> show (bool): Determines is figure is shown on screen
    """

    axis.set_title(settings['title'])
    axis.set_xlabel(settings['xlabel'])
    axis.set_ylabel(settings['ylabel'])
    axis.set_xlim((np.amin(xValues), np.amax(xValues)))
    axis.set_ylim((np.amin(yValues), np.amax(yValues)))
    if settings['outFile'] is not None:
        plt.savefig(settings['outFile'])
    if settings['show']:
        plt.show()
