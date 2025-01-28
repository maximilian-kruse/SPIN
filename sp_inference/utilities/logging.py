"""Logging module

This module provides the capabilities for the recording of data in the console and in files. The
logged information includes settings, computation process information and result arrays.

Classes:
--------
Logger: Logging class
"""

# ====================================== Preliminary Commands =======================================
import inspect
import os
import warnings
from typing import Any, Optional, Tuple

import numpy as np

from . import general as utils

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl


# ========================================== Logger Class ===========================================
class Logger:
    """Logging class

    Objects of the logger class print information to the console and save data in a file hierarchy.
    Settings and general information are written into a log file, result arrays are saved in
    corresponding subdirectories. Screen printing and file writing can be independently enabled or
    disabled.

    NOTE: This class is not a singleton, so be careful when using it in concurrent code.

    Attributes:
    -----------
    outDir (str): The parent output directory to write the log file to
    verbose (bool): Shows if logger prints to console

    Methods:
    --------
    print_centered: Writes heading to screen and log file
    print_ljust: Writes formatted text to screen and log file
    print_dict_to_file: Saves settings dictionary to log file
    log_solution: Processes inference results and write to files
    log_transient_vector: Writes transient data to files
    print_arrays_to_file: Writes numpy array to file
    """

    _fileType = "txt"
    _logFileName = "output.log"
    _inferenceOpts = ["drift", "diffusion", "all"]

    _checkDictLogSolution = {
        "file_names": (list, None, False),
        "function_spaces": (list, None, False),
        "simulation_times": (np.ndarray, None, True),
        "solution_data": (list, None, False),
    }

    # -----------------------------------------------------------------------------------------------
    def __init__(
        self,
        verbose: Optional[bool] = True,
        outDir: Optional[str] = None,
        printInterval: Optional[str] = 1,
    ) -> None:
        """Constructor

        Sets up output directory, verbosity and print interval (only relevant for transient models).

        Args:
            verbose (bool, optional): Determines if output is printed to screen. Defaults to True
            outDir (str, optional): Parent output directory for files. If None, no output is generated.
                                    Defaults to None
            printInterval (int, optional): Interval for file output, only for transient problems.
                                           This is a default setting that may be overwritten in the
                                           respective methods. Defaults to 1 (prints every time step)
        """

        self.outDir = outDir
        self.verbose = verbose
        self._printInterval = printInterval

        self._logFile = None
        if self.outDir is not None:
            os.makedirs(self.outDir, exist_ok=True)
            self._logFile = os.path.join(self.outDir, self._logFileName)

    # -----------------------------------------------------------------------------------------------
    def print_centered(self, message: Any, filler: str) -> None:
        """Prints heading to file

        This routine prints a centered message to the screen and log file, with the provided filler
        forming a header bar.

        Args:
            message (Any): Message to print
            filler (str): Line filler

        Raises:
            TypeError: Checks that filler is valid character
        """

        if not isinstance(filler, str):
            raise TypeError("Print filler needs to be string character.")

        try:
            width = min(os.get_terminal_size().columns, 60)
        except OSError:
            width = 60
        if bool(message):
            message = " " + message + " "

        if self._verbose:
            print(message.center(width, filler), flush=True)
        if self._logFile is not None:
            with open(self._logFile, "a") as outFile:
                print(message.center(width, filler), flush=True, file=outFile)

    # -----------------------------------------------------------------------------------------------
    def print_ljust(
        self, message: Any, width: Optional[int] = None, end: Optional[str] = "\n"
    ) -> None:
        """Prints structured output to console and log files

        Args:
            message (Any): Message to print
            width (Optional[int], optional): Width of the string block. If None, this is
                                             the length of the message. Defaults to None
            end (Optional[str], optional): End character of the string block. Defaults to "\n".

        Raises:
            TypeError: Checks that width is valid
            TypeError: Checks that end-of-line character is valid
        """

        if width is None:
            width = len(str(message)) + 1
        if not (isinstance(width, int) and width > 0):
            raise TypeError("Message width needs to be positive integer.")
        if not (isinstance(end, str)):
            raise TypeError("End-of-message character needs to be valid string.")

        if self._verbose:
            print(str(message).ljust(width), end=end, flush=True)
        if self._logFile is not None:
            with open(self._logFile, "a") as outFile:
                print(str(message).ljust(width), end=end, flush=True, file=outFile)

    # -----------------------------------------------------------------------------------------------
    def print_dict_to_file(self, heading: str, infoDict: dict[str, Any]) -> None:
        """Prints a settings dict to the log file

        Args:
            heading (str): Name of the dictionary, used as header
            infoDict (dict[str, Any]): Dictionary to write to file

        Raises:
            TypeError: Checks type of the heading
            TypeError: Checks type of the dictionary
        """

        leftSpace = 30
        rightSpace = 20

        if not isinstance(heading, str):
            raise TypeError("File heading needs to be specified as string.")
        if not isinstance(infoDict, dict):
            raise TypeError("Object to print needs to be dictionary with string keys.")

        if self._logFile is not None:
            with open(self._logFile, "a") as outFile:
                tableHead = f"{'Setting':<{leftSpace}} | {'Value':<{rightSpace}}"
                outFile.write("\n --- " + heading + " --- " + "\n")
                outFile.write(tableHead + "\n")
                outFile.write("-" * len(tableHead) + "\n")
                for key, value in infoDict.items():
                    if callable(value):
                        value = inspect.getsource(value)
                        value = (value.split(":", 1)[1]).split(",", 1)[0].strip()
                    if hasattr(value, "__iter__"):
                        if any(callable(entry) for entry in value):
                            value = inspect.getsource(value[0])
                            value = (value.split("[", 1)[1]).split("]", 1)[0].strip()
                    outFile.write(
                        f"{str(key):<{leftSpace}} | {str(value):<{rightSpace}}\n"
                    )
                outFile.write("\n")

    # -----------------------------------------------------------------------------------------------
    def log_solution(
        self,
        paramsToInfer: str,
        isStationary: bool,
        dataStructs: dict[str, Any],
        printInterval: Optional[int] = None,
        subDir: Optional[str] = None,
    ) -> Tuple[list[np.ndarray]]:
        """Logs the results of an inference procedure

        The standard results of an inference procedure are an array each for the parameter mean,
        the variance and the associated forward solution. The latter may also be time-dependent.
        These arrays are written to files with the given names and into the previously set
        subdirectory.
        Before printing the results to files, this routine processes the input by forming numpy
        arrays of the solution values over the associated spacial domain. Thus, the returned
        structures can directly be plotted.

        Args:
            paramsToInfer (str): Inference parameter functions, options are 'drift', 'diffusion' and 'all'
            isStationary (bool): Determines if the problem is stationary
            dataStructs (dict[str, Any]): Data for processing and printing
                -> file_names (list[str]): List of three file names
                -> function_spaces (list[fe.FunctionSpace]): List of function spaces for forward,
                                                             parameter and adjoint variables
                -> simulation_times (np.ndarray): Simulation time array, only necessary for transient
                                                  models
                -> solution_data (list[fe.GenericVector, hl.TimeDependentVector]): Data to process
                                                                                   and print
            printInterval (Optional[int], optional): Print interval to overwrite the default set
                                                     in the constructor. Defaults to None.
            subDir (str, optional): Subdirectory to write array files to. If None, the parent
                                    directory is used. Defaults to None

        Raises:
            TypeError: Checks type of function spaces
            TypeError: Checks type of file name list
            TypeError: Checks type of solution arrays

        Returns:
            Tuple[list[np.ndarray]]: Mean, variance and forward solution as numpy arrays, together
                                     with associated domain point values, ready for printing as
                                     x-y-pairs
        """

        if paramsToInfer not in self._inferenceOpts:
            raise ValueError("Inference options are: " + ", ".join(self._inferenceOpts))
        if paramsToInfer == "all":
            numParams = 2
        else:
            numParams = 1

        if not isStationary:
            if "simulation_times" not in dataStructs.keys():
                raise KeyError(
                    "Need to provide simulation times for logging of transient solution."
                )
            simTimes = dataStructs["simulation_times"]

        utils.check_settings_dict(dataStructs, self._checkDictLogSolution)
        funcSpaces = dataStructs["function_spaces"]
        fileNameList = dataStructs["file_names"]
        solArrayList = dataStructs["solution_data"]

        if not (
            isinstance(funcSpaces, list)
            and len(funcSpaces) == 3
            and all(isinstance(space, fe.FunctionSpace) for space in funcSpaces)
        ):
            raise TypeError(
                "Funcspaces need to be list of three FEniCS function spaces."
            )
        if not (
            isinstance(fileNameList, list)
            and len(funcSpaces) == 3
            and all(isinstance(fileName, str) for fileName in fileNameList)
        ):
            raise TypeError("file names need to be list of three strings.")
        if not (
            isinstance(solArrayList, list)
            and len(solArrayList) == 3
            and isinstance(solArrayList[0], fe.GenericVector)
            and isinstance(solArrayList[1], fe.GenericVector)
            and isinstance(solArrayList[2], (fe.GenericVector, hl.TimeDependentVector))
        ):
            raise TypeError(
                "Solution arrays need to be list of [Vector, Vector, Vector/TDV]."
            )

        gridPointsState = funcSpaces[hl.STATE].tabulate_dof_coordinates().flatten()
        gridPointsParam = utils.reshape_to_np_format(
            funcSpaces[hl.PARAMETER].tabulate_dof_coordinates(), numParams
        )[..., 0]

        meanFileName, varFileName, forwardFileName = fileNameList
        meanSol, varSol, forwardSol = solArrayList
        meanHeading, varHeading, forwardHeading = self._get_default_headers(
            paramsToInfer
        )

        meanData = [
            gridPointsParam,
            utils.reshape_to_np_format(meanSol.get_local(), numParams),
        ]
        varianceData = [
            gridPointsParam,
            utils.reshape_to_np_format(varSol.get_local(), numParams),
        ]
        self.print_arrays_to_file(meanFileName, meanHeading, meanData, subDir)
        self.print_arrays_to_file(varFileName, varHeading, varianceData, subDir)

        if isStationary:
            forwardData = [gridPointsState, forwardSol.get_local()]
            self.print_arrays_to_file(
                forwardFileName, forwardHeading, forwardData, subDir
            )
        else:
            forwardData = [gridPointsState, simTimes, utils.tdv_to_nparray(forwardSol)]
            self.log_transient_vector(
                forwardFileName, forwardHeading, forwardData, printInterval, subDir
            )

        return meanData, varianceData, forwardData

    # -----------------------------------------------------------------------------------------------
    def log_transient_vector(
        self,
        fileName: str,
        heading: list[str],
        transientData: list[int, float, np.ndarray],
        printInterval: Optional[int] = None,
        subDir: Optional[str] = None,
    ) -> None:
        """Writes transient vector to several output files

        Transient vectors are written to several files, one for each time point deduced from the
        given time point array and the print interval. The name of each file is assembled from the
        provided name stem and a 't=[time]' appendix.

        Args:
            fileName (str): File name stem
            heading (list[str]): File header
            transientData (list[int, float, np.ndarray]): list of grid points, time points and data
            printInterval (Optional[int], optional): Print interval. If None, the default set in the
                                                     constructor is used. Defaults to None
            subDir (str, optional): Subdirectory to write array files to. If None, the parent
                                    directory is used. Defaults to None

        Raises:
            TypeError: Checks type of all input arrays
            ValueError: Checks that data vector is defined over associated time points
        """

        if printInterval is None:
            piToUse = self._printInterval
        else:
            piToUse = printInterval

        gridPoints, timePoints, solVec = transientData
        timePoints = utils.process_input_data(timePoints)
        gridPoints = utils.process_input_data(gridPoints)
        solVec = utils.process_input_data(solVec)

        if timePoints.size == 1 and solVec.ndim == 1:
            solVec = np.expand_dims(solVec, axis=0)
        if gridPoints.size == 1 and solVec.ndim == 1:
            solVec = np.expand_dims(solVec, axis=1)
        if not timePoints.shape[0] == solVec.shape[0]:
            raise ValueError(
                "Time points and solution need to have same length"
                " along first dimension"
            )

        for i, t in enumerate(timePoints):
            if (i % piToUse == 0) or (t == timePoints[-1]):
                currForwardData = [gridPoints, solVec[i, :]]
                currforwardFileName = fileName + "_t=" + str(t)
                self.print_arrays_to_file(
                    currforwardFileName, heading, currForwardData, subDir
                )

    # -----------------------------------------------------------------------------------------------
    def print_arrays_to_file(
        self,
        fileName: str,
        headingList: list[str],
        arrayBlock: list[np.ndarray],
        subDir: Optional[str] = None,
    ) -> None:
        """Prints numpy array block to file

        It is assumed that the arrays are ordered column-wise, so that the standard 'numpy.savetxt'
        command can be used.

        Args:
            fileName (str): File name
            headingList (list[str]): Headers for the different columns
            arrayBlock (list[np.ndarray]): List of arrays, each entry defining one column
            subDir (str, optional): Subdirectory to write array files to. If None, the parent
                                    directory is used. Defaults to None

        Raises:
            TypeError: Checks type of the file name
            TypeError: Checks types of the heading list
            TypeError: Checks types of the array list
            TypeError: Checks type of the subdirectory
        """

        colSize = 14
        digits = 4

        if not isinstance(fileName, (str, type(None))):
            raise TypeError("Output file name needs to be provided as string.")
        if not (
            isinstance(headingList, list)
            and all(isinstance(headingEntry, str) for headingEntry in headingList)
        ):
            raise TypeError("Header list needs to be provided as list of strings.")
        if not (
            isinstance(arrayBlock, list)
            and all(isinstance(array, (float, np.ndarray)) for array in arrayBlock)
        ):
            raise TypeError("Array block needs to be provided as list of numpy arrays.")
        if not isinstance(subDir, (str, type(None))):
            raise TypeError("Subdirectory name needs to be provided as string.")

        if subDir is None:
            concatDir = self.outDir
        elif self.outDir is not None:
            concatDir = os.path.join(self.outDir, subDir)
            os.makedirs(concatDir, exist_ok=True)
        else:
            concatDir = None

        if fileName is not None and concatDir is not None:
            file = os.path.join(concatDir, fileName) + "." + self._fileType
            heading = ""
            for headingPart in headingList:
                heading += f"{headingPart:<{colSize + 1}}"
            heading += "\n"

            with open(file, "w") as outFile:
                outFile.write(heading)
                np.savetxt(
                    outFile, np.column_stack(arrayBlock), fmt=f"%+-{colSize}.{digits}e"
                )

    # -----------------------------------------------------------------------------------------------
    def _get_default_headers(self, paramsToInfer: str) -> None:
        """Returns standard headings of array files for inference data"""

        if paramsToInfer == "drift":
            meanHeading = ["x", "mean(f(x))"]
            varHeading = ["x", "var(f(x))"]
        elif paramsToInfer == "diffusion":
            meanHeading = ["x", "mean(g^2(x))"]
            varHeading = ["x", "var(g^2(x))"]
        elif paramsToInfer == "all":
            meanHeading = ["x", "mean(f(x))", "mean(g^2(x))"]
            varHeading = ["x", "var(f(x))", "var(g^2(x))"]
        else:
            raise ValueError(
                "Unknown option for paramsToInfer. "
                "Valid options are 'drift', 'diffusion', 'all'"
            )

        forwardHeading = ["x", "forward(x)"]
        return meanHeading, varHeading, forwardHeading

    # -----------------------------------------------------------------------------------------------
    @property
    def outDir(self) -> str:
        return self._outDir

    @property
    def verbose(self) -> bool:
        return self._verbose

    # -----------------------------------------------------------------------------------------------
    @outDir.setter
    def outDir(self, outDir: str) -> None:
        if not isinstance(outDir, (str, type(None))):
            raise TypeError("Input does not have valid data type 'str' or 'None'.")
        self._outDir = outDir

    @verbose.setter
    def verbose(self, verbose: bool) -> None:
        if not isinstance(verbose, bool):
            raise TypeError("Input does not have valid data type 'bool'.")
        self._verbose = verbose
