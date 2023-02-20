"""General utilities module

This module contains a collection of supplementary functions. These functions mainly deal with the
conversion between FEniCS/hIPPYlib and python/numpy data types. Further capabilities include a 
checking routine for settings dictionaries and the conversion of numerical data into a uniform
format.

Functions:
----------
nparray_to_fevec: Converts numpy array into FEniCS vector
nparray_to_fefunc: Converts numpy array into FEniCS function
nparray_to_tdv: Converts 2D numpy array into hIPPYlib time-dependent vector
tdv_to_nparray: Reverses conversion
pyfunc_to_fevec: Converts python callable into FEniCS vector
pyfunc_to_fefunc: Converts python callable into FEniCS function
reshape_fe_array: Reshapes FEniCS vector array for multiple variables
check_settings_dict: Checks settings dictionary against given prototype
process_input_data: Processes numeric input data into uniform numpy format
process_output_data: Processes numeric output data into uniform format
"""

#====================================== Preliminary Commands =======================================
import warnings
import numpy as np
from typing import Any, Callable, Optional, Union

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe
    import hippylib as hl


#======================================== Utility Functions ========================================

#---------------------------------------------------------------------------------------------------
def nparray_to_fevec(array: np.ndarray) -> fe.GenericVector:
    """Converts a numpy array into a FEniCS vector

    The array values are assigned to the local value of the created vector.

    Args:
        array (np.ndarray): 1D Input array

    Raises:
        TypeError: Checks type of input array

    Returns:
        fe.GenericVector: FEniCS vector
    """

    if not (isinstance(array, np.ndarray)):
        raise TypeError("Need to provide a 1D numpy array.")

    array = array.flatten()
    feVec = fe.Vector()
    feVec.init(array.size)
    feVec.set_local(array)

    return feVec

#---------------------------------------------------------------------------------------------------
def nparray_to_fefunc(array: np.ndarray, 
                      funcSpace: Union[fe.FunctionSpace, fe.VectorFunctionSpace])\
                      -> fe.Function:
    """Converts a numpy array into a FEniCS function

    This procedure requires the conversion to a FEniCS vector (see above) and a subsequent call to
    the hippylib vector-to-function routine.

    Args:
        array (np.ndarray): Input array
        funcSpace (fe.FunctionSpace, fe.VectorFunctionSpace): Function space to create vector on

    Returns:
        fe.Function: FEniCS function
    """

    feVec = nparray_to_fevec(array)
    feFunc = hl.vector2Function(feVec, funcSpace)

    return feFunc

#---------------------------------------------------------------------------------------------------
def nparray_to_tdv(timePoints: np.ndarray, tdArray: np.ndarray) -> hl.TimeDependentVector:
    """Converts a numpy array into a hIPPYlib time-dependent vector

    The provided data array needs to be two-dimensional, whereas the first dimension corresponds
    to the different time points specified by the time point array.

    Args:
        timePoints (np.ndarray): Time point array
        tdArray (np.ndarray): Data for all time points

    Raises:
        TypeError: Checks input array types
        ValueError: Checks shape of time point array
        ValueError: Checks shape of data array

    Returns:
        hl.TimeDependentVector: Time-dependent vector
    """

    if not all(isinstance(array, np.ndarray) for array in [timePoints, tdArray]):
        raise TypeError("Function inputs need to be numpy arrays.")
    if not ((timePoints.shape[0] == tdArray.shape[0]) 
         or (timePoints.ndim == tdArray.ndim == 1)):
        raise ValueError("Time points and value array do not match in first dimension.")
    if not len(tdArray.shape) in (1, 2):
        raise ValueError("This function only supports 1D space variables")

    if tdArray.ndim == 1:
        tdArray = tdArray[np.newaxis, :]
    tdVec = hl.TimeDependentVector(timePoints)
    vecSize = tdArray.shape[1]

    for i, _ in enumerate(timePoints):
        tdVec.data[i].init(vecSize)
        tdVec.data[i].set_local(tdArray[i, :])

    return tdVec

#---------------------------------------------------------------------------------------------------
def tdv_to_nparray(tdVec: hl.TimeDependentVector) -> np.ndarray:
    """Converts a hIPPYlib time-dependent vector into a numpy array

    Args:
        tdVec (hl.TimeDependentVector): Input vector

    Raises:
        TypeError: Checks input type
        AttributeError: Checks if TDV has been initialized

    Returns:
        np.ndarray: Converted array
    """

    if not isinstance(tdVec, hl.TimeDependentVector):
        raise TypeError("Function inputs need to be hIPPYlib TimeDependentVector.")
    if any(dataVec.local_size() == 0 for dataVec in tdVec.data):
        raise AttributeError("TimeDependentVector has not been initialized properly.")

    timePoints = tdVec.times
    vecSize = tdVec.data[0].local_size()
    tdArray = np.ndarray((timePoints.size, vecSize))

    for i, _ in enumerate(timePoints):
        tdArray[i, :] = tdVec.data[i].get_local()

    return tdArray

#---------------------------------------------------------------------------------------------------
def pyfunc_to_fevec(pyFuncHandle: Union[Callable, list[Callable]], 
                    funcSpace: Union[fe.FunctionSpace, fe.VectorFunctionSpace, 
                    fe.TensorFunctionSpace]) -> fe.GenericVector:
    """Converts a python callable or list of callables into a FEniCS vector

    The callable(s) need to be vectorized, such that it can be evaluated over the degrees of freedom
    of the provided function space. For vector and tensor function spaces, a list of callables needs
    to be given, whose number of entries matches the number of subspaces. When using this routine,
    be aware of the ordering and reduced number of subspaces in symmetrical tensor function spaces.

    Args:
        pyFuncHandle (Union[Callable, list[Callable]]): Callable or list of callables to convert
        funcSpace (Union[fe.FunctionSpace, fe.VectorFunctionSpace, fe.TensorFunctionSpace]):
            Space to create FEniCS vector on

    Raises:
        TypeError: Checks that space and number of callables match

    Returns:
        fe.GenericVector: FEniCS vector over provided function space
    """

    numSubSpaces = funcSpace.num_sub_spaces()

    if numSubSpaces in [0, 1]:
        if not callable(pyFuncHandle):
            raise TypeError("Need callable python object for function space" 
                            " of dimension zero or one.")
        feVec = _pyfunc_to_fevec(pyFuncHandle, funcSpace)
    else:
        if not isinstance(pyFuncHandle, list) and len(list) == numSubSpaces:
            raise TypeError("Need list of callables matching the number of FE subspaces.")

        subFuncs = []
        for i in range(numSubSpaces):
            pySubFunc = pyFuncHandle[i]
            subSpace = funcSpace.sub(i).collapse()
            feSubVec = _pyfunc_to_fevec(pySubFunc, subSpace)
            subFuncs.append(hl.vector2Function(feSubVec, subSpace))
        feFunc = fe.Function(funcSpace)
        fe.assign(feFunc, subFuncs)
        feVec = feFunc.vector()

    assert isinstance(feVec, fe.GenericVector), "Output is not a proper FEniCS vector."
    return feVec

#---------------------------------------------------------------------------------------------------
def pyfunc_to_fefunc(pyFuncHandle: Union[Callable, list[Callable]],
                     funcSpace: Union[fe.FunctionSpace, fe.VectorFunctionSpace, fe.TensorFunctionSpace])\
                     -> fe.Function:
    """Converts a python callable or list of callables into a FEniCS function

    Args:
        pyFuncHandle (Union[Callable, list[Callable]]): Callable or list of callables to convert
        funcSpace (Union[fe.FunctionSpace, fe.VectorFunctionSpace, fe.TensorFunctionSpace]):
            Space to create FEniCS vector on

    Returns:
        fe.Function: FEniCS function from python function(s)
    """

    feVec = pyfunc_to_fevec(pyFuncHandle, funcSpace)
    feFunc = hl.vector2Function(feVec, funcSpace)

    assert isinstance(feFunc, fe.Function), "Output is not a proper FEniCS function."
    return feFunc    

#---------------------------------------------------------------------------------------------------
def reshape_fe_array(flatArray: np.ndarray, numComponents: int) -> np.ndarray:
    """Reshapes array from FEniCS vector into form suitable for multiple variables

    FEniCS vectors, even if deduced from multi-component functions, are always one-dimensional. The
    values of the different components at each dof are simply stacked into a 1D vector. This routine
    converts such flattened arrays into two-dimensional arrays, where the new dimension accounts for
    the number of components. This makes such structures much easier to process further.

    Args:
        flatArray (np.ndarray): Flat array from FEniCS vector
        numComponents (int): Number of components

    Raises:
        TypeError: Checks input array type
        TypeError: Checks number of components

    Returns:
        np.ndarray: Reshaped array
    """

    if not isinstance(flatArray, np.ndarray):
        raise TypeError("Input needs to be numpy array.")
    if not isinstance(numComponents, int) and numComponents > 0:
        raise TypeError("Number of components needs to be positive integer.")

    arrayList = []
    for i in range(numComponents):
        arrayList.append(flatArray[i::numComponents])
    reshapedArray = np.column_stack(arrayList)

    return reshapedArray

#---------------------------------------------------------------------------------------------------
def check_settings_dict(dictToTest: dict[str, Any], checkDict: dict[str, Any]) -> None:
    """Checks settings dictionary agains prototype

    The prototype contains information on the type  and possible bounds of the different values.
    It further determines which arguments are optional. Therefore, the prototype dictionary contains
    the same keys as the actual settings dicts. Its value, however, need to be a tuple of the form
    (type (single or list of possible types), bounds (as list or None), isOptional (bool)).

    Args:
        dictToTest (dict[str, Any]): Settings dictionary to check
        checkDict (dict[str, Any]): Prototype to check against

    Raises:
        KeyError: Checks if all mandatory settings are given
        TypeError: Checks if entries have correct type
        ValueError: Checks if entries are within prescribed bounds
    """

    assert isinstance (dictToTest, dict) and isinstance (checkDict, dict), \
           "Comparison objects need to be dictionaries."

    for checkKey in checkDict.keys():
        if not checkKey in dictToTest.keys():
            if checkDict[checkKey][2]:
                continue
            raise KeyError(f"Necessary key '{checkKey}' not found.")

        if not isinstance(dictToTest[checkKey], checkDict[checkKey][0]):
            raise TypeError(f"Entry for key '{checkKey}' has type {type(dictToTest[checkKey])},"
                            f"but needs to have type {checkDict[checkKey][0]}.")

        if checkDict[checkKey][1] is not None:
            lowerLimit = checkDict[checkKey][1][0]
            upperLimit = checkDict[checkKey][1][1]
            if not (lowerLimit <= dictToTest[checkKey] <= upperLimit):
                raise ValueError(f"Dict entry '{checkKey}' is not within prescribed bounds"
                                 f"[{lowerLimit},{upperLimit}].")

#---------------------------------------------------------------------------------------------------
def process_input_data(inputData: Union[int, float, np.ndarray],
                       enforce1D: Optional[bool]=False)-> np.ndarray:
    """Converts input data into standard format (numpy array)

    Args:
        inputData (Union[int, float, np.ndarray]): [Numeric input
        enforce1D (Optional[bool], optional): Determines if input is supposed to be one-dimensional.
                                              Defaults to False

    Raises:
        TypeError: Checks type of input data
        ValueError: Checks if data is one-dimensional (if required)

    Returns:
        np.ndarray: Data in standardized format
    """

    if not isinstance(inputData, (int, float, np.ndarray)):
        raise TypeError("Domain points need to be provided as int, float, or numpy array.")
    if not isinstance(inputData, np.ndarray):
        inputData = np.array(inputData, ndmin=1)
    if enforce1D and inputData.ndim > 1:
        raise ValueError("Input data has to be one-dimensional")

    return inputData

#---------------------------------------------------------------------------------------------------
def process_output_data(outputData: np.ndarray)-> Union[float, np.ndarray]:
    """Converts output data into most convenient output format

    Args:
        outputData (np.ndarray): Raw output data
    Raises:
        AssertionError: Checks that output has valid type

    Returns:
        Union[float, np.ndarray]: Converted data
    """

    if isinstance(outputData, np.ndarray):
        pointArray = np.squeeze(outputData)
    if hasattr(outputData, 'size'):
        if outputData.size == 1:
            pointArray = float(outputData) 
    elif not isinstance(outputData, (int, float)):
        raise AssertionError("Output needs to be number or numpy object.")
        
    return pointArray

#---------------------------------------------------------------------------------------------------
def _pyfunc_to_fevec(pyFuncHandle: Callable, funcSpace: fe.FunctionSpace) -> fe.GenericVector:
    """Converts python callable into FEniCS vector on scalar function space"""

    if not isinstance(funcSpace, fe.FunctionSpace):
        raise TypeError("Need to provide a proper FEniCS function space.")

    dofCoords = funcSpace.tabulate_dof_coordinates()
    pyArray = pyFuncHandle(dofCoords)
    feVec = fe.Function(funcSpace).vector()
    feVec.set_local(pyArray)

    assert isinstance(feVec, fe.GenericVector), "Output is not a proper FEniCS vector."
    return feVec
