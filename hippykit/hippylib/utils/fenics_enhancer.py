from numbers import Number
from typing import Callable, Iterable, Union

import dolfin as dl
import numpy as np
from multimethod import multimethod
from sympy import lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr

#---------------------------------------------------------------------------------------------------
def count_sub_spaces(func_space: dl.FunctionSpace) -> int:
    """
    Count the number of sub spaces in a given FunctionSpace.

    Parameters:
        func_space (dl.FunctionSpace): The FunctionSpace to count sub spaces in.

    Returns:
        int: The number of sub spaces in the given FunctionSpace.
    """
    num_components = func_space.num_sub_spaces()
    if num_components == 0:
        return 1
    else:
        num_sub_spaces = 0
        for component in range(num_components):
            sub_space = func_space.extract_sub_space([component])
            num_sub_spaces += count_sub_spaces(sub_space)
        return num_sub_spaces

#---------------------------------------------------------------------------------------------------  
def get_space_depth(func_space: dl.FunctionSpace) -> int:
    """
    Calculate the depth of a given function space.

    Args:
        func_space (dl.FunctionSpace): The function space to calculate the depth for.

    Returns:
        int: The depth of the function space.

    """
    num_components = func_space.num_sub_spaces()
    if num_components == 0:
        return 0
    else:
        depth = num_components * [1,]
        for component in range(num_components):
            sub_space = func_space.extract_sub_space([component])
            depth[component] += get_space_depth(sub_space)
        return max(depth)
    
#---------------------------------------------------------------------------------------------------
def convert_np_array_np2fe(input_arrays: Union[np.ndarray, Iterable[np.ndarray]],
                           function_space: dl.FunctionSpace) -> np.ndarray:
    """
    Convert numpy array or iterable of numpy arrays to a Fenics compatible numpy array.

    Args:
        input_arrays (Union[np.ndarray, Iterable[np.ndarray]]): The input numpy array or
                                                                iterable of numpy arrays. 
        function_space (dl.FunctionSpace): The Fenics function space.

    Returns:
        np.ndarray: The converted Fenics compatible numpy array.

    Raises:
        ValueError: If the function space has depth greater than 1.
        ValueError: If the number of input arrays does not match the number of
                    function space components.
        ValueError: If the dimension of an input array does not match the function space dimension.
        ValueError: If the length of an input array does not match the required number of values.

    """
    if isinstance(input_arrays, np.ndarray):
        input_arrays = [input_arrays,]
    space_depth = get_space_depth(function_space)
    num_components = count_sub_spaces(function_space)
    num_dimensions = function_space.mesh().geometry().dim()

    if space_depth > 1:
        raise ValueError(f"Function space has depth {space_depth}, "
                         f"but this method cannot treat nested function spaces")
    if not (num_components == len(input_arrays)):
        raise ValueError(f"Number of input arrays ({len(input_arrays)}) "
                         f"and function space components ({num_components}) do not match.")
    
    fe_format_array = np.zeros((function_space.dim(), num_dimensions))
    fe_format_array = np.squeeze(fe_format_array)
    for i, array in enumerate(input_arrays):
        array = array.squeeze()
        array_dim = array.ndim if array.ndim > 0 else 1
        if not array_dim == num_dimensions:
            raise ValueError(f"Array for component {i} has dimension {array_dim}, "
                             f"But the function space dimension is {num_dimensions}.")
        if not array.shape[0] == function_space.dim() / num_components:
            raise ValueError(f"Input array has length {array.shape[0]}, "
                            f"but the function space requires "
                            f"{function_space.dim() / num_components} values.")
        fe_format_array[i::num_components] = array

    return fe_format_array

#---------------------------------------------------------------------------------------------------
def convert_np_array_fe2np(input_array: np.ndarray,
                           function_space: dl.FunctionSpace) -> np.ndarray:
    """
    Convert a NumPy array to a NumPy array with a specific format.

    Parameters:
        input_array (np.ndarray): The input NumPy array.
        function_space (dl.FunctionSpace): The function space.

    Returns:
        np.ndarray: The converted NumPy array.

    Raises:
        ValueError: If the function space has depth greater than 1.
        ValueError: If the input array shape does not match the function space dimension.
        ValueError: If the input array length does not match the function space requirement.
    """
    space_depth = get_space_depth(function_space)
    num_components = count_sub_spaces(function_space)
    num_dimensions = function_space.mesh().geometry().dim()
    input_array = input_array.squeeze()
    array_dim = input_array.ndim if input_array.ndim > 0 else 1

    if space_depth > 1:
        raise ValueError(f"Function space has depth {space_depth}, "
                         f"but this method cannot treat nested function spaces")
    if array_dim != num_dimensions:
        raise ValueError(f"Input array has shape {input_array.shape}, "
                         f"but the function space dimension is {num_dimensions}.")
    if not input_array.shape[0] == function_space.dim():
        raise ValueError(f"Input array has length {input_array.shape[0]}, "
                         f"but the function space requires {function_space.dim()} values.")
    
    np_format_array = []
    for i in range(num_components):
        component_array = input_array[i::num_components]
        np_format_array.append(component_array)

    return np_format_array

#---------------------------------------------------------------------------------------------------
def create_fe_function_from_np_array(input_array: np.ndarray,
                                     function_space: dl.FunctionSpace) -> dl.Function:
    """
    Create a Fenics Function object from a NumPy array.

    Args:
        input_array (np.ndarray): The input NumPy array.
        function_space (dl.FunctionSpace): The function space for the Fenics Function.

    Returns:
        dl.Function: The created Fenics Function object.
    """
    fe_function = dl.Function(function_space)
    array_template = np.squeeze(fe_function.vector().get_local())
    input_array = np.squeeze(input_array)
    if not input_array.shape == array_template.shape:
        raise ValueError(f"Input array has shape {input_array.shape}, "
                         f"but the function space requires {array_template.shape}.")
    fe_function.vector().set_local(input_array)
    return fe_function

#---------------------------------------------------------------------------------------------------
@multimethod
def convert_to_np_callable(input_string: str, dim: int) -> Callable:
    """
    Generate a numpy callable function from a given input string and dimension.

    Parameters:
        input_string (str): The input string representing the mathematical expression.
        dim (int): The dimension of the input string. Can only take values 1, 2, or 3.

    Returns:
        numpy_callable (Callable): The numpy callable function generated from the input string.

    Raises:
        ValueError: If the dimension is not 1, 2, or 3.
        Exception: If the input string cannot be converted to a numpy callable.
    """
    if dim not in (1, 2, 3):
        raise ValueError("Dimension can only take values 1, 2, or 3.")

    try:
        expr_symbols = symbols('x y z')[:dim]
        sympy_callable = parse_expr(input_string)
        numpy_callable = lambdify(expr_symbols, sympy_callable, 'numpy')
    except Exception as e:
        raise Exception("Could not convert input string to numpy callable.") from e

    return numpy_callable

#---------------------------------------------------------------------------------------------------
@multimethod
def convert_to_np_callable(input_number: Number, dim: int) -> Callable:
    """
    Generates a callable function based on the input number and dimension.

    Parameters:
        input_number (Number): The number to be multiplied with the output array.
        dim (int): The dimension of the output array. Must be 1, 2, or 3.

    Returns:
        Callable: A callable function that takes in a variable number of coordinate arrays
                  and returns an array of the same size as the first coordinate array,
                  with each element being equal to the input number.
    """

    if dim not in (1, 2, 3):
        raise ValueError("Dimension can only take values 1, 2, or 3.")
    
    def np_callable(*coordinates: Iterable[np.ndarray]):
        if not len(coordinates) == dim:
            raise ValueError(f"Function has dimension {dim}, but {len(coordinates)} "
                             f"coordinates were given.")
        coordinates = list(coordinates)
        for i, coord in enumerate(coordinates):
            coordinates[i] = np.squeeze(coord)
            if not coordinates[i].size == coordinates[0].size:
                raise ValueError(f"Number of values for coordinate 0 ({coordinates[0].size}) "
                                 f"and coordinate {i} ({coordinates[i]}) do not match.")
        return input_number * np.ones(coordinates[0].size)
    
    return np_callable

#---------------------------------------------------------------------------------------------------
@multimethod
def convert_to_np_callable(inputs: Iterable[Union[str, Number]], dim: int) -> Iterable[Callable]:
    """
    A function that takes in an iterable of inputs and an integer dimension and returns
    an iterable of callable objects. 

    Parameters:
        - inputs (Iterable[Union[str, Number]]): An iterable containing elements that can be
                                                 either strings or numbers.
        - dim (int): An integer representing the dimension of the callable objects.

    Returns:
        - numpy_callables (Iterable[Callable]): An iterable of callable objects that are converted
                                                from the inputs.
    """
    numpy_callables = []
    for entry in inputs:
        np_callable = convert_to_np_callable(entry, dim)
        numpy_callables.append(np_callable)

    return numpy_callables

#---------------------------------------------------------------------------------------------------
@multimethod
def convert_to_np_array(input_val: Union[str, Number],
                        function_space: dl.FunctionSpace) -> np.ndarray:
    """
    Convert the input value to a NumPy array.

    Parameters:
        input_val (Union[str, Number]): The input value to be converted.
        function_space (dl.FunctionSpace): The function space to which the input value belongs.

    Returns:
        np.ndarray: The converted NumPy array.

    Raises:
        ValueError: If the function space has more than two subspaces.

    Notes:
        - This method only supports function spaces with at most two subspaces.
        - The input value is converted using the function convert_to_np_callable.
        - The function space's degree of freedom (dof) coordinates are tabulated
          and split into separate arrays.
        - The converted NumPy array is evaluated at the dof coordinates.

    """
    num_sub_spaces = count_sub_spaces(function_space)
    if function_space.num_sub_spaces() > 1:
        raise ValueError(f"This method only supports function spaces with at most two subspaces, "
                         f"but the given space has {num_sub_spaces} components.")
    dim = function_space.mesh().geometry().dim()
    numpy_callable = convert_to_np_callable(input_val, dim)
    np_dof_coordinates = np.hsplit(function_space.tabulate_dof_coordinates(), dim)
    numpy_func_eval = numpy_callable(*np_dof_coordinates)
    numpy_func_eval = np.squeeze(numpy_func_eval)

    return numpy_func_eval

#---------------------------------------------------------------------------------------------------
@multimethod
def convert_to_np_array(inputs: Iterable[Union[str, Number]],
                        function_space: dl.FunctionSpace) -> Iterable[np.ndarray]:
    """
    Generates a numpy array representation for each entry in the given inputs using
    the specified function space.

    Args:
        inputs (Iterable[Union[str, Number]]): A collection of inputs to be converted to
                                               numpy arrays.
        function_space (dl.FunctionSpace): The function space used to determine the
                                           number of subspaces.

    Returns:
        Iterable[np.ndarray]: A collection of numpy arrays representing the converted inputs.

    Raises:
        ValueError: If the function space has less than two subspaces.

    Example:
        >>> inputs = ['1', 2, '3']
        >>> function_space = dl.FunctionSpace()
        >>> convert_to_np_array(inputs, function_space)
        [array([1]), array([2]), array([3])]
    """
    num_sub_spaces = count_sub_spaces(function_space)
    if num_sub_spaces < 2:
        raise ValueError(f"This method only supports function spaces with at least two subspaces. "
                         f"but the given space has only {num_sub_spaces} components.")
    numpy_func_evals = []
    for i, entry in enumerate(inputs):
        sub_space = function_space.extract_sub_space([i]).collapse()
        np_eval = convert_to_np_array(entry, sub_space)
        np_eval = np.squeeze(np_eval)
        numpy_func_evals.append(np_eval)
    
    fe_format_array = convert_np_array_np2fe(numpy_func_evals, function_space)
    return fe_format_array

#---------------------------------------------------------------------------------------------------
def convert_to_fe_function(input_val: Union[str, Number, Iterable[Union[str, Number]]],
                           function_space: dl.FunctionSpace) -> np.ndarray:
    """
    Convert the input value to a finite element (FE) function.

    Parameters:
        input_val (Union[str, Number, Iterable[Union[str, Number]]]): The input value
                                                                      to be converted.
        function_space (dl.FunctionSpace): The function space to create the FE function in.

    Returns:
        np.ndarray: The FE function created from the input value.
    """
    np_array = convert_to_np_array(input_val, function_space)
    fe_function = create_fe_function_from_np_array(np_array, function_space)

    return fe_function


