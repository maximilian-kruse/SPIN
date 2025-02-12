"""_summary_."""
from collections.abc import Iterable

import dolfin as dl
import hippylib as hl
import numpy as np
import numpy.typing as npt
import scipy as sp


# --------------------------------------------------------------------------------------------------
def create_dolfin_function(
    string_expression: str | Iterable[str],
    function_space: dl.FunctionSpace,
) -> dl.Function:
    """Compile a dolfin function from string expressions.

    The user can either provide a single string expression for scalar function spaces, or a list
    for vector spaces. The number of expression strings has to match the number of components in a
    vector space.

    !!! info "Expression syntax"
        Expressions to be compiled in dolfin need to adhere to C++ syntax. Dolfin can compile most
        functions from the [`cmath`](https://en.cppreference.com/w/cpp/header/cmath) library.

    Args:
        string_expression (str | Iterable[str]): Expression strings to compile
        function_space (dl.FunctionSpace): FUnctions space of dolfin function to create

    Raises:
        ValueError: Checks that only a single string is supplied for scalar vector spaces
        ValueError: Checks that number of strings matches number of components for vector spaces

    Returns:
        dl.Function: Created dolfin function
    """
    element_degree = function_space.ufl_element().degree()
    num_components = function_space.num_sub_spaces()
    if num_components == 0 and not isinstance(string_expression, str):
        raise ValueError("Only a single string expression is allowed for scalar function spaces.")
    elif not (isinstance(string_expression, Iterable) and num_components != len(string_expression)):  # noqa: RET506
        raise ValueError("Number of expression strings must match number of components in space.")
    parameter_expression = dl.Expression(string_expression, degree=element_degree)
    parameter_function = dl.Function(function_space)
    parameter_function.interpolate(parameter_expression)
    return parameter_function


# --------------------------------------------------------------------------------------------------
def convert_to_numpy(
    vector: dl.Vector | dl.PETScVector,
    function_space: dl.FunctionSpace,
) -> npt.NDArray[np.floating]:
    r"""Convert a dolfin vector to a numpy array.

    This method takes into account the number of components in the given function space. For $D$
    components, the resulting array has shape $D\times N$ entries, where $N$ is the number of dofs
    of the underlying mesh.

    Args:
        vector (dl.Vector | dl.PETScVector): Vector to convert
        function_space (dl.FunctionSpace): Function space vector has been defined on

    Raises:
        ValueError: Checks that the size of the vector matches the function space dimension

    Returns:
        npt.NDArray[np.floating]: Converted numpy array
    """
    if not vector.size() == function_space.dim():
        raise ValueError("Vector size does not match function space dimension.")
    vector = vector.get_local()
    num_components = function_space.num_sub_spaces()

    if num_components <= 1:
        numpy_array = vector
    else:
        components = []
        for i in range(num_components):
            component_dofs = function_space.sub(i).dofmap().dofs()
            components.append(vector[component_dofs])
        numpy_array = np.stack(components, axis=0)
    return numpy_array


# --------------------------------------------------------------------------------------------------
def convert_to_dolfin(
    array: npt.NDArray[np.floating],
    function_space: dl.FunctionSpace,
) -> dl.Function:
    r"""Convert a numpy array to a dolfin function.

    This method is the counterpart to the
    [`convert_to_numpy`][spin.fenics.converter.convert_to_numpy] method. It takes into account the
    number of components in the given function space. For $D$ components, the input array has to
    have shape $D\times N$ entries, where $N$ is the number of dofs of the underlying mesh. Scalar
    function spaces are compatible with one-dimensional arrays as well.

    Args:
        array (npt.NDArray[np.floating]): Numpy array to convert
        function_space (dl.FunctionSpace): Function space to project to

    Raises:
        ValueError: Checks that the size of the array matches the function space dimension

    Returns:
        dl.Function: Created dolfin function
    """
    if not array.size == function_space.dim():
        raise ValueError("Array size does not match function space dimension.")
    dolfin_function = dl.Function(function_space)
    num_components = function_space.num_sub_spaces()
    if num_components <= 1:
        dolfin_function.vector().set_local(array.flatten())
    else:
        for i in range(num_components):
            component_dofs = function_space.sub(i).dofmap().dofs()
            dolfin_function.vector()[component_dofs] = array[i, :]
    dolfin_function.vector().apply("insert")
    return dolfin_function


# --------------------------------------------------------------------------------------------------
def convert_multivector_to_numpy(
    multivector: hl.MultiVector, function_space: dl.FunctionSpace
) -> list[npt.NDArray[np.floating]]:
    """Convert a Hippylib multivector to a list of numpy arrays.

    Strictly speaking, this method is not a Fenics converter, but specific to Hippylib. It converts
    a Hippylib
    [`Multivector`](https://hippylib.readthedocs.io/en/latest/hippylib.algorithms.html?highlight=multivector#module-hippylib.algorithms.multivector)
    (a compiled collection of dolfin vectors) to a list of numpy arrays. Each individual vector has
    to be defined from the same function space, and is converted with the
    [`convert_to_numpy`][spin.fenics.converter.convert_to_numpy] method.

    Args:
        multivector (hl.MultiVector): Multivector to convert
        function_space (dl.FunctionSpace): Function space individual vectors have been defined on

    Raises:
        ValueError: Checks that the size of individual vectors matches the function space dimension.

    Returns:
        Iterable[npt.NDArray[np.floating]]: Converted list of arrays
    """
    if not multivector[0].size() == function_space.dim():
        raise ValueError("Vector size does not match function space dimension.")
    num_vectors = multivector.nvec()
    list_of_arrays = [convert_to_numpy(multivector[i], function_space) for i in range(num_vectors)]
    return list_of_arrays


# --------------------------------------------------------------------------------------------------
def convert_to_multivector(
    list_of_arrays: Iterable[npt.NDArray[np.floating]], function_space: dl.FunctionSpace
) -> hl.MultiVector:
    """_summary_.

    Args:
        list_of_arrays (Iterable[npt.NDArray[np.floating]]): _description_
        function_space (dl.FunctionSpace): _description_

    Returns:
        hl.MultiVector: _description_
    """
    num_vectors = len(list_of_arrays)
    size_giving_vector = convert_to_dolfin(list_of_arrays[0], function_space).vector()
    multivector = hl.MultiVector(size_giving_vector, num_vectors)
    for i in range(num_vectors):
        vector = convert_to_dolfin(list_of_arrays[i], function_space).vector()
        multivector[i].set_local(vector.get_local())
        multivector[i].apply("insert")
    return multivector


# --------------------------------------------------------------------------------------------------
def get_coordinates(
    function_space: dl.FunctionSpace,
) -> npt.NDArray[np.floating]:
    """_summary_.

    Args:
        function_space (dl.FunctionSpace): _description_

    Returns:
        npt.NDArray[np.floating]: _description_
    """
    num_components = function_space.num_sub_spaces()
    coordinates = function_space.tabulate_dof_coordinates()
    if num_components > 1:
        component_dofs = function_space.sub(0).dofmap().dofs()
        coordinates = coordinates[component_dofs]
    return coordinates


# --------------------------------------------------------------------------------------------------
def extract_components(
    vector: dl.Vector | dl.PETScVector,
    components: Iterable[dl.Vector | dl.PETScVector],
    function_space: dl.FunctionSpace,
) -> Iterable[dl.Vector | dl.PETScVector]:
    """_summary_.

    Args:
        vector (dl.Vector | dl.PETScVector): _description_
        components (Iterable[dl.Vector  |  dl.PETScVector]): _description_
        function_space (dl.FunctionSpace): _description_

    Returns:
        Iterable[dl.Vector | dl.PETScVector]: _description_
    """
    for i, component in enumerate(components):
        subspace = function_space.sub(i)
        component_dofs = subspace.dofmap().dofs()
        component.set_local(vector[component_dofs])
        component.apply("insert")
    return components


# --------------------------------------------------------------------------------------------------
def combine_components(
    components: Iterable[dl.Vector, dl.PETScVector],
    vector: dl.Vector | dl.PETScVector,
    function_space: dl.FunctionSpace,
) -> dl.Vector | dl.PETScVector:
    """_summary_.

    Args:
        components (Iterable[dl.Vector, dl.PETScVector]): _description_
        vector (dl.Vector | dl.PETScVector): _description_
        function_space (dl.FunctionSpace): _description_

    Returns:
        dl.Vector | dl.PETScVector: _description_
    """
    for i, component in enumerate(components):
        component_dofs = function_space.sub(i).dofmap().dofs()
        vector[component_dofs] = component.get_local()
    vector.apply("insert")
    return vector


# --------------------------------------------------------------------------------------------------
def convert_matrix_to_scipy(matrix: dl.Matrix | dl.PETScMatrix) -> sp.sparse.coo_array:
    """_summary_.

    Args:
        matrix (dl.Matrix | dl.PETScMatrix): _description_

    Returns:
        sp.sparse.coo_array: _description_
    """
    rows = []
    columns = []
    values = []

    for i in range(matrix.size(0)):
        non_zero_cols, non_zero_values = matrix.getrow(i)
        rows.extend([i] * len(non_zero_cols))
        columns.extend(non_zero_cols)
        values.extend(non_zero_values)

    scipy_matrix = sp.sparse.coo_array(
        (values, (rows, columns)), shape=(matrix.size(0), matrix.size(1))
    )
    return scipy_matrix
