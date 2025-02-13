r"""Conversion routines between Fenics and Numpy/Scipy.

The underlying array representation of Fenics/dolfin vectors and matrices can sometimes be
intransparent. To enforce a generic and consistent interface in SPIN, this module provides.
bi-directional conversion routines between dolfin and Numpy/Scipy data structures. The structure of
arrays not resembling coordinates is determined by the components of the underlying dolfin function
space. For a vector with $K$ components and $N$ degrees of freedom (according to the underlying
mesh), the resulting numpy array has shape $K\times N$. For coordinate arrays, the dimension of the
domain is relevant. For a mesh with $N$ degrees of freedom, discretizing a domain od dimension $D$,
the resulting array has shape $N\times D$.

Functions:
    create_dolfin_function: Compile a dolfin function from string expressions.
    convert_to_numpy: Convert a dolfin vector to a numpy array.
    convert_to_dolfin: Convert a numpy array to a dolfin function.
    convert_multivector_to_numpy: Convert a Hippylib multivector to a list of numpy arrays.
    convert_to_multivector: Convert a list of numpy arrays to a Hippylib multivector.
    get_coordinates: Get the coordinates of the mesh underlying a function space.
    extract_components: Extract components of a vector defined on a vector function space.
    combine_components: Combine a list of component vectors into a vector on a vector function space.
    convert_matrix_to_scipy: Convert a dolfin matrix to a Scipy sparse array.
"""

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
        string_expression (str | Iterable[str]): Expression strings to compile.
        function_space (dl.FunctionSpace): Function space of dolfin function to create.

    Raises:
        ValueError: Checks that only a single string is supplied for scalar vector spaces.
        ValueError: Checks that number of strings matches number of components for vector spaces.

    Returns:
        dl.Function: Created dolfin function.
    """
    element_degree = function_space.ufl_element().degree()
    num_components = function_space.num_sub_spaces()
    if num_components == 0 and not isinstance(string_expression, str):
        raise ValueError("Only a single string expression is allowed for scalar function spaces.")
    elif not (isinstance(string_expression, Iterable) and num_components != len(string_expression)):  # noqa: RET506
        raise ValueError(
            f"Number of expression strings ({len(string_expression)})"
            f" must match number of components in function space space ({num_components})."
        )
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

    This method takes into account the number of components in the given function space. For $K$
    components, the resulting array has shape $K\times N$ entries, where $N$ is the number of dofs
    of the underlying mesh.

    Args:
        vector (dl.Vector | dl.PETScVector): Vector to convert.
        function_space (dl.FunctionSpace): Function space vector has been defined on.

    Raises:
        ValueError: Checks that the size of the vector matches the function space dimension.

    Returns:
        npt.NDArray[np.floating]: Converted numpy array.
    """
    if not vector.size() == function_space.dim():
        raise ValueError(
            f"Vector size ({vector.size()}) does not match "
            f"function space dimension ({function_space.dim()})."
        )
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
    number of components in the given function space. For $K$ components, the input array has to
    have shape $K\times N$ entries, where $N$ is the number of dofs of the underlying mesh. Scalar
    function spaces are compatible with one-dimensional arrays as well.

    Args:
        array (npt.NDArray[np.floating]): Numpy array to convert.
        function_space (dl.FunctionSpace): Function space to project to.

    Raises:
        ValueError: Checks that the size of the array matches the function space dimension.

    Returns:
        dl.Function: Created dolfin function.
    """
    if not array.size == function_space.dim():
        raise ValueError(
            f"Array size ({array.size}) does not match "
            f"function space dimension ({function_space.dim()})."
        )
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
        multivector (hl.MultiVector): Multivector to convert.
        function_space (dl.FunctionSpace): Function space individual vectors have been defined on.

    Raises:
        ValueError: Checks that the size of individual vectors matches the function space dimension.

    Returns:
        Iterable[npt.NDArray[np.floating]]: Converted list of arrays.
    """
    if not multivector[0].size() == function_space.dim():
        raise ValueError(
            f"Vector size ({multivector[0].size()}) does not match "
            f"function space dimension ({function_space.dim()})."
        )
    num_vectors = multivector.nvec()
    list_of_arrays = [convert_to_numpy(multivector[i], function_space) for i in range(num_vectors)]
    return list_of_arrays


# --------------------------------------------------------------------------------------------------
def convert_to_multivector(
    list_of_arrays: Iterable[npt.NDArray[np.floating]], function_space: dl.FunctionSpace
) -> hl.MultiVector:
    """Convert a list of Numpy arrays to a Hippylib multivector.

    Counterpart of the
    [`convert_multivector_to_numpy`][spin.fenics.converter.convert_multivector_to_numpy] method.
    This method converts a list of numpy arrays to a Hippylib
    [`Multivector`](https://hippylib.readthedocs.io/en/latest/hippylib.algorithms.html?highlight=multivector#module-hippylib.algorithms.multivector).
    The size of each individual array has to match the dimension of the supplied function space.
    Conversion of individual arrays is done with the
    [`convert_to_dolfin`][spin.fenics.converter.convert_to_dolfin] method.

    Args:
        list_of_arrays (Iterable[npt.NDArray[np.floating]]): Numpy arrays to convert.
        function_space (dl.FunctionSpace): Function space to define dolfin vectors on.

    Raises:
        ValueError: Checks that the size of individual arrays matches the function space dimension.

    Returns:
        hl.MultiVector: Output multivector.
    """
    for i, array in list_of_arrays:
        if not array.size == function_space.dim():
            raise ValueError(
                f"Size ({array.size}) of array {i} does not match "
                f"function space dimension ({function_space.dim()})."
            )
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
    r"""Get the coordinates of the mesh underlying a function space.

    For vector spaces, the same coordinates are assumed by all components, so that only the
    coordinates of the first component are returned. For a mesh with $N$ vertices, in $D$
    dimensions, the resulting array has shape $N\times D$.

    Args:
        function_space (dl.FunctionSpace): Function space to extract coordinates from.

    Returns:
        npt.NDArray[np.floating]: Numpy array of mesh coordinates.
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
    """Extract components of a vector defined on a vector function space.

    This method extracts the components of a vector defined on a vector function space and returns
    a list of scalar vectors defined on the respective subspaces. The vector function space needs
    to be homogeneous, mixed spaces are not allowed.

    !!! note "Inplace operation"
        This method modifies the components in-place, which have to be provided as input argument.
        No new memory is allocated.

    Args:
        vector (dl.Vector | dl.PETScVector): Vector to split up.
        components (Iterable[dl.Vector  |  dl.PETScVector]): Components to write vector content to.
        function_space (dl.FunctionSpace): Vector function space of the initial vector.

    Raises:
        ValueError: Checks that the number of components matches the number of subspaces.
        ValueError: Checks that the size of the input vector matches the function space dimension.

    Returns:
        Iterable[dl.Vector | dl.PETScVector]: Iterable of component vectors.
    """
    if not function_space.num_sub_spaces() == len(components):
        raise ValueError(
            f"Number of vector components ({len(components)}) does not match "
            f"number of function space components ({function_space.num_sub_spaces()})."
        )
    if not vector.size() == function_space.dim():
        raise ValueError(
            f"Overall vector size ({vector.size()}) does not match "
            f"function space dimension ({function_space.dim()})."
        )

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
    """Combine a list of component vectors into a vector on a vector function space.

    !!! note "Inplace operation"
        This method modifies the vector in-place, which has to be provided as input argument.
        No new memory is allocated.

    Args:
        components (Iterable[dl.Vector, dl.PETScVector]): Components to assemble into vector.
        vector (dl.Vector | dl.PETScVector): Combined vector.
        function_space (dl.FunctionSpace): Vector function space for combined vector.

    Raises:
        ValueError: Checks that the number of components matches the number of subspaces.
        ValueError: Checks that the size of the output vector matches the function space dimension.

    Returns:
        dl.Vector | dl.PETScVector: Combined vector.
    """
    if not function_space.num_sub_spaces() == len(components):
        raise ValueError(
            f"Number of vector components ({len(components)}) does not match "
            f"number of function space components ({function_space.num_sub_spaces()})."
        )
    if not vector.size() == function_space.dim():
        raise ValueError(
            f"Overall vector size ({vector.size()}) does not match "
            f"function space dimension ({function_space.dim()})."
        )

    for i, component in enumerate(components):
        component_dofs = function_space.sub(i).dofmap().dofs()
        vector[component_dofs] = component.get_local()
    vector.apply("insert")
    return vector


# --------------------------------------------------------------------------------------------------
def convert_matrix_to_scipy(matrix: dl.Matrix | dl.PETScMatrix) -> sp.sparse.coo_array:
    """Convert a dolfin matrix to a Scipy sparse array.

    Args:
        matrix (dl.Matrix | dl.PETScMatrix): Dolfin matrix to convert

    Returns:
        sp.sparse.coo_array: Scipy COO array
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
