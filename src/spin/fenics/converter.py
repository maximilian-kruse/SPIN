from collections.abc import Iterable

import dolfin as dl
import hippylib as hl
import numpy as np
import numpy.typing as npt
import scipy as sp


# --------------------------------------------------------------------------------------------------
def create_dolfin_function(
    function_name: str | Iterable[str],
    function_space: dl.FunctionSpace,
) -> dl.Function:
    element_degree = function_space.ufl_element().degree()
    parameter_expression = dl.Expression(function_name, degree=element_degree)
    parameter_function = dl.Function(function_space)
    parameter_function.interpolate(parameter_expression)
    return parameter_function


# --------------------------------------------------------------------------------------------------
def convert_to_numpy(
    vector: dl.Vector | dl.PETScVector,
    function_space: dl.FunctionSpace,
) -> npt.NDArray[np.floating]:
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
    array: npt.NDArray[np.floating] | Iterable[npt.NDArray[np.floating]],
    function_space: dl.FunctionSpace,
) -> dl.Function:
    dolfin_function = dl.Function(function_space)
    num_components = function_space.num_sub_spaces()
    if num_components <= 1:
        dolfin_function.vector().set_local(array.flatten())
    else:
        for i in range(num_components):
            component_dofs = function_space.sub(i).dofmap().dofs()
            dolfin_function.vector()[component_dofs] = array[i]
    dolfin_function.vector().apply("insert")
    return dolfin_function


# --------------------------------------------------------------------------------------------------
def convert_multivector_to_numpy(
    multivector: hl.MultiVector, function_space: dl.FunctionSpace
) -> Iterable[npt.NDArray[np.floating]]:
    num_vectors = multivector.nvec()
    list_of_arrays = [convert_to_numpy(multivector[i], function_space) for i in range(num_vectors)]
    return list_of_arrays


# --------------------------------------------------------------------------------------------------
def convert_to_multivector(
    list_of_arrays: Iterable[npt.NDArray[np.floating]], function_space: dl.FunctionSpace
) -> hl.MultiVector:
    num_vectors = len(list_of_arrays)
    size_giving_vector = convert_to_dolfin(list_of_arrays[0], function_space).vector()
    multivector = hl.MultiVector(size_giving_vector, num_vectors)
    for i in range(num_vectors):
        multivector[i].set_local(list_of_arrays[i].flatten())
        multivector[i].apply("insert")
    return multivector


# --------------------------------------------------------------------------------------------------
def get_coordinates(
    function_space: dl.FunctionSpace,
) -> npt.NDArray[np.floating]:
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
    for i, component in enumerate(components):
        component_dofs = function_space.sub(i).dofmap().dofs()
        vector[component_dofs] = component.get_local()
    vector.apply("insert")
    return vector


# --------------------------------------------------------------------------------------------------
def convert_matrix_to_scipy(matrix: dl.Matrix | dl.PETScMatrix) -> sp.sparse.coo_array:
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
