from collections.abc import Iterable
from typing import Annotated

import dolfin as dl
import hippylib as hl
import numpy as np
import numpy.typing as npt
from beartype.vale import Is


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
    array: npt.NDArray[np.floating], function_space: dl.FunctionSpace
) -> dl.Function:
    dolfin_function = dl.Function(function_space)
    dolfin_function.vector().set_local(array.flatten())
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
    return np.flip(coordinates, axis=0)
