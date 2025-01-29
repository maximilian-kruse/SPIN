import dolfin as dl

from collections.abc import Iterable


# ----------------------------------------------------------------------------------------------
def create_dolfin_function(
    function_name: str | Iterable[str] | Iterable[Iterable[str]],
    function_space: dl.FunctionSpace,
) -> dl.Function:
    element_degree = function_space.ufl_element().degree()
    parameter_expression = dl.Expression(function_name, degree=element_degree)
    parameter_function = dl.Function(function_space)
    parameter_function.interpolate(parameter_expression)
    return parameter_function
