import dolfin as dl
import ufl


# ==================================================================================================
def weak_form_mean_exit_time(
    forward_variable: ufl.Argument | dl.Function,
    adjoint_variable: ufl.Argument | dl.Function,
    drift: ufl.Coefficient | ufl.tensors.ListTensor,
    squared_diffusion: ufl.tensors.ListTensor,
) -> ufl.Form:
    weak_form = (
        dl.dot(drift * adjoint_variable, dl.grad(forward_variable)) * dl.dx
        - 0.5
        * dl.dot(dl.div(squared_diffusion * adjoint_variable), dl.grad(forward_variable))
        * dl.dx
        + dl.Constant(1) * adjoint_variable * dl.dx
    )
    return weak_form


# --------------------------------------------------------------------------------------------------
def weak_form_mean_exit_time_moments(
    forward_variable: ufl.Argument | dl.Function,
    adjoint_variable: ufl.Argument | dl.Function,
    drift: ufl.Coefficient | ufl.tensors.ListTensor,
    squared_diffusion: ufl.tensors.ListTensor,
) -> ufl.Form:
    weak_form_component_1 = (
        dl.dot(drift * adjoint_variable[0], dl.grad(forward_variable[0])) * dl.dx
        - 0.5
        * dl.dot(dl.div(squared_diffusion * adjoint_variable[0]), dl.grad(forward_variable[0]))
        * dl.dx
        + dl.Constant(1) * adjoint_variable[0] * dl.dx
    )
    weak_form_component_2 = (
        dl.dot(drift * adjoint_variable[1], dl.grad(forward_variable[1])) * dl.dx
        - 0.5
        * dl.dot(dl.div(squared_diffusion * adjoint_variable[1]), dl.grad(forward_variable[1]))
        * dl.dx
        + 2 * forward_variable[0] * adjoint_variable[1] * dl.dx
    )
    weak_form = weak_form_component_1 + weak_form_component_2
    return weak_form


# --------------------------------------------------------------------------------------------------
def weak_form_fokker_planck(
    forward_variable: ufl.Argument | dl.Function,
    adjoint_variable: ufl.Argument | dl.Function,
    drift: ufl.Coefficient | ufl.tensors.ListTensor,
    squared_diffusion: ufl.tensors.ListTensor,
) -> ufl.Form:
    weak_form = (
        dl.div(drift * forward_variable) * adjoint_variable * dl.dx
        + 0.5
        * dl.dot(dl.div(squared_diffusion * forward_variable), dl.grad(adjoint_variable))
        * dl.dx
        + dl.Constant(0) * adjoint_variable * dl.dx
    )
    return weak_form
