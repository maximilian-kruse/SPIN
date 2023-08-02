""" Variational form module

This module resembles a database of weak formulations for different generating PDEs governing an
inference problem. It can be easily extended by defining a new method in the class below. The form
setup follows the notation of the FEniCS library. In addition, every form callable takes 4 arguments:
Forward, drift, diffusion and adjoint/test function.
After a method is defined, it is directly accessibly through the respective method name. For transient
PDEs, the form methods implement the stationary portion. It is then assumed that the transient form
is a superposition of the stationary operator and the weak form of the time derivative of the
solution variable. Furthermore, the forms are posed for PDE problems with Dirichlet boundary
conditions.

Classes:
--------
VariationalFormHandler: Class holding Callables to implemented variational forms
"""

#====================================== Preliminary Commands =======================================
import warnings
from typing import Callable, Tuple
from abc import ABC, abstractmethod

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe


#==================================== Variational Form Handler =====================================
class VariationalFormHandler(ABC):
    """Form handler class

    This class provides all forms in self-contained and side-effect free class methods. They can 
    therefor be called without instantiation of this class.

    Methods:
        get_form: Returns callable to form specified by string name
        get_option_list: Gets names of all implemented form methods

    NOTE: Other methods are form methods and can be displayed via get_option_list.
    """

    solutionDim = None
  
    #-----------------------------------------------------------------------------------------------
    @abstractmethod
    def form(cls) -> fe.Form:
        pass
        
    @property
    def solutionDim(cls) -> int:
        if VariationalFormHandler._solutionDim is None:
            raise ValueError("Property has not been initialized.")
        return VariationalFormHandler._solutionDim
        

class FokkerPlanckFormHandler(VariationalFormHandler):
    solutionDim = 1

    @classmethod
    def form(cls, forwardVar: fe.Function, driftParam: fe.Function,
             squaredDiffusionParam: fe.Function, adjointVar: fe.Function) -> fe.Form:
        """Fokker Planck operator"""

        weakForm = fe.div(driftParam * forwardVar) * adjointVar * fe.dx \
                 + 0.5 * fe.dot(fe.div(squaredDiffusionParam * forwardVar), fe.grad(adjointVar)) * fe.dx \
                 + fe.Constant(0) * adjointVar * fe.dx

        return weakForm

class MeanExitTimeFormHandler(VariationalFormHandler):
    solutionDim = 1

    @classmethod
    def form(cls, forwardVar: fe.Function, driftParam: fe.Function,
             squaredDiffusionParam: fe.Function, adjointVar: fe.Function) -> fe.Form:
        """Mean exit time operator"""

        weakForm = fe.dot(driftParam * adjointVar, fe.grad(forwardVar)) * fe.dx \
                 - 0.5 * fe.dot(fe.div(squaredDiffusionParam * adjointVar), fe.grad(forwardVar)) * fe.dx \
                 + fe.Constant(1) * adjointVar * fe.dx

        return weakForm
    


def get_form(name: str) -> Tuple[Callable, int]:
    match name:
        case "fokker_planck":
            FormHandler =  FokkerPlanckFormHandler
        case "mean_exit_time":
            FormHandler = MeanExitTimeFormHandler
        case _:
            raise NotImplementedError("The requested form is not implemented.")
        
    return FormHandler.form, FormHandler.solutionDim