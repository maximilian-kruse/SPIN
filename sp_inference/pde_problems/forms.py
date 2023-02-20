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
from typing import Callable

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import fenics as fe

#==================================== Variational Form Handler =====================================
class VariationalFormHandler:
    """Form handler class

    This class provides all forms in self-contained and side-effect free class methods. They can 
    therefor be called without instantiation of this class.

    Methods:
        get_form: Returns callable to form specified by string name
        get_option_list: Gets names of all implemented form methods

    NOTE: Other methods are form methods and can be displayed via get_option_list.
    """
    
    #-----------------------------------------------------------------------------------------------
    @classmethod
    def get_form(cls, modelType: str) -> Callable:
        """Returns form callable with name of the input string

        Args:
            modelType (str): Form method name

        Raises:
            TypeError: Checks input string
            AttributeError: Checks if form exists

        Returns:
            Callable: Call handle to form
        """
        if not isinstance(modelType, str):
            raise TypeError("Model type needs to be given as string.")
            
        weakForm = getattr(cls, modelType, None)
        if weakForm is None:
            raise AttributeError("Cannot find given function " + modelType + ". "
                                 "Possible options are: " + ', '.join(cls.get_option_list()))
        
        return weakForm

    #-----------------------------------------------------------------------------------------------
    @classmethod
    def get_option_list(cls) -> list[str]:
        """Returns all options for call handles

        Returns:
            list[str]: List of implemented form options

        NOTE: If you implement a new method in this class that is not a form callable, you need to
              register its name in the exception list below for the display to remain valid.
              Otherwise the corresponding test will fail.
        """
        exceptionList = ['get_form', 'get_option_list']

        optionList = [opt for opt in dir(cls) if opt.startswith('__') is False 
                      and opt not in exceptionList]

        return optionList

    #-----------------------------------------------------------------------------------------------
    @classmethod
    def fokker_planck(cls, forwardVar: fe.Function, driftParam: fe.Function,
                      squaredDiffusionParam: fe.Function, adjointVar: fe.Function) -> fe.Form:
        """Fokker Planck operator"""

        weakForm = fe.div(driftParam * forwardVar) * adjointVar * fe.dx \
                 + 0.5 * fe.dot(fe.div(squaredDiffusionParam * forwardVar), fe.grad(adjointVar)) * fe.dx  \
                 - fe.Constant(0) * adjointVar * fe.dx

        return weakForm

    #-----------------------------------------------------------------------------------------------
    @classmethod
    def mean_exit_time(cls, forwardVar: fe.Function, driftParam: fe.Function,
                       squaredDiffusionParam: fe.Function, adjointVar: fe.Function) -> fe.Form:
        """Mean exit time operator"""

        weakForm = fe.dot(driftParam * adjointVar, fe.grad(forwardVar)) * fe.dx \
                 - 0.5 * fe.dot(fe.div(squaredDiffusionParam * adjointVar), fe.grad(forwardVar)) * fe.dx \
                 + fe.Constant(1) * adjointVar * fe.dx

        return weakForm