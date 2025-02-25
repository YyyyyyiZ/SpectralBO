import abc
from abc import abstractmethod


class OptimizationProblem(object):
    """Base class for optimization problems."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.dim = None
        self.lb = None
        self.ub = None
        self.int_var = None
        self.cont_var = None

    def __check_input__(self, x):
        if x.shape[1] != self.dim:
            raise ValueError("Dimension mismatch")

    @abstractmethod
    def eval(self, record):  # pragma: no cover
        pass
