import torch
import numpy as np
from .optimization_problem import OptimizationProblem

class Rosenbrock(OptimizationProblem):
    """Rosenbrock function for n×dim inputs, implemented with PyTorch

    .. math::
        f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ 100(x_i^2 - x_{i+1})^2 + (1 - x_i)^2 \right]

    subject to

    .. math::
        -2.048 \leq x_i \leq 2.048

    Global optimum: :math:`f(1, 1, ..., 1) = 0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=14):
        self.dim = dim
        self.min = 0
        self.minimum = torch.ones(dim)
        self.lb = -2.048 * torch.ones(dim)
        self.ub = 2.048 * torch.ones(dim)
        self.int_var = torch.tensor([])  # No integer variables
        self.cont_var = torch.arange(0, dim)  # Continuous variables
        self.info = f"{dim}-dimensional Rosenbrock function \n" + "Global optimum: f(1,1,...,1) = 0"

    def eval(self, x):
        """Evaluate the Rosenbrock function for each row of x (n×dim input)

        :param x: Data points (shape: n×dim, where n is the number of data points and dim is the dimension)
        :type x: torch.Tensor
        :return: Value for each data point in x (shape: n,)
        :rtype: torch.Tensor
        """

        total = 0.0
        for i in range(self.dim - 1):
            total += 100 * (x[:, i]**2 - x[:, i + 1])**2 + (1 - x[:, i])**2

        return total[:, None]
