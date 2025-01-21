import torch
from .optimization_problem import OptimizationProblem


class Exponential(OptimizationProblem):
    """Exponential function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^n e^{jx_j} - \\sum_{j=1} e^{-5.12 j}

    subject to

    .. math::
        -5.12 \\leq x_i \\leq 5.12

    Global optimum: :math:f(0,0,...,0)=0

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = -5.12 * torch.ones(dim)
        self.lb = -5.12 * torch.ones(dim)
        self.ub = 5.12 * torch.ones(dim)
        self.int_var = torch.tensor([], dtype=torch.long)
        self.cont_var = torch.arange(0, dim)
        self.info = str(dim) + "-dimensional Exponential function \n" + "Global optimum: f(-5.12,-5.12,...,-5.12) = 0"

    def eval(self, x):
        """Evaluate the Exponential function at x.

        :param x: Data point (n × dim), where n is the number of samples
        :type x: torch.Tensor
        :return: Value at x (n × 1)
        :rtype: torch.Tensor
        """
        self.__check_input__(x)

        # Constants
        indices = torch.arange(1, self.dim + 1, dtype=torch.float32)  # (dim,)
        constant_term = torch.exp(-5.12 * indices).sum()  # This is a scalar, sum across all dimensions

        # Compute the sum of exponentials for each sample
        exp_part = torch.exp(indices * x)  # (n × dim)
        exp_sum = torch.sum(exp_part, dim=1)  # (n,) sum across the dimensions for each sample

        # Final result (n,)
        result = exp_sum - constant_term  # Subtract constant term from the sum for each sample

        # Return as n × 1 tensor
        return result[:, None]  # Shape is (n, 1)
