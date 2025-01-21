import torch
import numpy as np


class Griewank:
    """Griewank function implemented using PyTorch

    .. math::
        f(x_1,\ldots,x_n) = 1 + \frac{1}{4000} \sum_{j=1}^n x_j^2 - \
        \prod_{j=1}^n \cos \left( \frac{x_i}{\sqrt{i}} \right)

    subject to

    .. math::
        -512 \leq x_i \leq 512

    Global optimum: :math:`f(0,0,...,0)=0`

    :ivar dim: Number of dimensions
    :ivar bounds: Variable bounds (tuple of lower and upper bounds)
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=5):
        self.dim = dim
        self.min = 0
        self.minimum = torch.zeros(dim, dtype=torch.float32)
        self.lb = -512 * torch.ones(dim, dtype=torch.float32)
        self.ub = 512 * torch.ones(dim, dtype=torch.float32)
        self.int_var = torch.tensor([], dtype=torch.long)
        self.cont_var = torch.arange(0, dim, dtype=torch.long)
        self.info = f"{dim}-dimensional Griewank function \nGlobal optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Griewank function at x.

        :param x: Data point
        :type x: torch.Tensor
        :return: Value at x
        :rtype: torch.Tensor
        """
        x = x.to(dtype=torch.float32)
        self.__check_input__(x)
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, 1), device=x.device, dtype=torch.float32)

        total = torch.sum(x ** 2, dim=1) / 4000
        indices = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=torch.float32)
        prod = torch.prod(torch.cos(x / torch.sqrt(indices)), dim=1)

        f[:, 0] = total - prod + 1

        # Adding a small random noise for stochasticity
        # f += torch.sqrt(torch.tensor(0.0001, device=x.device, dtype=torch.float32)) * torch.randn_like(f)
        return f

    def __check_input__(self, x):
        """Check if the input tensor is valid."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input must have {self.dim} dimensions, got {x.shape[1]} instead.")
        if torch.any(x < self.lb) or torch.any(x > self.ub):
            raise ValueError("Input is out of bounds.")
