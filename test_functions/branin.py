import torch
from .optimization_problem import OptimizationProblem


class Branin(OptimizationProblem):
    """Branin function implemented in PyTorch

    Details: http://www.sfu.ca/~ssurjano/branin.html

    Global optimum: :math:`f(-\\pi,12.275)=0.397887`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self):
        self.min = 0.397887
        self.minimum = torch.tensor([-torch.pi, 12.275])
        self.dim = 2
        self.lb = -3.0 * torch.ones(2)
        self.ub = 3.0 * torch.ones(2)
        self.int_var = torch.tensor([], dtype=torch.int32)
        self.cont_var = torch.arange(0, 2)
        self.info = (
            "2-dimensional Branin function \nGlobal minimum: "
            + "f(-pi, 12.275) = 0.397887"
        )

    def eval(self, x):
        """Evaluate the Branin function at x

        :param x: Data points
        :type x: torch.Tensor of shape (n, 2)
        :return: Values at x
        :rtype: torch.Tensor of shape (n,)
        """
        self.__check_input__(x)
        x1 = x[:, 0]
        x2 = x[:, 1]

        # Constants for the Branin function
        t = 1 / (8 * torch.pi)
        s = 10
        r = 6
        c = 5 / torch.pi
        b = 5.1 / (4 * torch.pi ** 2)
        a = 1

        # Branin function components
        term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        term2 = s * (1 - t) * torch.cos(x1)
        f = term1 + term2 + s

        return f.unsqueeze(1)

    def __check_input__(self, x):
        """Internal function to check input validity."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input must have shape (n, {self.dim})")
        if torch.any(x < self.lb) or torch.any(x > self.ub):
            raise ValueError("Input values must be within the bounds")
