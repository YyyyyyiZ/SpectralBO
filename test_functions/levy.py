import torch


class Levy:
    """Levy function implemented using PyTorch

    Details: https://www.sfu.ca/~ssurjano/levy.html

    Global optimum: :math:`f(1,1,...,1)=0`

    :ivar dim: Number of dimensions
    :ivar bounds: Variable bounds (2 x dim tensor)
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0.0
        self.minimum = torch.ones(dim, dtype=torch.float32)
        self.lb = -5 * torch.ones(dim, dtype=torch.float32)
        self.ub = 5 * torch.ones(dim, dtype=torch.float32)
        self.int_var = torch.tensor([], dtype=torch.long)
        self.cont_var = torch.arange(0, dim, dtype=torch.long)
        self.info = f"{dim}-dimensional Levy function \nGlobal maximum: f(1,1,...,1) = 0"

    def eval(self, x):
        """Evaluate the Levy function at x.

        :param x: Data point
        :type x: torch.Tensor
        :return: Value at x
        :rtype: torch.Tensor
        """
        x = x.to(dtype=torch.float32)
        self.__check_input__(x)
        batch_size = x.shape[0]
        d = self.dim

        # Compute w
        w = 1 + (x - 1.0) / 4.0

        # Compute the function value
        term1 = torch.sin(torch.pi * w[:, 0]) ** 2
        term2 = torch.sum((w[:, 1:d - 1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w[:, 1:d - 1] + 1) ** 2), dim=1,
                          keepdim=True)
        term3 = ((w[:, d - 1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:, d - 1]) ** 2)).unsqueeze(1)

        f = term1.unsqueeze(1) + term2 + term3

        return f

    def __check_input__(self, x):
        """Check if the input tensor is valid."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input must have {self.dim} dimensions, got {x.shape[1]} instead.")
        if torch.any(x < self.lb) or torch.any(x > self.ub):
            raise ValueError("Input is out of bounds.")
