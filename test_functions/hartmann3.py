import torch


class Hartmann3:
    """Hartmann 3 function implemented using PyTorch

    Details: http://www.sfu.ca/~ssurjano/hart3.html

    Global optimum: :math:`f(0.114614,0.555649,0.852547)=-3.86278`

    :ivar dim: Number of dimensions
    :ivar bounds: Variable bounds (2 x dim tensor)
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self):
        self.dim = 3
        self.lb = torch.zeros(3, dtype=torch.float32)
        self.ub = torch.ones(3, dtype=torch.float32)
        self.int_var = torch.tensor([], dtype=torch.long)
        self.cont_var = torch.arange(0, 3, dtype=torch.long)
        self.min = -3.86278
        self.minimum = torch.tensor([0.114614, 0.555649, 0.852547], dtype=torch.float32)
        self.info = "3-dimensional Hartmann function \nGlobal maximum: f(0.114614,0.555649,0.852547) = -3.86278"

    def eval(self, x):
        """Evaluate the Hartmann 3 function at x

        :param x: Data point
        :type x: torch.Tensor
        :return: Value at x
        :rtype: torch.Tensor
        """
        x = x.to(dtype=torch.float32)
        self.__check_input__(x)

        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.float32)
        A = torch.tensor([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ], dtype=torch.float32)
        P = torch.tensor([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]
        ], dtype=torch.float32) * 1e-4

        f = torch.zeros((x.shape[0], 1), dtype=torch.float32)
        for i in range(4):
            tmp = torch.zeros_like(f)
            for j in range(3):
                tmp += (A[i, j] * (x[:, j] - P[i, j]) ** 2).unsqueeze(1)
            f += alpha[i] * torch.exp(-tmp)

        return -f

    def __check_input__(self, x):
        """Check if the input tensor is valid."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input must have {self.dim} dimensions, got {x.shape[1]} instead.")
        if torch.any(x < self.lb) or torch.any(x > self.ub):
            raise ValueError("Input is out of bounds.")
