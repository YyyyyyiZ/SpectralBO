import torch
from .optimization_problem import OptimizationProblem


class Hartmann6(OptimizationProblem):
    """Hartmann 6 function

    Details: http://www.sfu.ca/~ssurjano/hart6.html

    Global optimum: :math:f(0.201,0.150,0.476,0.275,0.311,0.657)=-3.322

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
        self.min = -3.32237
        self.minimum = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        self.dim = 6
        self.lb = torch.zeros(6)
        self.ub = torch.ones(6)
        self.int_var = torch.tensor([], dtype=torch.long)
        self.cont_var = torch.arange(0, 6)
        self.info = (
            "6-dimensional Hartmann function \nGlobal optimum: "
            + "f(0.2016,0.15001,0.47687,0.27533,0.31165,0.657) = -3.3223"
        )

    def eval(self, x):
        """Evaluate the Hartmann 6 function at x

        :param x: Data point (n × dim), where n is the number of samples
        :type x: torch.Tensor
        :return: Value at x (n × 1)
        :rtype: torch.Tensor
        """
        self.__check_input__(x)

        # Constants
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
        A = torch.tensor(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )
        P = 1e-4 * torch.tensor(
            [
                [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
                [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
            ]
        )

        # Initialize result tensor for n samples (n × 1)
        result = torch.zeros(x.shape[0], dtype=torch.float32)

        # Vectorized computation for each sample in x
        for ii in range(4):
            # Calculate the inner part for all x in batch
            inner = torch.sum(A[ii, :] * (x - P[ii, :]) ** 2, dim=1)
            # Add contribution of this part to the final result
            result += alpha[ii] * torch.exp(-inner)

        return -result[:, None]  # Return as n × 1 tensor


