import torch
from .push_world import *
from .optimization_problem import OptimizationProblem


class Robot3(OptimizationProblem):
    """Robot Pushing 3D implemented in PyTorch

    Details: https://ieeexplore.ieee.org/document/7989109

    Global optimum: :math:`f = 0`
    """

    def __init__(self, gx, gy):
        self.min = 0
        self.dim = 3
        self.gx = gx
        self.gy = gy
        self.lb = torch.tensor([-5, -5, 1], dtype=torch.float32)
        self.ub = torch.tensor([5, 5, 30], dtype=torch.float32)
        self.info = (
                "Negative 3-dimensional Robot pushing problem \nGlobal maximum: "
                + "f = 0"
        )

    def eval(self, x):
        self.__check_input__(x)
        rx, ry, simu_steps = x[:, 0], x[:, 1], x[:, 2] * 10

        # Create the simulation world
        world = b2WorldInterface(False)
        oshape, osize, ofriction, odensity, bfriction = 'circle', 1, 0.01, 0.05, 0.01
        hand_shape, hand_size = 'rectangle', (0.3, 1)
        thing, base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0, 0))

        init_angle = torch.atan2(ry, rx)
        robot = end_effector(world, (rx.item(), ry.item()), base, init_angle.item(), hand_shape, hand_size)

        ret = simu_push(world, thing, robot, base, int(simu_steps.item()))
        ret = torch.tensor(ret, dtype=torch.float32)

        # Calculate distance to goal
        goal = torch.tensor([self.gx, self.gy], dtype=torch.float32)
        distance = torch.linalg.norm(goal - ret, keepdim=True)  # Returns a 2D tensor

        return distance.unsqueeze(-1)

    def __check_input__(self, x):
        """Internal function to check input validity."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input must have shape (n, {self.dim})")
        if torch.any(x < self.lb) or torch.any(x > self.ub):
            raise ValueError("Input values must be within the bounds")


class Robot4(OptimizationProblem):
    """Robot Pushing 4D implemented in PyTorch

    Details: https://ieeexplore.ieee.org/document/7989109

    Global optimum: :math:`f = 0`
    """

    def __init__(self, gx, gy):
        self.min = 0
        self.dim = 4
        self.gx = gx
        self.gy = gy
        self.lb = torch.tensor([-5, -5, 1, 0], dtype=torch.float32)
        self.ub = torch.tensor([5, 5, 30, 2 * torch.pi], dtype=torch.float32)
        self.info = (
                "Negative 4-dimensional Robot pushing problem \nGlobal maximum: "
                + "f = 0"
        )

    def eval(self, x):
        self.__check_input__(x)
        rx, ry, simu_steps, init_angle = x[:, 0], x[:, 1], x[:, 2] * 10, x[:, 3]

        # Create the simulation world
        world = b2WorldInterface(False)
        oshape, osize, ofriction, odensity, bfriction = 'circle', 1, 0.01, 0.05, 0.01
        hand_shape, hand_size = 'rectangle', (0.3, 1)
        thing, base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0, 0))

        xvel, yvel = -rx, -ry
        regu = torch.linalg.norm(torch.stack([xvel, yvel], dim=-1), dim=-1)
        xvel, yvel = xvel / regu * 10, yvel / regu * 10

        robot = end_effector(world, (rx.item(), ry.item()), base, init_angle.item(), hand_shape, hand_size)
        ret = simu_push2(world, thing, robot, base, xvel.item(), yvel.item(), int(simu_steps.item()))
        ret = torch.tensor(ret, dtype=torch.float32)

        # Calculate distance to goal
        goal = torch.tensor([self.gx, self.gy], dtype=torch.float32)
        distance = torch.linalg.norm(goal - ret, keepdim=True)

        return distance.unsqueeze(-1)

    def __check_input__(self, x):
        """Internal function to check input validity."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input must have shape (n, {self.dim})")
        if torch.any(x < self.lb) or torch.any(x > self.ub):
            raise ValueError("Input values must be within the bounds")
