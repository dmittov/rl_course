import abc
import numpy as np
from functools import cached_property


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, state: np.ndarray) -> int:
        pass


class OrininalSmartAgent(BaseAgent):
    # https://zhiqingxiao.github.io/rl-book/html/MountainCar-v0_ClosedForm.html
    def act(self, state: np.ndarray) -> int:
        position, velocity = state
        lb = min(
            -0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008
        )
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2  # push right
        else:
            action = 0  # push left
        return action


class DisceteSmartAgent(OrininalSmartAgent):
    # game properties: https://github.com/openai/gym/wiki/MountainCar-v0
    min_position = -1.2
    max_position = 0.6
    min_velocity = -0.07
    max_velocity = 0.07

    def __init__(self, velocity_buckets: int, position_buckets: int):
        # discretization options
        self.velocity_buckets = velocity_buckets
        self.position_buckets = position_buckets

    @cached_property
    def velocity_step(self):
        return (self.max_velocity - self.min_velocity) / self.velocity_buckets

    @cached_property
    def position_step(self):
        return (self.max_position - self.min_position) / self.position_buckets

    def act(self, state: np.ndarray) -> int:
        position, velocity = state
        discrete_position = int((position - self.min_position) / self.position_step)
        restored_position = discrete_position * self.position_step + self.min_position
        discrete_velocity = int((velocity - self.min_velocity) / self.velocity_step)
        restored_velocity = discrete_velocity * self.velocity_step + self.min_velocity
        return super().act(np.array([restored_position, restored_velocity]))
