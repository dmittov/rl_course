import abc
import numpy as np
from functools import cached_property
import random
from typing import Tuple
from gym import ObservationWrapper, Env


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


class Discretizer(ObservationWrapper):

    DiscreteObservation = Tuple[int, int, int]

    # game properties: https://github.com/openai/gym/wiki/MountainCar-v0
    min_position = -1.2
    max_position = 0.6
    min_velocity = -0.07
    max_velocity = 0.07

    def __init__(self, env: Env, velocity_buckets: int, position_buckets: int) -> None:
        super().__init__(env)
        # discretization options
        self.velocity_buckets = velocity_buckets
        self.position_buckets = position_buckets

    @cached_property
    def velocity_step(self):
        return (self.max_velocity - self.min_velocity) / self.velocity_buckets

    @cached_property
    def position_step(self):
        return (self.max_position - self.min_position) / self.position_buckets

    @property
    def n_states(self):
        return self.velocity_buckets * self.position_buckets

    def observation(self, observation: np.ndarray) -> DiscreteObservation:
        position, velocity = observation
        discrete_position = int((position - self.min_position) / self.position_step)
        discrete_velocity = int((velocity - self.min_velocity) / self.velocity_step)
        state = discrete_position * self.velocity_buckets + discrete_velocity
        restored_position = discrete_position * self.position_step + self.min_position
        restored_velocity = discrete_velocity * self.velocity_step + self.min_velocity
        return state, restored_position, restored_velocity


class DiscreteAgent(abc.ABC):
    def __init__(self, states: int):
        self.__states = states

    @abc.abstractmethod
    def act(self, observation: Discretizer.DiscreteObservation) -> int:
        pass


class DisceteSmartAgent(OrininalSmartAgent, DiscreteAgent):
    def __init__(self, states: int):
        super().__init__(states)

    def act(self, observation: Discretizer.DiscreteObservation) -> int:
        _, position, velocity = observation
        return super().act(np.array([position, velocity]))


def get_behavioral_policy(agent: DiscreteAgent) -> list:
    # TODO: size check
    if 1 == 0:
        raise ValueError("Discretization size mismatch")
    # TODO:
    pass


# class OffPolicyMCAgent(DiscreteAgent):
#     def __init__(
#         self,
#         states: int,
#         behavioral_policy: np.ndarray,
#     ):
#         self.__action_space = [0, 1, 2]
#         # TODO: size check
#         if 1 == 0:
#             raise ValueError("Discretization size mismatch")
#         super().__init__(states)
#         self.__behavioral_policy = behavioral_policy
#         self.__action_values = [
#             [1.0 for _ in self.__action_space] for _ in range(states)
#         ]
#         self.__weights = [[0.0 for _ in self.__action_space] for _ in range(states)]
#         self.__policy = [[1.0 / 3 for _ in self.__action_space] for _ in range(states)]
#         self.gamma = 1.0

#     def compute_policy(self, eps: float) -> None:
#         for state in range(self.states):
#             greedy_action = np.argmax(self.__action_values[state])
#             for action in self.__action_space:
#                 # with 1 - eps probability perform greedy action
#                 # and distribute eps across other options
#                 self.__policy[state][action] = (
#                     1.0 - eps if action == greedy_action else eps / 2
#                 )

#     def behavioral_act(self, observation: Tuple(int, int, int)) -> int:
#         state, _, _ = observation
#         action = np.random.choice(
#             self.__action_space, p=self.__behavioral_policy[state]
#         )
#         return action

#     def act(self, observation: Tuple(int, int, int)) -> int:
#         # follow greedy policy on inference
#         state, _, _ = observation
#         action = np.argmax(self.__action_values[state])
#         return action

#     def update(self, steps: list) -> None:
#         """
#         step = (reward, state, action)
#         """
#         # FIXME: not checked
#         G = 0.0
#         rho = 1.0
#         for step in steps[::-1]:
#             reward, state, action = step
#             position, velocity = state
#             G = G * self.gamma + reward
#             self.__weights[position, velocity, action] += rho
#             w = rho / self.__weights[position, velocity, action]
#             val = G - self.__action_values[position, velocity, action]
#             self.__action_values[position][velocity][action] += w * val
#             rho *= (
#                 self.__policy[position][velocity][action]
#                 / self.__behavioral_policy[position][velocity][action]
#             )
#             if rho == 0:
#                 break
