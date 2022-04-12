import abc
import numpy as np
from functools import cached_property
from typing import Any
from gym import ObservationWrapper, Env


class BaseAgent(abc.ABC):
    @property
    def action_space(self) -> np.ndarray:
        return np.array([0, 1, 2])

    @abc.abstractmethod
    def act(self, observation: Any) -> int:
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


class Discretizer:
    # game properties: https://github.com/openai/gym/wiki/MountainCar-v0
    min_position = -1.2
    max_position = 0.6
    min_velocity = -0.07
    max_velocity = 0.07

    def __init__(self, position_buckets: int, velocity_buckets: int) -> None:
        # discretization options
        self.velocity_buckets = velocity_buckets
        self.position_buckets = position_buckets

    @cached_property
    def n_states(self):
        return self.velocity_buckets * self.position_buckets

    @cached_property
    def velocity_step(self):
        return (self.max_velocity - self.min_velocity) / self.velocity_buckets

    @cached_property
    def position_step(self):
        return (self.max_position - self.min_position) / self.position_buckets

    def discretize(self, observation: np.ndarray) -> int:
        position, velocity = observation
        discrete_position = int((position - self.min_position) / self.position_step)
        discrete_velocity = int((velocity - self.min_velocity) / self.velocity_step)
        state = discrete_position * self.velocity_buckets + discrete_velocity
        return state

    def restore(self, state: int) -> np.ndarray:
        discrete_position = state // self.velocity_buckets
        restored_position = self.min_position + self.position_step * discrete_position
        discrete_velocity = state % self.velocity_buckets
        restored_velocity = self.min_velocity + self.velocity_step * discrete_velocity
        return np.array([restored_position, restored_velocity])


def get_discrete_policy(
    discretizer: Discretizer, agent: BaseAgent, eps: float
) -> np.ndarray:
    policy = []
    for state in range(discretizer.n_states):
        action_probs = []
        restored_observation = discretizer.restore(state)
        greedy_action = agent.act(restored_observation)
        for action in agent.action_space:
            if action == greedy_action:
                action_probs.append(1.0 - eps)
            else:
                action_probs.append(eps / 2)
        policy.append(action_probs)
    return np.array(policy)


class DiscreteWrapper(ObservationWrapper):
    def __init__(self, env: Env, discretizer: Discretizer) -> None:
        super(ObservationWrapper, self).__init__(env)
        self.__discretizer = discretizer

    def observation(self, observation: np.ndarray) -> int:
        return self.__discretizer.discretize(observation)


class BaseDiscreteAgent(BaseAgent):
    def __init__(self, n_states: int):
        self.n_states = n_states

    @abc.abstractmethod
    def act(self, state: int) -> int:
        pass


class DisceteSmartAgent(BaseDiscreteAgent):
    def __init__(self, n_states: int, policy: np.ndarray):
        if (
            (len(policy.shape) != 2)
            or (policy.shape[0] != n_states)
            or (policy.shape[1] != len(self.action_space))
        ):
            raise ValueError("Invalid policy shape")
        super().__init__(n_states)
        self.__policy = policy

    def act(self, state: int) -> int:
        return np.random.choice(self.action_space, p=self.__policy[state])


class OffPolicyMCAgent(BaseDiscreteAgent):
    def __init__(
        self,
        n_states: int,
        behavioral_policy: np.ndarray,
    ):
        if (
            (len(behavioral_policy.shape) != 2)
            or (behavioral_policy.shape[0] != n_states)
            or (behavioral_policy.shape[1] != len(self.action_space))
        ):
            raise ValueError("Invalid behavioral policy shape")
        super().__init__(n_states)
        self.__behavioral_policy = behavioral_policy
        self.__action_values = np.array(
            [[1.0 for _ in self.action_space] for _ in range(n_states)]
        )
        self.__weights = np.array(
            [[0.0 for _ in self.action_space] for _ in range(n_states)]
        )
        self.gamma = 1.0

    def behavioral_act(self, state: int) -> int:
        action = np.random.choice(self.action_space, p=self.__behavioral_policy[state])
        return action

    def act(self, state: int) -> int:
        # follow greedy policy on inference
        action = np.argmax(self.__action_values[state])
        return action

    def update(self, steps: list) -> None:
        """
        step = (reward, state, action)
        """
        # FIXME: not checked
        G = 0.0
        rho = 1.0
        for step in steps[::-1]:
            reward, state, action = step
            G = G * self.gamma + reward
            self.__weights[state, action] += rho
            w = rho / self.__weights[state, action]
            val = G - self.__action_values[state, action]
            self.__action_values[state, action] += w * val
            rho /= self.__behavioral_policy[state, action]
            if rho == 0:
                break
            # TODO: stop condition (?)
