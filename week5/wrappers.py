from typing import Union, Tuple, List
from gym import Env, Wrapper
from gym.core import ObsType, ActType
from functools import cached_property
from gym import ObservationWrapper, Env
import numpy as np
import abc
from agents import BaseAgent

# observation, action, reward, is_done, info
StepResult = Tuple[ObsType, ActType, float, bool, dict]


class BaseDiscretizer(abc.ABC):
    @abc.abstractproperty
    def n_states(self):
        pass

    @abc.abstractmethod
    def discretize(self, observation: ObsType) -> int:
        pass

    @abc.abstractmethod
    def restore(self, state: int) -> ObsType:
        pass


class BlackjackDiscretizer(BaseDiscretizer):
    @cached_property
    def n_states(self):
        return (21 - 4 + 1) * 10 * 2

    def discretize(self, observation: ObsType) -> int:
        # players_sum: 4..21
        # dealers_value: 1..10
        # players_ace
        players_sum, dealers_value, has_ace = observation
        has_ace = 1 if has_ace else 0
        player = 18 * has_ace + (players_sum - 4)  # 0..35
        state = 36 * (dealers_value - 1) + player
        return state

    def restore(self, state: int) -> ObsType:
        player = state % 36
        dealers_value = (state // 36) + 1
        players_sum = (player % 18) + 4
        has_ace = player // 18
        return (players_sum, dealers_value, has_ace)


class MountainCarDiscretizer(BaseDiscretizer):
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

    def discretize(self, observation: ObsType) -> int:
        position, velocity = observation
        discrete_position = int((position - self.min_position) / self.position_step)
        discrete_velocity = int((velocity - self.min_velocity) / self.velocity_step)
        state = discrete_position * self.velocity_buckets + discrete_velocity
        return state

    def restore(self, state: int) -> ObsType:
        discrete_position = state // self.velocity_buckets
        restored_position = self.min_position + self.position_step * discrete_position
        discrete_velocity = state % self.velocity_buckets
        restored_velocity = self.min_velocity + self.velocity_step * discrete_velocity
        return np.array([restored_position, restored_velocity])


def get_discrete_policy(
    discretizer: BaseDiscretizer, agent: BaseAgent, eps: float
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
    def __init__(self, env: Env, discretizer: BaseDiscretizer) -> None:
        super(ObservationWrapper, self).__init__(env)
        self.__discretizer = discretizer

    def observation(self, observation: np.ndarray) -> int:
        return self.__discretizer.discretize(observation)


class ActionLogger(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.__log: List[StepResult] = []
        self.__prev_obeservation = None

    @property
    def log(self) -> List[StepResult]:
        return self.__log

    def step(self, action: ObsType) -> StepResult:
        result = self.env.step(action)
        observation, reward, is_done, info = result
        log_record = (self.__prev_obeservation, action, reward, is_done, info)
        self.__log.append(log_record)
        self.__prev_obeservation = observation
        return result

    def reset(self, **kwargs) -> Union[ObsType, tuple[ObsType, dict]]:
        self.__log = []
        observation = self.env.reset(**kwargs)
        self.__prev_obeservation = observation
        return observation
