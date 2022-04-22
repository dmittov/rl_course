import abc
from typing import Iterable
import numpy as np
from gym.core import ObsType, ActType


class BaseAgent(abc.ABC):
    @abc.abstractproperty
    def action_space(self) -> Iterable[ActType]:
        pass

    @abc.abstractmethod
    def act(self, observation: ObsType) -> ActType:
        pass


class MountainCarAgent(BaseAgent):
    @property
    def action_space(self) -> Iterable[ActType]:
        return np.array([0, 1, 2])


class BlackjackAgent(BaseAgent):
    @property
    def action_space(self) -> Iterable[ActType]:
        return np.array([0, 1])


class OrininalSmartAgent(MountainCarAgent):
    # https://zhiqingxiao.github.io/rl-book/html/MountainCar-v0_ClosedForm.html
    def act(self, state: ObsType) -> ActType:
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


class BaseDiscreteAgent(BaseAgent):
    def __init__(self, n_states: int) -> None:
        self.__n_states = n_states

    def n_states(self) -> int:
        return self.__n_states


class DisceteSmartAgent(BaseDiscreteAgent, MountainCarAgent):
    def __init__(self, n_states: int, policy: np.ndarray):
        super().__init__(n_states)
        self.__policy = policy

    def act(self, state: ObsType) -> ActType:
        return np.random.choice(self.action_space, p=self.__policy[state])


class ActionValueAgent(BaseDiscreteAgent):
    def __init__(self, n_states: int) -> None:
        super().__init__(n_states)

        self.action_values = np.array(
            [[0.0 for _ in self.action_space] for _ in range(n_states)]
        )
        self.weights = np.array(
            [[0.0 for _ in self.action_space] for _ in range(n_states)]
        )


class OffPolicyMCAgent(ActionValueAgent):
    def __init__(self, behavioral_policy: np.ndarray) -> None:
        n_states = len(behavioral_policy)
        super().__init__(n_states)
        if len(behavioral_policy) != n_states:
            raise ValueError(
                f"Behavioral policy size ({len(behavioral_policy)}) doesn't match the numer of states ({n_states})"
            )
        self.__behavioral_policy = behavioral_policy

    @property
    def behavioral_policy(self) -> np.array:
        return self.__behavioral_policy

    def behavioral_act(self, state: ObsType) -> ActType:
        action = np.random.choice(self.action_space, p=self.behavioral_policy[state])
        return action

    def act(self, state: ObsType) -> ActType:
        # follow greedy policy on inference
        action = np.argmax(self.action_values[state])
        return action


class BlackjackOffPolicyMCAgent(OffPolicyMCAgent, BlackjackAgent):
    def __init__(self, behavioral_policy: np.ndarray) -> None:
        super().__init__(behavioral_policy)


class MountainCarOffPolicyMCAgent(OffPolicyMCAgent, MountainCarAgent):
    def __init__(self, behavioral_policy: np.ndarray) -> None:
        super().__init__(behavioral_policy)
