import abc
from typing import Iterable, Generic, TypeVar, Type, ClassVar
import numpy as np
from gym.core import ObsType, ActType


class ActionSpace(abc.ABC):
    @property
    @abc.abstractclassmethod
    def action_space(cls) -> Iterable[ActType]:
        pass


class MountainCarActions(ActionSpace):
    @property
    def action_space(self) -> Iterable[ActType]:
        return np.array([0, 1, 2])


class BlackjackActions(ActionSpace):
    @property
    def action_space(self) -> Iterable[ActType]:
        return np.array([0, 1])


TAction = TypeVar("TAction", bound=ActionSpace)


class BaseAgent(Generic[TAction], abc.ABC):
    @property
    def action_storage(self):
        return ClassVar[Type[TAction]]

    @property
    def action_space(self) -> Iterable[ActType]:
        return self.action_storage.action_space

    @abc.abstractmethod
    def act(self, observation: ObsType) -> ActType:
        pass


class OrininalSmartAgent(BaseAgent[MountainCarActions]):
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


class BaseDiscreteAgent(BaseAgent[TAction]):
    def __init__(self, n_states: int) -> None:
        self.__n_states = n_states

    def n_states(self) -> int:
        return self.__n_states


class DisceteSmartAgent(BaseDiscreteAgent[MountainCarActions]):
    def __init__(self, n_states: int, policy: np.ndarray):
        super().__init__(n_states)
        self.__policy = policy

    @property
    def n_states(self) -> int:
        return self.__n_states

    def act(self, state: ObsType) -> ActType:
        return np.random.choice(self.action_space, p=self.__policy[state])


class ActionValueAgent(BaseDiscreteAgent[TAction]):
    def __init__(self, n_states: int) -> None:
        super().__init__(n_states)
        self.action_values = np.array(
            [[0.0 for _ in self.action_space] for _ in range(n_states)]
        )
        self.weights = np.array(
            [[0.0 for _ in self.action_space] for _ in range(n_states)]
        )


class OffPolicyMCAgent(ActionValueAgent[TAction]):
    @abc.abstractproperty
    def behavioral_policy(self) -> np.array:
        pass

    def behavioral_act(self, state: int) -> int:
        action = np.random.choice(self.action_space, p=self.behavioral_policy[state])
        return action

    def act(self, state: int) -> int:
        # follow greedy policy on inference
        action = np.argmax(self.action_values[state])
        return action


class BlackjackOffPolicyMCAgent(OffPolicyMCAgent[BlackjackActions]):
    def __init__(self, behavioral_policy: np.ndarray) -> None:
        self.__behavioral_policy = behavioral_policy
        self.__n_states = len(self.__behavioral_policy)

    @property
    def n_states(self):
        return self.__n_states

    @property
    def behavioral_policy(self):
        return self.__behavioral_policy


# class MountainCarOffPolicyMCAgent(MountainCarActionMixin, OffPolicyMCAgent):
#     pass
