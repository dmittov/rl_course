import abc
import random


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, state, actions) -> int:
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def load(self, path: str) -> None:
        pass


class RandomAgent(BaseAgent):
    def act(self, state, actions) -> int:
        _ = state
        action = random.choice(actions)
        return action

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write("random agent")

    def load(self, path: str) -> None:
        pass


# Fake Menace
MenaceAgent = RandomAgent
