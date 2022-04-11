import abc
import random
import pickle


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


class MenaceAgent(BaseAgent):
    def __init__(self):
        self.__match_boxes = dict()

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.__match_boxes, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            self.__match_boxes[path] = pickle.load(fh)

    def act(self, state, actions) -> int:
        pass
