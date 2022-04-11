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
        self.__last_action = None
        self.__is_random_fallback = False

    @property
    def match_boxes(self):
        return self.__match_boxes

    def get_actions(self, state):
        return self.__match_boxes.get(state, None)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.__match_boxes, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            self.__match_boxes = pickle.load(fh)

    def loose(self):
        if self.__last_action:
            state, action = self.__last_action
            self.__match_boxes[state].remove(action)

    def reset(self):
        self.__last_action = None
        self.__is_random_fallback = False

    def act(self, state, actions) -> int:
        if self.__is_random_fallback:
            return random.choice(actions)
        if state not in self.__match_boxes:
            self.__match_boxes[state] = set(actions)
        # no allowed actions
        if not self.__match_boxes[state]:
            state, action = self.__last_action
            self.__match_boxes[state].remove(action)
            self.__last_action = None
            self.__is_random_fallback = True
            return random.choice(actions)
        action = random.choice(list(self.__match_boxes[state]))
        self.__last_action = (state, action)
        return action
