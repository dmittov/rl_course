from math import trunc
from typing import List, Dict
from dataclasses import dataclass
from functools import cached_property
import abc
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class State:
    x: int

    def __lt__(self, other):
        return self.x < other.x


@dataclass
class Effect:
    state: State
    reward: float
    is_finished: bool


class Game:
    """
    Game has no state, it is not needed for the task
    """

    @dataclass(frozen=True)
    class __State2d:
        y: int
        x: int

    def __init__(self, sz: int = 4) -> None:
        self.__sz = sz
        self.__n_states = self.__sz**2

    @property
    def grid_size(self):
        return self.__sz

    @cached_property
    def states(self) -> List[State]:
        return {State(x) for x in range(self.__n_states)}

    def is_terminal(self, state: State) -> bool:
        if (state.x < 1) or (state.x > self.__n_states - 2):
            return True
        return False

    def __to_2d(self, state: State) -> __State2d:
        y = state.x // self.__sz
        x = state.x % self.__sz
        return self.__State2d(y, x)

    def __to_1d(self, state: __State2d) -> State:
        return State(state.y * self.__sz + state.x)

    def is_valid(self, state: State) -> bool:
        if (state.x < 0) or (state.x >= self.__n_states):
            return False
        return True

    @property
    def actions(self):
        return list(self.__action_offsets.keys())

    __action_offsets = {
        "UP": [-1, 0],
        "DOWN": [1, 0],
        "LEFT": [0, -1],
        "RIGHT": [0, 1],
    }

    def __get_next_state(self, state: State, action: str) -> State:
        state_2d = self.__to_2d(state)
        diff_y, diff_x = self.__action_offsets[action]
        new_y = state_2d.y + diff_y
        new_x = state_2d.x + diff_x
        new_state = self.__to_1d(self.__State2d(new_y, new_x))
        if not self.is_valid(new_state):
            return state
        return new_state

    def get_actions(self, state: State) -> List[str]:
        _ = state
        valid_actions = []
        for action in self.actions:
            valid_actions.append(action)
        return valid_actions

    def move(self, state: State, action: str) -> Effect:
        new_state = self.__get_next_state(state, action)
        reward = -1
        is_finished = False
        if self.is_terminal(new_state):
            reward = -1
            is_finished = True
        return Effect(new_state, reward, is_finished)


class Policy(abc.ABC):
    @abc.abstractmethod
    def get_probs(self, state: State) -> Dict[str, float]:
        pass


class RandomPolicy:
    def __init__(self, game) -> None:
        self.__game = game

    def get_probs(self, state: State) -> Dict[str, float]:
        valid_actions = self.__game.get_actions(state)
        p = 1.0 / len(valid_actions)
        return {action: p for action in valid_actions}


class StateValueTrainer:
    def __init__(
        self,
        game: Game,
        discount: float = 1.0,
        eps: float = 1e-4,
        max_iter: int = 10_000,
    ) -> None:
        self.__game = game
        self.__state_values = {state: 0 for state in self.__game.states}
        self.__eps = eps
        self.__max_iter = max_iter
        self.__discount = discount

    @property
    def game(self) -> Game:
        return self.__game

    @property
    def state_values(self):
        return self.__state_values

    def __fit_pure(self, policy: Policy) -> None:
        delta = 0.0
        new_state_values = {state: 0 for state in self.__game.states}
        for state in self.__game.states:
            if self.__game.is_terminal(state):
                continue
            expectation = 0.0
            for action, p in policy.get_probs(state).items():
                effect = self.__game.move(state, action)
                expectation += (
                    effect.reward + self.__discount * self.__state_values[effect.state]
                ) * p
            new_state_values[state] = expectation
            delta = max(
                delta, abs(new_state_values[state] - self.__state_values[state])
            )
        self.__state_values = new_state_values
        return delta

    def __fit_sweep(self, policy: Policy) -> None:
        delta = 0.0
        for state in self.__game.states:
            if self.__game.is_terminal(state):
                continue
            expectation = 0.0
            for action, p in policy.get_probs(state).items():
                effect = self.__game.move(state, action)
                expectation += (
                    effect.reward + self.__discount * self.__state_values[effect.state]
                ) * p
            delta = max(delta, abs(expectation - self.__state_values[state]))
            self.__state_values[state] = expectation
        return delta

    def fit(self, policy: Policy) -> None:
        for iter in range(self.__max_iter):
            delta = self.__fit_sweep(policy)
            if delta < self.__eps:
                logger.info("Converged after %d iterations", iter)
                return
        logger.warn("Not converged")


def print_state_values(trainer: StateValueTrainer) -> None:
    for s in sorted(trainer.state_values):
        if (s.x % trainer.game.grid_size) == 0:
            print("\n")
        print(f"{trainer.state_values[s]:.1f}", end="\t")


def main() -> None:
    game = Game(4)
    trainer = StateValueTrainer(game, max_iter=10_000)
    random_policy = RandomPolicy(game)
    trainer.fit(random_policy)
    print_state_values(trainer)


if __name__ == "__main__":
    main()
