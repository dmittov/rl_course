from dataclasses import dataclass
from random import random
from numpy.random import uniform
import numpy as np
import itertools


@dataclass
class State:
    reward: int
    is_done: bool
    bank: int


class Environment:
    def __init__(self, bank: int, target: int, p: float) -> None:
        # players bank
        self.__init_bank = bank
        self.__target = target
        self.p = p
        self.reset()

    @property
    def target(self) -> int:
        return self.__target

    def reset(self) -> State:
        self.bank = self.__init_bank
        return State(reward=0, is_done=False, bank=self.bank)

    def step(self, stake: int) -> State:
        if (stake > self.bank) or (stake < 1):
            raise ValueError("Invalid stake")
        diff = stake if uniform(0, 1) < self.p else -stake
        self.bank += diff
        state = State(reward=0, is_done=False, bank=self.bank)
        if self.bank >= self.__target:
            state.is_done = True
            state.reward = 1
        elif self.bank <= 0:
            state.is_done = True
            state.reward = 0
        return state

    def get_actions(self):
        state = self.bank
        actions = [*range(1, state + 1)]
        return actions


class Agent:

    def __init__(self, env, **kwargs):
        self.env = env
        self.states = np.arange(self.env.target + 1)
        self.state_values = uniform(0, 1, self.env.target + 1)
        self.greedy_actions = kwargs.get("greedy_actions", np.ones(self.env.target + 1).astype(int))
        self.eps = kwargs.get("eps", 0.1)
        self.gamma = kwargs.get("gamma", 1.)

    def estimate_action(self, bank: int, stake: int) -> float:
        v_win = 0.
        v_lose = 0.
        if (bank + stake) >= self.env.target:
            v_win = 1.
        else:
            v_win = self.gamma * self.state_values[bank + stake]
        if (bank - stake) <= 0:
            v_lose = 0.
        else:
            v_lose = self.gamma * self.state_values[bank - stake]
        action_value = (self.env.p * v_win) + ((1. - self.env.p) * v_lose)
        return action_value

    def estimate_states(self, eps: float) -> float:
        # non-terminal states
        diff = 0.
        for state in self.states[1:-1]:
            # bet [1 .. state]
            G = 0.
            possible_actions = list(range(1, state + 1))
            for stake in possible_actions:
                # value of action
                action_value = self.estimate_action(state, stake)
                action_prob = (1. - eps) if stake == self.greedy_actions[state] else (eps / (len(possible_actions) - 1))
                G += action_value * action_prob
            diff = max(abs(self.state_values[state] - G), diff)
            self.state_values[state] = G
        return diff

    def improve_policy(self) -> bool:
        has_improvement = False
        for state in self.states[1:-1]:
            best_action = None
            current_estimation = -1  # any reasonable estimation is >= 0 
            possible_actions = list(range(1, state + 1))
            for stake in possible_actions:
                action_value = self.estimate_action(state, stake)
                if action_value > current_estimation:
                    current_estimation = action_value
                    best_action = stake
            if best_action != self.greedy_actions[state]:
                has_improvement = True
                self.greedy_actions[state] = best_action
        return has_improvement


    def fit(self) -> None:
        max_iter = 10_000
        for it in range(max_iter):
            for idx in itertools.count():
                if self.estimate_states(self.eps) <= 1e-6:
                    # print(f"steps to estimate state values: {idx}")
                    break
            if not self.improve_policy():
                # print(f"policy was not improved")
                if self.eps < 1e-9:
                    break
                if self.eps < 1e-6:
                    self.eps = 0.
                self.eps /= 10

    def act(self, bank: int) -> int:
        return self.greedy_actions[bank]
