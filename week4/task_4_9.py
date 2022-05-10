from dataclasses import dataclass
from random import random
from numpy.random import uniform
import numpy as np


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
        self.__p = p
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
        diff = stake if uniform(0, 1) < self.__p else -stake
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
    def __init__(self, env):
        self.env = env
        self.states = np.arange(self.env.target + 1)
        self.state_values = np.ones(self.env.target + 1) * 0.5
        # rewards in states
        self.rewards = np.zeros(self.env.target + 1)
        self.rewards[self.env.target] = 1

        # self.greedy_actions = np.ones(self.env.target + 1)

    def value_estimation(self):
        # get all possible actions
        # get all states that can be caused by said actions (2 times as much)
        # weight the value of those states by their probability + reward of state directly
        # use thosew to estimate value of action
        # decide on greedy action and update current value of state
        actions = self.env.get_actions()
        new_states = self.env.bank

        #

    def policy_improvement(self):
        for state in self.states:
            value = self.value_estimation(state)

    def act(self):
        actions = self.env.get_actions()
