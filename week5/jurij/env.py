import numpy as np


class Environment:
    def __init__(self, p: float):
        self.p = p
        self.state = 0
        self.valid_actions = ['l', 'r']

    def reset(self):
        self.state = 0

    def game_over(self):
        return True if self.state != 0 else False

    def get_valid_actions(self):
        return self.valid_actions

    def get_valid_states(self):
        return [0]

    def act(self, action):
        if action not in self.valid_actions:
            raise Exception(f'Non valid action, please choose from {self.valid_actions}')

        reward = 0

        if action == 'r':
            self.state = 1

        else:
            if np.random.uniform(0, 1) <= self.p:
                self.state = 1
                reward = 1

        return self.game_over(), reward
