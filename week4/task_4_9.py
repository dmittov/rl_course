import numpy as np


class Environment:
    def __init__(self, target: int, p: float) -> None:
        # players bank
        self.target = target
        self.p = p
        self.terminal_states = [0, target]
        self.rewards = self.set_up_rewards(target)

    def set_up_rewards(self, target):
        rewards = [0] * (target + 1)
        rewards[target] = 1
        return rewards


class Agent:
    def __init__(self, env):
        self.env = env
        self.init_state_action_matrix(env.target)
        self.init_values(env.target)

    def init_state_action_matrix(self, target):
        res = []
        for i in range(1, target):
            row_len = target - i if i > np.floor(target / 2) else i
            row = [0] * row_len
            res.append(row)
        self.sam = res

    def init_values(self, target):
        self.values = [0] * (target + 1)

    def value_update(self):
        prob = self.env.p

        for s, actions in enumerate(self.sam):
            for action, _ in enumerate(actions):
                win_s = (s + 1) + (action + 1)
                loss_s = (s + 1) - (action + 1)
                q_win = prob * (self.values[win_s] + self.env.rewards[win_s])
                q_loss = (1 - prob) * (self.values[loss_s] + self.env.rewards[loss_s])
                self.sam[s][action] = q_win + q_loss
        self.values = self.apply_policy()

    def apply_policy(self):
        values = [np.mean(i) for i in self.sam]
        values.append(0)
        values.insert(0, 0)
        return values
