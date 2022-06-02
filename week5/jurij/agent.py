import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, env, gamma=1):
        self.env = env
        self.gamma = gamma
        self.target_policy = {"l": 1.0, "r": 0.0}
        self.behaviour_policy = {"l": 0.5, "r": 0.5}
        self.reward_scaling = self.get_scaling()
        self.rewards = []

    def get_scaling(self):
        scaling = {}
        for k, v in self.target_policy.items():
            scaling[k] = v / self.behaviour_policy[k]
        return scaling

    def get_action_rho(self, action):
        return self.reward_scaling[action]

    def random_action(self):
        return np.random.choice(list(self.target_policy.keys()))

    def play_episode(self):
        rewards = []
        game_over = False
        while not game_over:
            action = self.random_action()
            game_over, reward = self.env.act(action)
            rewards.append([action, reward])

        gamma = self.gamma ** len(rewards)
        rhos = [self.get_action_rho(i[0]) for i in rewards]
        reward_scaled = (
            np.prod(rhos) * gamma * rewards[-1][-1]
        )  # because only the last state action gives a reward if any
        self.rewards.append(reward_scaled)

    def reset(self):
        self.env.reset()

    def plot_rewards(self):
        y = [*range(1, len(self.rewards) + 1)]
        series = np.cumsum(self.rewards) / y
        plt.plot(series)
        plt.hlines(y=1, xmin=y[0], xmax=y[-1], colors=["black"])
        plt.xscale("log")
        plt.show()
