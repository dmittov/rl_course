import numpy as np


class GridAgent:
    def __init__(self, env, eps, alpha, gamma):
        self.Q = self.init_q(env.get_state_actions())
        self.env = env
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

    def init_q(self, state_actions):
        Q = {}
        for state, actions in state_actions.items():
            Q[state] = {}
            for action in actions:
                Q[state][action] = np.random.randn(1)[0]
        return Q

    def act(self):
        # eps greedy policy
        s = self.env.current_state()
        action = self.apply_policy(s)
        r = self.env.move(action)
        s_prime = self.env.current_state()
        q_prime_max_a = 0 if self.env.game_over() else max(self.Q[s_prime].values())

        self.Q[s][action] = self.Q[s][action] + self.alpha * (
            r + self.gamma * q_prime_max_a - self.Q[s][action]
        )

    def apply_policy(self, state):
        if np.random.uniform(0, 1) <= self.eps:
            action = np.random.choice(self.env.get_actions(state))
        else:
            action = self.get_best_action(state)
        return action

    def get_best_action(self, state):
        idx = np.argmax(list(self.Q[state].values()))
        return list(self.Q[state].keys())[idx]
