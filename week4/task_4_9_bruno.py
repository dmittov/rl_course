import numpy as np
from enum import Enum
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class Env:
    def __init__(self, p_heads: float = 0.5, goal: int = 100):
        self.p_heads = p_heads
        self.goal = goal

    def get_non_null_state_plus_reward_probabilities(
        self,
        current_state: int,
        action: int,
    ) -> Dict[Tuple[int, int], float]:
        """
        Returns:
            Dictionary where keys are tuples (next-state, reward),
            and values are the corresponding probabilities, considering
            current state and action taken
        """
        if action > current_state:
            raise ValueError("Cannot bet more than what you have")

        state_if_win = int(np.clip(current_state + action, 0, self.goal))
        state_if_lose = int(np.clip(current_state - action, 0, self.goal))
        reward_if_win = 1 if state_if_win == self.goal else 0
        return {
            (state_if_win, reward_if_win): self.p_heads,
            (state_if_lose, 0): 1 - self.p_heads,
        }


class InitializationStrategy(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'
    ZEROS = 'zeros'
    ONES = 'ones'
    ZERO_POINT_FIVES = 'zero_point_fives'


class Agent:
    def __init__(
        self, env: Env, values_init_strategy: InitializationStrategy, discount: float = 0.9
    ):
        self.env = env
        self.values_init_strategy = values_init_strategy
        self.discount = discount

        self.n_states = self.infer_n_states()
        self.state_values = self.initialize_state_values()
        self.optimal_policy = np.zeros((self.n_states, ))  # Mapping from state to action

    def infer_n_states(self) -> int:
        return self.env.goal + 1

    def initialize_state_values(self) -> np.array:
        if self.values_init_strategy == InitializationStrategy.UNIFORM:
            state_values = np.random.random(self.n_states)
        elif self.values_init_strategy == InitializationStrategy.NORMAL:
            state_values = np.random.randn(self.n_states)
        elif self.values_init_strategy == InitializationStrategy.ZEROS:
            state_values = np.zeros(self.n_states)
        elif self.values_init_strategy == InitializationStrategy.ONES:
            state_values = np.ones(self.n_states)
        elif self.values_init_strategy == InitializationStrategy.ZERO_POINT_FIVES:
            state_values = np.ones(self.n_states) * 0.5
        else:
            raise ValueError(f"Unknown initialization strategy {self.values_init_strategy}")

        state_values[0] = 0.0
        state_values[-1] = 1.0
        return state_values

    def value_iteration(self, max_iterations: int = 100, convergence_threshold: float = 1e-3) -> np.array:
        iteration = 0
        current_delta = 1e6  # Just a high number
        while current_delta > convergence_threshold:
            current_delta = 0
            for state in range(1, self.n_states - 1):
                old_value = self.state_values[state]

                new_values_per_action = np.zeros(shape=(self.n_states, ))
                # for action in range(min(state, self.env.goal - state) + 1):
                for action in range(self.n_states):
                    try:
                        non_null_probs_dict = self.env.get_non_null_state_plus_reward_probabilities(state, action)
                    except ValueError:
                        break
                    new_values_per_action[action] = sum([
                        prob * (reward + self.discount * self.state_values[new_state])
                        for (new_state, reward), prob in non_null_probs_dict.items()
                    ])
                best_action = np.argmax(new_values_per_action)
                max_value = new_values_per_action[best_action]
                self.state_values[state] = max_value
                self.optimal_policy[state] = best_action
                current_delta = max(current_delta, np.abs(old_value - self.state_values[state]))
            if iteration == max_iterations:
                print(f"Had to interrupt after {max_iterations} iterations because values didn't "
                      f"converge (delta is {current_delta:.3f})")
                break
            iteration += 1
        else:
            print(f"Values converged after {iteration} iterations!")
        return self.state_values, self.optimal_policy

    def plot_optimal_policies(self, label: str):
        plt.plot(range(len(self.optimal_policy)), self.optimal_policy, label=label)
        # TODO: plot initial state values to see if that has an effect on how the optimal policy converges
        # TODO: plot evolution of optimal policy across iterations


if __name__ == '__main__':
    p_heads = 0.4
    discount = 0.8
    init = InitializationStrategy.ZERO_POINT_FIVES
    environment = Env(p_heads=p_heads)
    agent = Agent(environment, init, discount=discount)
    values, optimal_policies = agent.value_iteration(max_iterations=100, convergence_threshold=1e-5)
    label = f"Init={init.value}, p_heads={p_heads:.2f}, discount={discount:.2f}"
    agent.plot_optimal_policies(label)
    plt.legend()
    plt.show()
