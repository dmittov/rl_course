import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from gridworld import windy_grid
from grid_agent import GridAgent

x = 6
y = 5
start = (6, 2)
base_epsilon = 0.2
alpha = 0.4
gamma = 0.9

grid = windy_grid(x, y, start)
agent = GridAgent(grid, base_epsilon, alpha, gamma)

for i in range(1, 5000):
    terminal = False
    grid.reset()

    # comment out this line if you want a constant epsilon
    # agent.eps = min(1, 10 * base_epsilon/i)
    while not terminal:
        agent.act()
        terminal = grid.game_over()

results = np.zeros(shape=(x + 1, y + 1))
labels = np.empty(shape=(x + 1, y + 1), dtype=str)

for state, actions in agent.Q.items():
    x = state[0]
    y = state[1]
    results[x, y] = max(actions.values())
    labels[x, y] = agent.get_best_action(state)

terminal_state = grid.current_state()
labels[terminal_state[0], terminal_state[1]] = None
results[start] = None

sns.heatmap(results, annot=labels, fmt="", linewidths=0.1)
plt.show()
