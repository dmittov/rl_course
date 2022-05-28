from env import Environment
from agent import Agent
import numpy as np


if __name__ == '__main__':
    env = Environment(0.1)
    agent = Agent(env)
    for i in range(1_000_000):
        agent.play_episode()
        agent.reset()
    print(np.mean(agent.rewards))
    agent.plot_rewards()
