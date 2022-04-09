import scipy.stats as stats
from typing import Tuple, Any
from dataclasses import dataclass


class World:
    def __init__(self, k: int):
        mean_rewards = stats.norm(0, 1).rvs(10)
        self.samplers = [stats.norm(mreward, 1) for mreward in mean_rewards]

    def step(self, action: int) -> Tuple[Any, float, bool, Any]:
        reward = self.samplers[action].rvs(1)[0]
        next_state = None
        is_done = True
        info = None
        return next_state, reward, is_done, info
