import gym
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Episode:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    is_done: float


class Recorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes: List[Episode] = []
        self.state = None

    def get_record(self) -> List[Episode]:
        return self.episodes

    def reset(self):
        self.state = self.env.reset()
        self.episodes = []
        return self.state

    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        episode = Episode(
            state=self.state,
            action=action,
            next_state=next_state,
            reward=reward,
            is_done=is_done,
        )
        self.episodes.append(episode)
        self.state = next_state
        return next_state, reward, is_done, info
