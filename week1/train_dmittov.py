from agent import RandomAgent, BaseAgent
from tqdm import tqdm
from gym_tictactoe.env import TicTacToeEnv, X_REWARD
import pickle
import random


def main() -> None:
    pass


class MenaceAgent(BaseAgent):
    def __init__(self):
        self.__match_boxes = dict()
        self.__last_action = None
        self.__is_random_fallback = False

    @property
    def match_boxes(self):
        return self.__match_boxes

    def get_actions(self, state):
        return self.__match_boxes.get(state, None)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.__match_boxes, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            self.__match_boxes = pickle.load(fh)

    def loose(self):
        if self.__last_action:
            state, action = self.__last_action
            self.__match_boxes[state].remove(action)

    def reset(self):
        self.__last_action = None
        self.__is_random_fallback = False

    def act(self, state, actions) -> int:
        if self.__is_random_fallback:
            return random.choice(actions)
        if state not in self.__match_boxes:
            self.__match_boxes[state] = set(actions)
        # no allowed actions
        if not self.__match_boxes[state]:
            state, action = self.__last_action
            self.__match_boxes[state].remove(action)
            self.__last_action = None
            self.__is_random_fallback = True
            return random.choice(actions)
        action = random.choice(list(self.__match_boxes[state]))
        self.__last_action = (state, action)
        return action


if __name__ == "__main__":
    agent = MenaceAgent()
    partner_agent = RandomAgent()
    env = TicTacToeEnv()
    env.set_start_mark("X")
    # train
    n_episodes = 100_000
    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        agent.reset()
        is_done = False
        while not is_done:
            partner_agent_action = partner_agent.act(state, env.available_actions())
            state, reward, is_done, info = env.step(partner_agent_action)
            if is_done:
                # partner wins / agent lose
                if reward == X_REWARD:
                    agent.loose()
                continue
            agent_action = agent.act(state, env.available_actions())
            state, reward, is_done, info = env.step(agent_action)
    agent.save("agent.dat")
    # test
    n_wins = 0
    n_loss = 0
    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        is_done = False
        agent.reset()
        while not is_done:
            partner_agent_action = partner_agent.act(state, env.available_actions())
            state, reward, is_done, info = env.step(partner_agent_action)
            if is_done:
                n_loss -= reward
                continue
            agent_action = agent.act(state, env.available_actions())
            state, reward, is_done, info = env.step(agent_action)
            if is_done:
                n_wins += reward
    print(f"Win rate: {n_wins / n_episodes} Loss rate: {n_loss / n_episodes}")
