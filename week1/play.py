from gym_tictactoe.env import (
    TicTacToeEnv,
)
from agent import MenaceAgent

# https://github.com/haje01/gym-tictactoe/blob/master/examples/human_agent.py
class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-9], q for quit: ")
            if uloc.lower() == "q":
                return None
            try:
                action = int(uloc) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                break

        return action


def main() -> None:
    agent = MenaceAgent()
    agent.load("agent.dat")
    env = TicTacToeEnv()
    is_done = False
    human_agent = HumanAgent("X")
    env.set_start_mark("X")
    env.reset()
    agent.reset()
    env.render()
    while not is_done:
        action = human_agent.act(env.available_actions())
        state, reward, is_done, _ = env.step(action)
        env.render()
        if is_done:
            if reward < 0:
                print("Human wins")
            else:
                print("Draw")
            exit(0)
        action = agent.act(state, env.available_actions())
        state, reward, is_done, _ = env.step(action)
        env.render()
        if is_done:
            print("Computer wins")


if __name__ == "__main__":
    main()
