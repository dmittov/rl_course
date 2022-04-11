from agent import RandomAgent, MenaceAgent
from tqdm import tqdm
from gym_tictactoe.env import TicTacToeEnv, X_REWARD


def main() -> None:
    pass


if __name__ == "__main__":
    agent = MenaceAgent()

    env = TicTacToeEnv()
    env.set_start_mark("X")

    partner_agent = RandomAgent()
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
    partner_agent = RandomAgent()
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
