from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from wrappers import Episode


class ApproxQLearningNetwork(nn.Module):
    def __init__(self, n_features: int, n_actions: int, hidden_dimensions: int):
        super().__init__()
        self.linear_one = nn.Linear(in_features=n_features, out_features=24)
        self.linear_two = nn.Linear(in_features=24, out_features=48)

        self.output = nn.Linear(in_features=48, out_features=n_actions)

    def forward(self, batch):
        x = batch

        x = self.linear_one(x)
        x = torch.tanh(x)

        x = self.linear_two(x)
        x = torch.tanh(x)

        # Q-values
        x = self.output(x)
        return x


class ApproxQAgent:
    def __init__(self, network, optimizer, eps=0.1, gamma=0.95):
        self.eps = eps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = network.to(self.device)
        self.optimizer = optimizer
        self.gamma = gamma

    def act(self, state, greedy=True):
        if not greedy and np.random.random() < self.eps:
            action_probs = np.ones(2) * (1.0 / 2)
            action = np.random.choice(2, 1, replace=False, p=action_probs)[0]
            return action
        t_state = torch.Tensor(state)
        t_action_values = self.network(t_state)
        action_values = t_action_values.cpu().detach().numpy()
        action = action_values.argmax()
        return action

    def __get_current_q_values(self, batch: List[Episode]):
        states = np.vstack([episode.state for episode in batch])
        actions = np.array([episode.action for episode in batch])
        t_states = torch.Tensor(states)
        # unsqueeze(1) - make a column vector from a row
        t_actions = torch.Tensor(actions).long().unsqueeze(1)
        # [episode_id, action_value]
        t_current_q_values = self.network(t_states).gather(1, t_actions)
        return t_current_q_values

    def __get_experince_q_values(self, batch: List[Episode]):
        next_states = np.vstack([episode.next_state for episode in batch])
        t_next_states = torch.Tensor(next_states)
        rewards = np.array([episode.reward for episode in batch])
        t_rewards = torch.Tensor(rewards)
        # multipy mask
        is_done = np.array([episode.is_done for episode in batch])
        t_is_done = torch.Tensor(is_done)
        # [episode_id, action_value]
        # max(1) - max in each row
        # max returns values of max elements and indices
        t_best_next_action_q_value = (
            self.gamma * self.network(t_next_states).max(1).values
        )
        # some states may be terminal states, multiply by is_done mask to make them zero
        t_best_next_action_q_value = t_best_next_action_q_value * (1 - t_is_done)
        # (6.8)
        t_target = t_best_next_action_q_value + t_rewards
        # make a column from a row
        t_target = t_target.unsqueeze(1)
        return t_target

    def train_on_batch(self, batch: List[Episode]):
        self.optimizer.zero_grad()
        t_current_q_values = self.__get_current_q_values(batch)
        t_experience_q_values = self.__get_experince_q_values(batch)
        t_loss = F.mse_loss(t_experience_q_values, t_current_q_values)
        t_loss.backward()
        self.optimizer.step()
        loss = float(t_loss.cpu().detach().numpy())
        return loss
