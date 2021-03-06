from typing import MutableSet, List, final, TypeVar, Generic
import numpy as np
from wrappers import StepResult
from agents import BaseAgent, OffPolicyMCAgent
import abc

Agent = TypeVar("Agent", bound=BaseAgent)


class BaseTrainer(abc.ABC, Generic[Agent]):
    """Base Trainer accepts Agent and uses episode logs (aka game replays)
    to update agent action values"""

    @abc.abstractmethod
    def update(self, agent: Agent, steps: List[StepResult]) -> MutableSet[int]:
        """Update action values, return set of updated states"""
        pass


class MCControlTrainer(BaseTrainer[OffPolicyMCAgent]):
    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    @final
    def update(
        self, agent: OffPolicyMCAgent, steps: List[StepResult]
    ) -> MutableSet[int]:
        """
        step = (reward, state, action)
        """
        # FIXME: not checked
        changed_states = set()
        G = 0.0
        rho = 1.0
        for step in steps[::-1]:
            state, action, reward, _, _ = step
            G = G * self.gamma + reward
            agent.weights[state, action] += rho
            w = rho / agent.weights[state, action]
            val = G - agent.action_values[state, action]
            prev_target_action = np.argmax(agent.action_values[state])
            agent.action_values[state, action] += w * val
            if np.argmax(agent.action_values[state]) != prev_target_action:
                changed_states.add(state)
            if np.argmax(agent.action_values[state]) != state:
                break
            rho /= agent.behavioral_policy[state, action]
            if rho == 0:
                break
        return changed_states
        # TODO: stop condition (?)
