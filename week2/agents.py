import abc
from dataclasses import dataclass
import random
from typing import Tuple, Optional, List
import scipy.stats as stats
import numpy as np


class BaseAgent(abc.ABC):
    def __init__(self, arms: int):
        self.arms = arms

    @abc.abstractmethod
    def act(self):
        pass

    @abc.abstractmethod
    def update(self, idx: int, value: float) -> None:
        pass


class RandomAgent(BaseAgent):
    def __init__(self, arms: int):
        super().__init__(arms)

    def act(self):
        return random.randint(0, self.arms - 1)

    def update(self, idx: int, value: float):
        _ = idx
        _ = value
        return None


class EpsilonGreedy(BaseAgent):
    def __init__(self, arms: int, epsilon: float = 0.1):
        super().__init__(arms)
        self.exploration_rate = epsilon
        self.samples: List[float] = [0.0 for _ in range(arms)]

    def _explore(self, probability) -> bool:
        return random.random() < probability

    def act(self) -> int:
        if self._explore(self.exploration_rate):
            return random.randint(0, self.arms - 1)
        else:
            return np.argmax(self.samples)

    def update(self, idx: int, value: float) -> None:
        self.samples[idx] = value


class OptimisticInitial(EpsilonGreedy):
    def __init__(self, arms: int, epsilon: float = 0.0, alpha: int = 5):
        super().__init__(arms, epsilon)
        self.initial_value = alpha
        self.samples = [self.initial_value for _ in range(arms)]


class ThompsonAgent(BaseAgent):
    """Use Thompson sampling to make decisions"""

    # A simple example
    # https://github.com/dmittov/misc/blob/master/Thompson%20Sampling.ipynb

    # https://en.wikipedia.org/wiki/Conjugate_prior

    # internal data class to store inv-gamma distribution parameters
    @dataclass
    class Params:
        mu: float
        tau2: float
        a: float
        b: float

    def __init__(self, arms: int):
        super().__init__(arms)
        # non informative priors for Inv Gamma distribution
        self.params = [self.Params(0, 0.5, 0.5, 0.5) for _ in range(arms)]

    def act(self):
        samples = []
        for idx in range(self.arms):
            mu, sigma = self.get_sampler_params(idx)
            sample = stats.norm(loc=mu, scale=sigma).rvs(1)[0]
            samples.append(sample)
        return np.argmax(samples)

    def get_sampler_params(self, action: int) -> Tuple[float, float]:
        params = self.params[action]
        # get mu from t-distribution
        mu = stats.t(
            df=2 * params.a,
            loc=params.mu,
            scale=(params.tau2 * (params.b / params.a)) ** 0.5,
        ).mean()
        sigma2 = stats.invgamma(a=params.a, scale=params.b).mean()
        return mu, sigma2**0.5

    def update(self, action: int, value: float) -> None:
        old_params = self.params[action]
        inv_tau2 = 1.0 + (1.0 / old_params.tau2)
        tau2 = 1.0 / inv_tau2
        mu = (old_params.mu / old_params.tau2 + value) * tau2
        a = old_params.a + 0.5
        b_add_numerator = (value - old_params.mu) ** 2
        b_add_denominator = 2 * old_params.tau2 * inv_tau2
        b = old_params.b + b_add_numerator / b_add_denominator
        self.params[action] = self.Params(mu, tau2, a, b)


class GradientAgent(BaseAgent):
    def __init__(self, arms: int, alpha: float):
        super().__init__(arms)
        self.alpha = alpha

        self.H = np.ones((arms,))
        self.mean_R = 0.0
        self.n = 0

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def act(self):
        return np.argmax(self.H)

    def update(self, idx: int, value: float):
        self.n += 1
        probas = self._softmax(self.H)
        self.mean_R = self.mean_R + (value - self.mean_R) / self.n
        for i in range(self.arms):
            if i == idx:
                self.H[i] = self.H[i] + self.alpha * (value - self.mean_R) * (
                    1 - probas[i]
                )
            else:
                self.H[i] = self.H[i] - self.alpha * (value - self.mean_R) * probas[i]


class UCBAgent(BaseAgent):
    """Use Upper confidence bounds to make decisions"""

    @dataclass
    class Params:
        c: float
        N: int
        t: int
        Q: Optional[float] = None

    def __init__(self, arms: int):
        super().__init__(arms)
        self.params = [self.Params(1, 0, 0) for _ in range(arms)]

    def act(self) -> int:
        results = []
        for i, arms in enumerate(self.params):
            if arms.Q is None:
                return i
            else:
                results.append(arms.Q + arms.c * np.sqrt(np.log(arms.t) / arms.N))
        return int(np.argmax(results))

    def update(self, action: int, value: float) -> None:
        for params in self.params:
            params.t += 1

        old_params = self.params[action]
        Q = old_params.Q
        N = old_params.N

        if Q is None:
            self.params[action].Q = value
        else:
            self.params[action].Q = (N / (N + 1)) * Q + (1 / (N + 1)) * value

        self.params[action].N += 1
