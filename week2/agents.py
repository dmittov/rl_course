import abc
from dataclasses import dataclass
import random
from typing import Tuple
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
    

class UCBAgent(BaseAgent):
    @dataclass
    class Params:
        mu: float
        c: float
        t: int
        N: int
    
    def __init__(self, arms: int):
        super().__init__(arms)
        self.params = [self.Params(0, 2, 1, 1) for _ in range(arms)]

    def get_value(self, action: int) -> float:
        params = self.params[action]
        return params.mu + params.c*np.sqrt(np.log(params.t)/params.N)
    
    def act(self):
        values = []
        for idx in range(self.arms):
            values.append(self.get_value(idx))
        return np.argmax(values)
    
    def update(self, action: int, value: float):
        t = self.params[action].t + 1
        for idx in range(self.arms):
            if idx==action:
                N = self.params[action].N + 1
                mu = self.params[action].mu*(N/(N+1)) + value*(1/(N+1))
                self.params[action] = self.Params(mu, 2, t, N)
            else:
                N = self.params[idx].N
                mu = self.params[idx].mu
                self.params[idx] = self.Params(mu, 2, t, N)
        

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
