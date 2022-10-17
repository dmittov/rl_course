import numpy as np
from typing import Iterable
import abc


class InterfaceInitialisation(abc.ABC):
    @abc.abstractstaticmethod
    def init(size: tuple) -> np.ndarray:
        pass


class NormalInitialisation(InterfaceInitialisation):
    @staticmethod
    def init(size: tuple) -> np.ndarray:
        return np.random.normal(0, 1, size)


class Parameter:
    # pytorch stored grad functions, we'll store grad values
    def __init__(self, 
                 size: tuple, 
                 winit: InterfaceInitialisation = NormalInitialisation) -> None:
        self.w = winit.init(size)
        self.grad = np.zeros(self.w.shape)
    

class Layer(abc.ABC):

    @abc.abstractproperty
    def parameters(self) -> Iterable[Parameter]:
        pass

    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, error: np.ndarray) -> np.ndarray:
        pass

class Sigmoid(Layer):

    @property
    def parameters(self) -> Iterable[Parameter]:
        return []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1. / (1. + np.exp(-x))
        return self.y

    def backward(self, error: np.ndarray) -> np.ndarray:
        return error * self.y * (1 - self.y)


class Linear(Layer):
    
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.w = Parameter((in_features, out_features))
        self.b = Parameter((out_features,))

    @property
    def parameters(self) -> Iterable[Parameter]:
        return [self.w, self.b]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.__batch_sz = x.shape[0]
        y = x @ self.w.w + self.b.w
        return y

    def backward(self, error: np.ndarray) -> np.ndarray:
        self.w.grad = (self.x.T @ error) / self.__batch_sz
        self.b.grad = error.sum(axis=0) / self.__batch_sz
        error = error @ self.w.w.T
        return error


class Network(Layer):
    def __init__(self) -> None:
        self.layers = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    @property
    def parameters(self) -> Iterable[Parameter]:
        for layer in self.layers:
            for parameter in layer.parameters:
                yield parameter

    def forward(self, x: np.ndarray) -> np.ndarray:
        # self.__in_shape = x.shape
        for layer in self.layers:
            x = layer.forward(x)
        self.__out_shape = x.shape
        return x

    def backward(self, error: np.ndarray) -> np.ndarray:
        error = np.ones(self.__out_shape)
        for layer in self.layers[::-1]:
            error = layer.backward(error)
        return error


class Optimizer(abc.ABC):
    def __init__(self, parameters: Iterable[Parameter]):
        self._parameters = list(parameters)

    def zero_grad(self):
        for parameter in self._parameters:
            parameter.grad = np.zeros(parameter.w.shape)

    @abc.abstractmethod
    def step(self):
        pass
            


class SGD(Optimizer):
    def __init__(self,
                 lr: float,
                 momentum: float,
                 parameters: Iterable[Parameter]) -> None:
        if not (0. <= momentum < 1.):
            raise ValueError("momentum must be in [0, 1)")
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.__log = [
            np.zeros(parameter.grad.shape)
            for parameter in self._parameters
        ]
    
    def zero_grad(self) -> None:
        for parameter in self._parameters:
            parameter.grad = np.zeros(parameter.w.shape)
    
    def step(self) -> None:
        # TODO: add momentum
        for idx, parameter in enumerate(self._parameters):
            parameter.w -= self.lr * parameter.grad
        # for idx, parameter in enumerate(self._parameters):
        #     grad = self.lr * parameter.grad * (1. - self.momentum) + \
        #         self.__log[idx] * self.momentum
        #     self.__log[idx] = grad
        #     parameter.w -= grad


class LogLoss(Layer):

    eps = 1e-31

    def __init__(self, y_true: np.ndarray) -> None:
        super().__init__()
        self.y_true = y_true
    
    @property
    def parameters(self) -> Iterable[Parameter]:
        return []

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.y_true * np.log(x + self.eps) \
            + (1. - self.y_true) * np.log(1. - x + self.eps)

    def backward(self, error: np.ndarray) -> np.ndarray:
        prediction = error
        error = self.y_true / (prediction + self.eps)
        error -= (1. - self.y_true) / (1. - prediction + self.eps)
        return error
