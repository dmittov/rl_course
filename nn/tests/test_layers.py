import pytest
from numpy_nn import *
from torch import nn
import torch


class TestNormalization:
    def test_2d_smoke(self):
        size = (128, 10)
        w = NormalInitialisation.init(size)
        assert w.shape == size

    def test_1d_smoke(self):
        size = (128,)
        w = NormalInitialisation.init(size)
        assert w.shape == size


class TestSigmoid:
    def test_zero(self):
        sigmoid = Sigmoid()
        result = sigmoid.forward(np.array([0.0]))
        expected = np.array([0.5])
        assert np.isclose(expected, result)


class TestLinear:
    @pytest.fixture
    def in_features(self) -> int:
        return 4

    @pytest.fixture
    def out_features(self) -> int:
        return 3

    @pytest.fixture
    def batch_size(self) -> int:
        return 12

    @pytest.fixture
    def linear(self, in_features: int, out_features: int) -> Linear:
        return Linear(in_features, out_features)

    def test_b_dimension(self, batch_size, linear):
        x = np.ones((batch_size, linear.in_features))
        _ = linear.forward(x)
        error = np.ones((batch_size, linear.out_features))
        _ = linear.backward(error)
        assert linear.b.w.shape == (linear.out_features,)

    def test_w_grad_dimension_matches(self, batch_size, linear):
        x = np.ones((batch_size, linear.in_features))
        _ = linear.forward(x)
        error = np.ones((batch_size, linear.out_features))
        _ = linear.backward(error)
        assert linear.w.w.shape == linear.w.grad.shape

    def test_b_grad_dimension_matches(self, batch_size, linear):
        x = np.ones((batch_size, linear.in_features))
        _ = linear.forward(x)
        error = np.ones((batch_size, linear.out_features))
        _ = linear.backward(error)
        assert linear.b.w.shape == linear.b.grad.shape


class TestLogLoss:
    def test_smoke(self):
        y_true = np.array([1, 0]).astype(np.float32)
        loss = LogLoss()
        y_pred = np.array([1.0, 0.0]).astype(np.float32)
        assert np.isclose(loss.loss(y_pred, y_true), 0.0)

    def test_correct_bce(self):
        sz = 100
        y_true = np.random.choice([0, 1], size=sz)
        y_pred = np.random.uniform(low=0.0, high=1.0, size=sz)
        expected_loss = float(
            nn.BCELoss()(
                torch.from_numpy(y_pred), torch.from_numpy(y_true.astype(float))
            )
            .detach()
            .numpy()
        )
        actual_loss = LogLoss().loss(y_pred, y_true)
        assert np.isclose(expected_loss, actual_loss)
