import numpy as np
from scipy.special import erf

class Activation:
    def __init__(self):
        self.activation_functions = {
            'sigmoid': self.Sigmoid(),
            'relu': self.ReLU(),
            'softmax': self.Softmax(),
            'tanh': self.Tanh(),
            'elu': self.ELU(),
            'leaky_relu': self.LeakyReLU(),
            'swish': self.Swish(),
            'gelu': self.GELU(),
            'linear': self.Linear(),
            'softplus': self.Softplus()
        }

    def get_activation_function(self, name):
        if name not in self.activation_functions:
            raise ValueError(f"Activation function '{name}' not found")
        return self.activation_functions[name]

    class ActivationFunction:
        def __init__(self):
            self.epsilon = 1e-15

        def forward(self, x):
            pass

        def derivative(self, x):
            pass

    class Sigmoid(ActivationFunction):
        def forward(self, x):
            return 1 / (1 + np.exp(-x))

        def derivative(self, x):
            return x * (1 - x)

    class ReLU(ActivationFunction):
        def forward(self, x):
            return np.maximum(0, x)

        def derivative(self, x):
            return np.where(x > 0, 1, 0)

    class Softmax(ActivationFunction):
        def forward(self, x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        def derivative(self, x):
            # For softmax, we don't actually use this method in the backward pass
            # Instead, we combine it with the loss function derivative
            # This is just a placeholder
            return x

    class Tanh(ActivationFunction):
        def forward(self, x):
            return np.tanh(x)

        def derivative(self, x):
            return 1 - np.tanh(x)**2

    class ELU(ActivationFunction):
        def __init__(self, alpha=1.0):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

        def derivative(self, x):
            return np.where(x > 0, 1, self.alpha * np.exp(x))

    class LeakyReLU(ActivationFunction):
        def __init__(self, alpha=0.01):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return np.maximum(self.alpha * x, x)

        def derivative(self, x):
            return np.where(x > 0, 1, self.alpha)

    class Swish(ActivationFunction):
        def __init__(self, beta=1.0):
            super().__init__()
            self.beta = beta

        def forward(self, x):
            return x * self.sigmoid(self.beta * x)

        def derivative(self, x):
            sigmoid_x = self.sigmoid(self.beta * x)
            return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

    class GELU(ActivationFunction):
        def forward(self, x):
            return 0.5 * x * (1 + erf(x / np.sqrt(2)))

        def derivative(self, x):
            cdf = 0.5 * (1 + erf(x / np.sqrt(2)))
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            return cdf + x * pdf

    class Linear(ActivationFunction):
        def forward(self, x):
            return x

        def derivative(self, x):
            return np.ones_like(x)

    class Softplus(ActivationFunction):
        def forward(self, x):
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

        def derivative(self, x):
            return 1 / (1 + np.exp(-x))
