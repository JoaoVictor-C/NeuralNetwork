import numpy as np
from scipy.special import erf, expit

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
            'softplus': self.Softplus(),
            'softsign': self.Softsign(),
            'mish': self.Mish()
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
            return expit(x)  # Use scipy's expit for numerical stability

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
            return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + self.epsilon)

        def derivative(self, x):
            return x

    class Argmax(ActivationFunction):
        def forward(self, x):
            return np.argmax(x, axis=-1)

        def derivative(self, x):
            return np.zeros_like(x) # Argmax is not differentiable, so we return 0

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
            x = np.clip(x, -50, 50)
            return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

        def derivative(self, x):
            x = np.clip(x, -50, 50)
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
            return x * expit(self.beta * x)

        def derivative(self, x):
            sigmoid_x = expit(self.beta * x)
            return sigmoid_x + x * self.beta * sigmoid_x * (1 - sigmoid_x)

    class GELU(ActivationFunction):
        def forward(self, x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

        def derivative(self, x):
            cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            return cdf + x * (1 - cdf**2) * (0.5 * np.sqrt(2 / np.pi) * (1 + 0.134145 * x**2))

    class Linear(ActivationFunction):
        def forward(self, x):
            return x

        def derivative(self, x):
            return np.ones_like(x)

    class Softplus(ActivationFunction):
        def forward(self, x):
            return np.logaddexp(0, x)

        def derivative(self, x):
            return 1 / (1 + np.exp(-x))

    class Softsign(ActivationFunction):
        def forward(self, x):
            return x / (1 + np.abs(x))

        def derivative(self, x):
            return 1 / (1 + np.abs(x))**2

    class Mish(ActivationFunction):
        def forward(self, x):
            return x * np.tanh(np.log(1 + np.exp(x)))

        def derivative(self, x):
            omega = 4 * (x + 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x + 6)
            delta = 2 * np.exp(x) + np.exp(2 * x) + 2
            tanh_softplus = np.tanh(np.log(1 + np.exp(x)))
            return np.exp(x) * omega / (delta ** 2) * tanh_softplus + tanh_softplus