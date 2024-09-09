import numpy as np

class Cost:
    def __init__(self):
        self.cost_functions = {
            'mse': self.MSE(),
            'cross_entropy': self.CrossEntropy()
        }

    def get_cost_function(self, name):
        if name not in self.cost_functions:
            raise ValueError(f"Cost function '{name}' not found")
        return self.cost_functions[name]

    class CostFunction:
        def loss(self, y_true, y_pred):
            pass

        def derivative(self, y_true, y_pred):
            pass

    class MSE(CostFunction):
        def loss(self, y_true, y_pred):
            return np.mean(np.square(y_true - y_pred))

        def derivative(self, y_true, y_pred):
            return 2 * (y_pred - y_true) / y_true.shape[0]

    class CrossEntropy(CostFunction):
        def loss(self, y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

        def derivative(self, y_true, y_pred):
            # This derivative is combined with Softmax,
            # so we can simply return y_pred - y_true
            return y_pred - y_true